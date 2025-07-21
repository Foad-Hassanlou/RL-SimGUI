# naf
import os

import cv2
import imageio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from gymnasium.wrappers import RecordVideo

from src.utils import Memory
from src.envs.continuous_envs import shaped_reward_Pendulum

# Base Neural Network Layer
class NeuralNetwork(nn.Module):
    def __init__(self, input_dim, hidden_sizes, activation=nn.ReLU, batch_norm=True):
        super(NeuralNetwork, self).__init__()
        layers = []
        prev_dim = input_dim
        for size in hidden_sizes:
            layers.append(nn.Linear(prev_dim, size))
            if batch_norm:
                layers.append(nn.BatchNorm1d(size))
            layers.append(activation())
            prev_dim = size
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# NAF Network
class NAFNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_sizes, activation=nn.ReLU, batch_norm=True, covariance="identity"):
        super(NAFNetwork, self).__init__()
        self.covariance = covariance
        self.shared = NeuralNetwork(state_dim, hidden_sizes, activation, batch_norm)
        shared_output_dim = hidden_sizes[-1]
        self.V = nn.Linear(shared_output_dim, 1)
        self.mu = nn.Linear(shared_output_dim, action_dim)
        if covariance == "diagonal":
            self.P_diag = nn.Linear(shared_output_dim, action_dim)
        else:
            self.P_diag = None

    def forward(self, state):
        P_diag = None
        shared = self.shared(state)
        V = self.V(shared)
        mu = torch.tanh(self.mu(shared))  # Actions in [-1, 1]
        if self.covariance == "identity":
            P_diag = torch.ones_like(mu)
        elif self.covariance == "diagonal":
            P_diag = torch.exp(self.P_diag(shared))  # Ensure positive for negative semi-definite A
        return V, mu, P_diag

# NAF Agent
class NAFAgent:
    def __init__(self, state_dim, action_dim, hidden_sizes, learning_rate, gamma, tau, epsilon, batch_size, memory_capacity, covariance="identity", device='cuda'):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.memory = Memory(memory_capacity, batch_size)
        self.device = device

        self.network = NAFNetwork(state_dim, action_dim, hidden_sizes, covariance=covariance).to(device)
        self.target_network = NAFNetwork(state_dim, action_dim, hidden_sizes, covariance=covariance).to(device)
        self.target_network.load_state_dict(self.network.state_dict())
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)

    def get_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        self.network.eval()
        with torch.no_grad():
            _, mu, _ = self.network(state)
            noise = np.random.normal(0, self.epsilon, self.action_dim)
            action = mu.cpu().numpy().flatten() + noise
            action = np.clip(action, -1, 1)
        self.network.train()
        return action

    def observe(self, state, action, reward, next_state, done):
        self.memory.store((state, action, reward, next_state, done))

    def learn(self):
        if len(self.memory) < self.batch_size:
            return None
        batch = self.memory.sample()
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.FloatTensor(np.array(actions)).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        V_next, _, _ = self.target_network(next_states)
        target_Q = rewards + self.gamma * V_next * (1 - dones)

        V, mu, P_diag = self.network(states)
        u_minus_mu = actions - mu
        A = -0.5 * torch.sum(P_diag * u_minus_mu ** 2, dim=1, keepdim=True)
        Q = A + V

        loss = nn.MSELoss()(Q, target_Q.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.soft_update(self.network, self.target_network, self.tau)
        return loss.item()

    def soft_update(self, source, target, tau):
        for source_param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.copy_(tau * source_param.data + (1 - tau) * target_param.data)

    def update_epsilon(self, episode):
        self.epsilon = max(0.01, 0.2 - 9.5e-5 * episode)

# NAF Agent Wrapper
class NAF_Agent:
    def __init__(
            self,
            env_name,
            env_fn,
            max_episodes: int,
            total_timesteps : int,
            plot_path: str,
            video_path: str,
            RL_load_path: str,
            hidden_sizes: list,
            learning_rate: float,
            gamma: float,
            tau: float,
            epsilon: float,
            batch_size: int,
            memory_capacity: int,
            covariance: str,
            max_steps: int,
    ):

        def make_env():
            """Create and return the environment by calling env_fn."""
            return env_fn

        """Initialize the NAF agent."""
        self.env_fn = make_env
        self.max_episodes = max_episodes
        self.max_steps = max_steps
        self.plot_path = plot_path
        self.video_path = video_path
        self.RL_load_path = RL_load_path
        self.env = make_env()
        self.env_name = env_name
        state_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.shape[0]
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.agent = NAFAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_sizes=hidden_sizes,
            learning_rate=learning_rate,
            gamma=gamma,
            tau=tau,
            epsilon=epsilon,
            batch_size=batch_size,
            memory_capacity=memory_capacity,
            covariance=covariance,
        )
        self.env.close()

    def train(self, max_episodes, max_steps, save_interval, save_path):
        """Train the NAF model and save results."""
        os.makedirs(self.plot_path, exist_ok=True)
        env = self.env_fn()
        episode_history = []
        reward_history = []
        loss_history = []
        epsilon_history = []

        for ep in range(1, self.max_episodes + 1):
            state, _ = env.reset()
            total_reward = 0
            losses = []
            for t in range(self.max_steps):
                action = self.agent.get_action(state)
                low = env.action_space.low
                high = env.action_space.high
                scaled_action = low + (action + 1) * (high - low) / 2
                next_state, reward, terminated, truncated, _ = env.step(scaled_action)
                done = terminated or truncated

                if self.env_name == "Pendulum-v1" :
                    # Apply reward shaping
                    shaped = shaped_reward_Pendulum(next_state, action, reward)
                    self.agent.observe(state, action, shaped, next_state, done)
                else:
                    # Use original reward for HalfCheetah
                    self.agent.observe(state, action, reward, next_state, done)

                loss = self.agent.learn()
                if loss is not None:
                    losses.append(loss)
                state = next_state
                total_reward += reward  # Keep tracking raw reward for logging
                if done:
                    break
            self.agent.update_epsilon(ep)
            avg_loss = np.mean(losses) if losses else np.nan
            episode_history.append(ep)
            reward_history.append(total_reward)
            loss_history.append(avg_loss)
            epsilon_history.append(self.agent.epsilon)
            print(
                f"Episode {ep}/{self.max_episodes}, Reward={total_reward:.2f}, Loss={avg_loss:.4f}, Epsilon={self.agent.epsilon:.4f}")

        env.close()

        # Save model
        torch.save(self.agent.network.state_dict(), self.RL_load_path)
        print(f"Model saved to {self.RL_load_path}")

        # Save to CSV
        df = pd.DataFrame({
            'Episode': episode_history,
            'Reward': reward_history,
            'Loss': loss_history,
            'Epsilon': epsilon_history
        })
        csv_path = os.path.join(self.plot_path, f"results_NAF_{self.env_name}.csv")
        df.to_csv(csv_path, index=False)
        print(f"Training results saved to {csv_path}")

        # Plot
        self.plot_results(df)

    def test(self, max_episodes, max_steps, load_path):
        """Evaluate the NAF model and record a video."""
        # Load model
        state_dict = torch.load(self.RL_load_path, map_location=self.device)
        self.agent.network.load_state_dict(state_dict)
        self.agent.network.eval()

        if self.env_name == "Pendulum-v1":
            canvas_size = 500
            rod_length = int(canvas_size * 0.222)
            rod_thickness = int(canvas_size * 0.04595)
            center = (canvas_size // 2, canvas_size // 2)
            bg_color = (255, 255, 255)  # White background
            font = cv2.FONT_HERSHEY_SIMPLEX

            env = self.env_fn()
            obs, _ = env.reset()
            done = False
            step = 0
            frames = []

            while not done and step < max_steps:
                action = self.agent.get_action(obs)
                low = env.action_space.low
                high = env.action_space.high
                scaled_action = low + (action + 1) * (high - low) / 2
                torque = float(scaled_action[0])
                obs, reward, terminated, truncated, _ = env.step(scaled_action)
                done = terminated or truncated

                cos_th, sin_th = obs[0], obs[1]
                theta = np.arctan2(sin_th, cos_th)

                canvas = np.full((canvas_size, canvas_size, 3), bg_color, dtype=np.uint8)
                end_x = int(center[0] + rod_length * np.sin(theta))
                end_y = int(center[1] - rod_length * np.cos(theta))
                pendulum_end = (end_x, end_y)

                # Draw pendulum rod
                cv2.line(canvas, center, pendulum_end, (204, 77, 77, 255), thickness=rod_thickness,
                         lineType=cv2.LINE_AA)
                cv2.circle(canvas, center, 5, (0, 0, 0, 255), thickness=-1, lineType=cv2.LINE_AA)  # Pivot

                # Draw torque arc
                max_torque = 2.0
                arc_radius = int(20 + 20 * abs(torque) / max_torque)
                arc_color = (0, 0, 0)
                arc_thickness = 1

                start_angle = 0 if torque >= 0 else 180
                end_angle = 270 if torque >= 0 else 450
                cv2.ellipse(canvas, center, (arc_radius, arc_radius), 0, start_angle, end_angle,
                            arc_color, arc_thickness, lineType=cv2.LINE_AA)

                angle_offset_deg = 90
                arrow_angle_deg = (end_angle + angle_offset_deg) % 360
                arc_arrow_angle_rad = np.radians(arrow_angle_deg)

                arc_end_x = int(center[0] + arc_radius * np.cos(arc_arrow_angle_rad))
                arc_end_y = int(center[1] - arc_radius * np.sin(arc_arrow_angle_rad))

                arc_tangent_angle = arc_arrow_angle_rad - np.pi / 2

                arrow_size = 10
                for offset_deg in [-30, 20]:
                    offset_rad = np.radians(offset_deg)
                    line_angle = arc_tangent_angle + offset_rad
                    x2 = int(arc_end_x + arrow_size * np.cos(line_angle))
                    y2 = int(arc_end_y - arrow_size * np.sin(line_angle))
                    cv2.line(canvas, (arc_end_x, arc_end_y), (x2, y2), arc_color, 1, cv2.LINE_AA)

                # Show torque value
                text = f"Torque: {torque:.2f}"
                cv2.putText(canvas, text, (10, canvas_size - 10), font, 0.6, (50, 50, 50), 2, cv2.LINE_AA)

                frames.append(canvas)
                step += 1

            env.close()

            if frames:
                video_path = os.path.join(self.video_path, f"NAF_{self.env_name}_video.mp4")
                os.makedirs(self.video_path, exist_ok=True)
                with imageio.get_writer(video_path, fps=30, macro_block_size=None) as writer:
                    for frame in frames:
                        writer.append_data(frame)
                print(f"Video saved at {video_path}")
            else:
                print("No frames captured, video not created.")

        else:
            # Evaluate
            n_eval_episodes = 50
            total_rewards = []
            for _ in range(n_eval_episodes):
                env = self.env_fn()
                state, _ = env.reset()
                total_reward = 0
                done = False
                step = 0
                while not done and step < max_steps:
                    action = self.agent.get_action(state)
                    low = env.action_space.low
                    high = env.action_space.high
                    scaled_action = low + (action + 1) * (high - low) / 2
                    state, reward, terminated, truncated, _ = env.step(scaled_action)
                    done = terminated or truncated
                    total_reward += reward
                    step += 1
                total_rewards.append(total_reward)
                env.close()
            mean_reward = np.mean(total_rewards)
            std_reward = np.std(total_rewards)
            print(f"Evaluation over {n_eval_episodes} episodes: mean_reward = {mean_reward:.2f} ± {std_reward:.2f}")

            # Record video
            record_env = RecordVideo(self.env_fn(), self.video_path, episode_trigger=lambda e: e == 0)
            state, _ = record_env.reset()
            done = False
            step = 0
            while not done and step < max_steps:
                action = self.agent.get_action(state)
                low = record_env.action_space.low
                high = record_env.action_space.high
                scaled_action = low + (action + 1) * (high - low) / 2
                state, reward, terminated, truncated, _ = record_env.step(scaled_action)
                done = terminated or truncated
                step += 1
            record_env.close()
            print(f"Video recorded to {self.video_path}")

    def plot_results(self, df):
        """Plot and save reward and loss plots."""
        rewards = df['Reward']
        n = len(rewards)
        window = min(50, n)
        if n >= 1:
            sma = np.convolve(rewards, np.ones(window) / window, mode='valid')
            episodes = df['Episode']
            sma_episodes = episodes[window - 1:]

            plt.figure(figsize=(10, 5))
            plt.plot(episodes, rewards, label='Raw Reward', linewidth=1, alpha=0.2)
            plt.plot(sma_episodes, sma, label=f'SMA ({window})', linewidth=2.5, alpha=0.9)
            lower = sma * 0.9
            upper = sma * 1.1
            plt.fill_between(sma_episodes, lower, upper, alpha=0.1)
            plt.title("Training Rewards (NAF on Pendulum)", fontsize=14)
            plt.xlabel("Episode", fontsize=12)
            plt.ylabel("Reward", fontsize=12)
            plt.legend(frameon=True)
            plt.tight_layout()
            plt.savefig(os.path.join(self.plot_path, f"reward_plot_NAF_{self.env_name}.png"), format='png', dpi=300, bbox_inches='tight')
            plt.close()

        if 'Loss' in df.columns and df['Loss'].notna().any():
            plt.figure(figsize=(8, 4))
            plt.plot(df['Episode'], df['Loss'], label='Loss', linewidth=1.5, alpha=0.6)
            plt.title("Training Loss (NAF on Pendulum)", fontsize=14)
            plt.xlabel("Episode", fontsize=12)
            plt.ylabel("Loss", fontsize=12)
            plt.legend(frameon=True)
            plt.tight_layout()
            plt.savefig(os.path.join(self.plot_path, f"loss_plot_NAF_{self.env_name}.png"), format='png', dpi=300, bbox_inches='tight')
            plt.close()