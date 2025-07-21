# dqn
import os

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from src.utils import ReplayMemory
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DQN_Network(nn.Module):
    def __init__(self, num_actions, input_dim):
        super(DQN_Network, self).__init__()
        self.FC = nn.Sequential(
            nn.Linear(input_dim, 12),
            nn.ReLU(inplace=True),
            nn.Linear(12, 8),
            nn.ReLU(inplace=True),
            nn.Linear(8, num_actions)
        )
        for layer in [self.FC]:
            for module in layer:
                if isinstance(module, nn.Linear):
                    nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')

    def forward(self, x):
        Q = self.FC(x)
        return Q

class DQN_Agent:
    def __init__(self, env, epsilon_max, epsilon_min, epsilon_decay,clip_grad_norm, learning_rate,
                  discount, memory_capacity, env_name, batch_size, update_frequency, plot_path, map_size):
        """Initialize the DQN agent."""
        self.env = env
        self.epsilon_max = epsilon_max
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.discount = discount
        self.action_space = env.action_space
        self.action_space.seed(2024)
        self.observation_space = env.observation_space
        self.replay_memory = ReplayMemory(memory_capacity)
        self.batch_size = batch_size
        self.update_frequency = update_frequency

        self.algorithm = "DQN"
        self.env_name = env_name
        self.plot_path = plot_path

        self.map_size = map_size

        # Determine input dimension based on environment
        if env_name == "FrozenLake-v1":
            input_dim = self.observation_space.n  # For one-hot encoded states
        elif env_name == "MountainCar-v0":
            input_dim = self.observation_space.shape[0]  # For continuous state spaces
        else:
            raise ValueError(f"Unsupported environment for DQN: {env_name}")

        output_dim = self.action_space.n
        self.main_network = DQN_Network(num_actions=output_dim, input_dim=input_dim).to(device)
        self.target_network = DQN_Network(num_actions=output_dim, input_dim=input_dim).to(device).eval()
        self.target_network.load_state_dict(self.main_network.state_dict())

        self.clip_grad_norm = clip_grad_norm
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.main_network.parameters(), lr=learning_rate)

    def select_action(self, state, epsilon=None):
        """Select an action using epsilon-greedy policy."""
        if epsilon is None:
            epsilon = self.epsilon_max
        if np.random.random() < epsilon:
            return self.action_space.sample()
        if not torch.is_tensor(state):
            state = torch.as_tensor(state, dtype=torch.float32, device=device)
        with torch.no_grad():
            Q_values = self.main_network(state)
            return torch.argmax(Q_values).item()

    def learn(self, batch_size, done):
        """Perform one learning step using a batch from replay memory."""
        states, actions, next_states, rewards, dones = self.replay_memory.sample(batch_size)
        actions = actions.unsqueeze(1)
        rewards = rewards.unsqueeze(1)
        dones = dones.unsqueeze(1)

        predicted_q = self.main_network(states).gather(dim=1, index=actions)
        with torch.no_grad():
            next_target_q_value = self.target_network(next_states).max(dim=1, keepdim=True)[0]
        next_target_q_value[dones] = 0
        y_js = rewards + (self.discount * next_target_q_value)
        loss = self.criterion(predicted_q, y_js)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.main_network.parameters(), self.clip_grad_norm)
        self.optimizer.step()
        return loss.item()

    def hard_update(self):
        """Update the target network with the main network's weights."""
        self.target_network.load_state_dict(self.main_network.state_dict())

    def update_epsilon(self):
        """Decay the epsilon value."""
        self.epsilon_max = max(self.epsilon_min, self.epsilon_max * self.epsilon_decay)

    def save(self, path):
        """Save the main network's state dict."""
        torch.save(self.main_network.state_dict(), path)

    def train(self, max_episodes, max_steps, save_interval, save_path):
        """Train the agent over multiple episodes and return results as a DataFrame."""
        rewards = []
        losses = []

        batch_size = self.batch_size
        update_frequency = self.update_frequency

        for episode in range(1, max_episodes + 1):
            state, _ = self.env.reset()
            done = False
            truncation = False
            episode_reward = 0
            episode_loss = 0
            step_size = 0
            learn_steps = 0

            while not done and not truncation and step_size < max_steps:
                action = self.select_action(state)
                next_state, reward, done, truncation, _ = self.env.step(action)
                self.replay_memory.store(state, action, next_state, reward, done)
                if len(self.replay_memory) > batch_size:
                    loss = self.learn(batch_size, (done or truncation))
                    episode_loss += loss
                    learn_steps += 1
                    if step_size % update_frequency == 0:
                        self.hard_update()
                state = next_state
                episode_reward += reward
                step_size += 1

            # Compute average loss for the episode
            if learn_steps > 0:
                episode_loss /= learn_steps
            else:
                episode_loss = np.nan

            self.update_epsilon()
            rewards.append(episode_reward)
            losses.append(episode_loss)

            # Save model at intervals
            if episode % save_interval == 0:
                self.save(save_path)
                print(f"Episode {episode}: Model saved.")

            # Corrected print statement with conditional formatting
            loss_str = f"{episode_loss:.2f}" if not np.isnan(episode_loss) else "N/A"
            print(f"Episode: {episode}, Reward: {episode_reward:.2f}, Loss: {loss_str}")

        # Create and return DataFrame
        results_df = pd.DataFrame({
            'Episode': range(1, max_episodes + 1),
            'Reward': rewards,
            'Loss': losses
        })

        self.plot_results(results_df)
        self.save_results_to_csv(results_df)

    def test(self, max_episodes, max_steps, load_path):
        """Test the agent over multiple episodes and return results as a DataFrame."""
        self.main_network.load_state_dict(torch.load(load_path))
        self.main_network.eval()
        rewards = []

        for episode in range(max_episodes):
            state, _ = self.env.reset()
            done = False
            truncation = False
            episode_reward = 0
            step_size = 0

            while not done and not truncation and step_size < max_steps:
                action = self.select_action(state, epsilon=0)  # Greedy policy
                next_state, reward, done, truncation, _ = self.env.step(action)
                state = next_state
                episode_reward += reward
                step_size += 1

            rewards.append(episode_reward)
            print(f"Test Episode {episode + 1}: Reward: {episode_reward:.2f}")  # Adjusted for 1-based indexing

        # Create and return DataFrame (Loss is NaN since no learning occurs)
        results_df = pd.DataFrame({
            'Episode': range(1, max_episodes + 1),
            'Reward': rewards,
            'Loss': [np.nan] * max_episodes
        })

    def plot_results(self, df):
        """Plot rewards and loss from the DataFrame with SMA and clipping as separate images."""
        rewards = np.array(df['Reward'])
        sma = np.convolve(rewards, np.ones(50) / 50, mode='valid')
        rewards_clipped = np.clip(rewards, None, 100)
        sma_clipped = np.clip(sma, None, 100)
        episodes = np.arange(1, len(rewards_clipped) + 1)
        sma_episodes = np.arange(50, len(rewards) + 1)

        # Plot rewards
        plt.figure(figsize=(10, 5))
        plt.plot(episodes, rewards_clipped, label='Raw Reward', linewidth=1, alpha=0.2)
        plt.plot(sma_episodes, sma_clipped, label='AVG (50)', linewidth=2.5, alpha=0.9)
        lower = sma_clipped * 0.9
        upper = sma_clipped * 1.1
        plt.fill_between(sma_episodes, lower, upper, alpha=0.1)
        plt.title(f"Training Rewards ({self.algorithm} on {self.env_name})", fontsize=14)
        plt.xlabel("Episode", fontsize=12)
        plt.ylabel("Reward", fontsize=12)
        plt.legend(frameon=True)
        plt.tight_layout()
        plt.savefig(f"{self.plot_path}/reward_plot_{self.algorithm}_{self.env_name}.png", format='png', dpi=300,
                    bbox_inches='tight')
        plt.close()

        # Plot loss (if available)
        if 'Loss' in df.columns and not df['Loss'].isna().all():
            plt.figure(figsize=(8, 4))
            plt.plot(df['Episode'], df['Loss'], label='Loss', linewidth=1.5, alpha=0.6)
            plt.title(f"Training Loss ({self.algorithm} on {self.env_name})", fontsize=14)
            plt.xlabel("Episode", fontsize=12)
            plt.ylabel("Loss", fontsize=12)
            plt.legend(frameon=True)
            plt.tight_layout()
            plt.savefig(f"{self.plot_path}/loss_plot_{self.algorithm}_{self.env_name}.png", format='png', dpi=300,
                        bbox_inches='tight')
            plt.close()

    def save_results_to_csv(self, df):
        """Save the DataFrame to a CSV file."""
        os.makedirs(self.plot_path, exist_ok=True)
        csv_path = f"{self.plot_path}/results_{self.algorithm}_{self.env_name}.csv"
        df.to_csv(csv_path, index=False)
        print(f'\n~~~~~~Training results saved to {csv_path}\n')
