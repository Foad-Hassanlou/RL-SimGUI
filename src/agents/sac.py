# sac
import os

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import StopTrainingOnMaxEpisodes, BaseCallback, CallbackList
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import imageio

from src.envs.continuous_envs import PendulumRewardWrapper

# Environment factory for Pendulum-v1
def make_env_Pendulum():
    # Use Gymnasium's Pendulum-v1 directly
    env = gym.make("Pendulum-v1", render_mode="rgb_array")
    env = PendulumRewardWrapper(env)
    return env


def make_env_HalfCheetah():
    env = gym.make("HalfCheetah-v4", render_mode="rgb_array")
    return env


class LossCallback(BaseCallback):
    """Custom callback for logging one critic loss per episode."""
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.losses = []
        self.episode_losses = []
        self.episode_count = 0
        self.current_loss = None

    def _on_step(self) -> bool:
        logs = self.model.logger.name_to_value
        loss = logs.get("train/critic_loss")
        if loss is not None:
            self.current_loss = loss
            self.losses.append((self.num_timesteps, loss))
        infos = self.locals.get("infos", [])
        for info in infos:
            if info.get("episode") is not None:
                self.episode_count += 1
                if self.current_loss is not None:
                    self.episode_losses.append(self.current_loss)
                else:
                    self.episode_losses.append(np.nan)
        return True


def _load_monitor_results(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath, comment='#')
    df = df.rename(columns={'r': 'Reward', 'l': 'Length'})
    df['Episode'] = np.arange(1, len(df) + 1)
    return df


class SAC_Agent:
    def __init__(
        self,
        env_name: str,
        policy: str,
        learning_rate: float,
        buffer_size: int,
        batch_size: int,
        tau: float,
        gamma: float,
        train_freq: int,
        gradient_steps: int,
        ent_coef,
        verbose: int,
        plot_path: str,
        video_path: str,
        RL_load_path: str,
        max_episodes: int,
        total_timesteps: int = int(2e6),
        n_envs: int = 1,
        max_steps: int = 200,
        save_interval: int = 1000,
    ):
        self.env_name = env_name
        self.policy = policy
        self.learning_rate = learning_rate
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.tau = tau
        self.gamma = gamma
        self.train_freq = train_freq
        self.gradient_steps = gradient_steps
        self.ent_coef = ent_coef
        self.verbose = verbose
        self.plot_path = plot_path
        self.video_path = video_path
        self.RL_load_path = RL_load_path.replace('.pth', '')
        self.max_episodes = max_episodes
        self.total_timesteps = total_timesteps
        self.n_envs = n_envs
        self.algorithm = "SAC"

        # Create vectorized and monitored environment
        if self.env_name == "Pendulum-v1":
            self.env_fn = make_env_Pendulum
        elif self.env_name == "HalfCheetah-v4":
            self.env_fn = make_env_HalfCheetah
        else:
            raise ValueError(f"Unsupported environment: {self.env_name}")

        self.vec_env = DummyVecEnv([self.env_fn for _ in range(self.n_envs)])
        self.monitor_path = os.path.join(self.plot_path, "monitor.csv")
        os.makedirs(self.plot_path, exist_ok=True)
        self.vec_env = VecMonitor(self.vec_env, filename=self.monitor_path)

        # Initialize SAC model
        self.model = SAC(
            policy=self.policy,
            env=self.vec_env,
            learning_rate=self.learning_rate,
            buffer_size=self.buffer_size,
            batch_size=self.batch_size,
            tau=self.tau,
            gamma=self.gamma,
            train_freq=self.train_freq,
            gradient_steps=self.gradient_steps,
            ent_coef=self.ent_coef,
            verbose=self.verbose,
        )

    def save_results_to_csv(self, df: pd.DataFrame):
        os.makedirs(self.plot_path, exist_ok=True)
        csv_path = os.path.join(self.plot_path, f"results_{self.algorithm}_{self.env_name}.csv")
        df.to_csv(csv_path, index=False)
        print(f"Training results saved to {csv_path}")

    def plot_results(self, df: pd.DataFrame):
        rewards = np.array(df['Reward'])
        n = len(rewards)
        window = min(50, n)

        if n >= 1:
            sma = np.convolve(rewards, np.ones(window) / window, mode='valid')
            episodes = np.arange(1, n + 1)
            sma_episodes = np.arange(window, n + 1)

            plt.figure(figsize=(10, 5))
            plt.plot(episodes, rewards, label='Raw Reward', linewidth=1, alpha=0.2)
            plt.plot(sma_episodes, sma, label=f'AVG ({window})', linewidth=2.5, alpha=0.9)
            lower = sma * 0.9
            upper = sma * 1.1
            plt.fill_between(sma_episodes, lower, upper, alpha=0.1)
            plt.title(f"Training Rewards ({self.algorithm} on {self.env_name})", fontsize=14)
            plt.xlabel("Episode", fontsize=12)
            plt.ylabel("Reward", fontsize=12)
            plt.legend(frameon=True)
            plt.tight_layout()
            plt.savefig(os.path.join(self.plot_path, f"reward_plot_{self.algorithm}_{self.env_name}.png"),
                        format='png', dpi=300, bbox_inches='tight')
            plt.close()

        if 'Loss' in df.columns and df['Loss'].notna().any():
            plt.figure(figsize=(8, 4))
            plt.plot(df['Episode'], df['Loss'], label='Critic Loss', linewidth=1.5, alpha=0.6)
            plt.title(f"Training Loss ({self.algorithm} on {self.env_name})", fontsize=14)
            plt.xlabel("Episode", fontsize=12)
            plt.ylabel("Loss", fontsize=12)
            plt.legend(frameon=True)
            plt.tight_layout()
            plt.savefig(os.path.join(self.plot_path, f"loss_plot_{self.algorithm}_{self.env_name}.png"),
                        format='png', dpi=300, bbox_inches='tight')
            plt.close()

    def train(self, max_episodes, max_steps, save_interval, save_path):
        """Train the SAC model with a StopTraining callback and log losses."""
        print(f"Training for up to {self.max_episodes} episodes...")
        stop_cb = StopTrainingOnMaxEpisodes(max_episodes=self.max_episodes, verbose=1)
        loss_cb = LossCallback(verbose=0)
        cb_list = CallbackList([stop_cb, loss_cb])

        self.model.learn(total_timesteps=self.total_timesteps, callback=cb_list)
        self.model.save(self.RL_load_path)
        print(f"Model saved to {self.RL_load_path}.zip")

        df = _load_monitor_results(self.monitor_path)
        if loss_cb.episode_losses:
            loss_series = pd.Series(loss_cb.episode_losses, index=np.arange(1, len(loss_cb.episode_losses)+1))
            df["Loss"] = loss_series.reindex(df["Episode"], fill_value=np.nan).values
        else:
            df["Loss"] = np.nan

        self.save_results_to_csv(df)
        self.plot_results(df)

    def test(self, max_episodes, max_steps, load_path):

        if self.env_name == "Pendulum-v1" :
            model = SAC.load(self.RL_load_path, env=VecMonitor(DummyVecEnv([self.env_fn for _ in range(self.n_envs)])))

            canvas_size = 500
            rod_length = int(canvas_size * 0.222)
            rod_thickness = int(canvas_size * 0.04595)
            center = (canvas_size // 2, canvas_size // 2)
            bg_color = (255, 255, 255)  # White background
            font = cv2.FONT_HERSHEY_SIMPLEX

            env = gym.make(self.env_name)
            obs, _ = env.reset()
            done = False
            step = 0
            frames = []

            while not done and step < max_steps:
                action, _ = model.predict(obs, deterministic=True)
                torque = float(action[0])
                obs, reward, terminated, truncated, info = env.step(action)
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
                video_path = os.path.join(self.video_path, f"{self.algorithm}_{self.env_name}_video.mp4")
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
            eval_env = DummyVecEnv([self.env_fn for _ in range(self.n_envs)])
            model = SAC.load(self.RL_load_path, env=eval_env)
            mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=n_eval_episodes)
            print(f"Evaluation over {n_eval_episodes} episodes: mean_reward = {mean_reward:.2f} ± {std_reward:.2f}")

            # Record
            env = self.env_fn()
            os.makedirs(self.video_path, exist_ok=True)
            video_path = os.path.join(self.video_path, f"{self.algorithm}_{self.env_name}_video.mp4")
            frames = []

            obs, _ = env.reset()
            done = False
            step = 0
            while not done and step < max_steps:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                frame = env.render()
                frames.append(frame)
                step += 1

            env.close()
            with imageio.get_writer(video_path, fps=30, macro_block_size=None) as writer:
                for frame in frames:
                    writer.append_data(frame)

            print(f"Video saved at {video_path}")