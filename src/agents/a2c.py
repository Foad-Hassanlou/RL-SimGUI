# a2c
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import StopTrainingOnMaxEpisodes, BaseCallback, CallbackList
import gymnasium as gym
from gymnasium.wrappers import FlattenObservation, RecordVideo
from src.envs.discrete_envs import step_wrapper,observation_wrapper_mountaincar, reward_wrapper_mountaincar

def make_env_Frozen8():
    # Flatten the discrete state into a 1D vector
    env = gym.make(
        "FrozenLake-v1",
        map_name="8x8",
        is_slippery=False,
        render_mode="rgb_array",
    )
    env = FlattenObservation(env)
    return env

def make_env_Frozen4():
    # Flatten the discrete state into a 1D vector
    env = gym.make(
        "FrozenLake-v1",
        map_name="4x4",
        is_slippery=False,
        render_mode="rgb_array",
    )
    env = FlattenObservation(env)
    return env

def make_env_Car():
    env = gym.make(
        "MountainCar-v0",
        render_mode="rgb_array",
    )
    env = step_wrapper(env, env_name="MountainCar-v0")
    return env

class LossCallback(BaseCallback):
    """Custom callback for logging one loss per episode."""
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.losses = []
        self.episode_losses = []
        self.episode_count = 0
        self.current_loss = None

    def _on_step(self) -> bool:
        logs = self.model.logger.name_to_value
        loss = logs.get("train/loss")
        if loss is not None:
            self.current_loss = loss
            self.losses.append((self.num_timesteps, loss))  # Store loss with timestep
        infos = self.locals.get("infos", [])
        for info in infos:
            if info.get("episode") is not None:
                self.episode_count += 1
                # Assign the most recent loss to this episode
                if self.current_loss is not None:
                    self.episode_losses.append(self.current_loss)
                else:
                    self.episode_losses.append(np.nan)  # Fallback if no loss available
        return True


def _load_monitor_results(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath, comment='#')
    df = df.rename(columns={'r': 'Reward', 'l': 'Length'})
    df['Episode'] = np.arange(1, len(df) + 1)
    return df

class A2C_Agent():
    def __init__(
            self,
            env_name,
            learning_rate, 
            gamma, 
            n_steps, 
            gae_lambda, 
            ent_coef, 
            vf_coef, 
            max_grad_norm, 
            verbose, 
            policy,
            plot_path,
            RL_load_path,
            video_path,
            max_episodes,
            total_timesteps: int = 1e9,
            n_envs: int = 8,
            map_size = 8
    ):


        self.total_timesteps = total_timesteps
        self.n_envs = n_envs
        self.plot_path = plot_path
        self.RL_load_path = RL_load_path
        self.video_path = video_path
        self.env_name = env_name
        self.max_episodes = max_episodes / n_envs
        self.algorithm = "A2C"

        self.map_size = map_size

        # Create vectorized environment with monitoring
        if self.env_name == "MountainCar-v0":
            self.vec_env = DummyVecEnv([make_env_Car for _ in range(self.n_envs)])
        elif self.env_name == "FrozenLake-v1" and map_size == 8:
            self.vec_env = DummyVecEnv([make_env_Frozen8 for _ in range(self.n_envs)])
        elif self.env_name == "FrozenLake-v1" and map_size == 4:
            self.vec_env = DummyVecEnv([make_env_Frozen4 for _ in range(self.n_envs)])

        self.monitor_path = os.path.join(self.plot_path, "monitor.csv")
        self.vec_env = VecMonitor(self.vec_env, filename=self.monitor_path)

        self.model = A2C(
            policy=policy,
            env= self.vec_env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            gamma=gamma,
            gae_lambda=gae_lambda,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            verbose=verbose,
        )

        # “Subtracting” a substring via replace
        text = self.RL_load_path
        to_remove = ".pth"
        self.RL_load_path = text.replace(to_remove, "")

    def save_results_to_csv(self, df: pd.DataFrame):
        os.makedirs(self.plot_path, exist_ok=True)
        csv_path = os.path.join(self.plot_path, f"results_{self.algorithm}_{self.env_name}.csv")
        df.to_csv(csv_path, index=False)
        print(f"Training results saved to {csv_path}")
    
    def train(self, max_episodes, max_steps, save_interval, save_path):

        print(f"Training for up to {max_episodes} episodes (max_timesteps={self.total_timesteps})...")

        stop_cb = StopTrainingOnMaxEpisodes(max_episodes=self.max_episodes, verbose=1)
        loss_cb = LossCallback(verbose=0)
        cb_list = CallbackList([stop_cb, loss_cb])

        self.model.learn(total_timesteps=self.total_timesteps, callback=cb_list)

        self.model.save(self.RL_load_path)

        print(f"Model saved to {self.RL_load_path}.zip")

        df = _load_monitor_results(self.monitor_path)

        # Assign losses to episodes
        if loss_cb.episode_losses:
            loss_series = pd.Series(loss_cb.episode_losses, index=np.arange(1, len(loss_cb.episode_losses)+1))
            df["Loss"] = loss_series.reindex(df["Episode"], fill_value=np.nan).values
        else:
            df["Loss"] = np.nan  # If no losses recorded, fill with NaN

        self.save_results_to_csv(df)
        self.plot_results(df)

    def test(self, max_episodes, max_steps, load_path):

        vec_env = self.vec_env
        model = A2C.load(self.RL_load_path, env=vec_env)

        mean_reward, std_reward = evaluate_policy(model, vec_env, n_eval_episodes=50)
        print(f"Evaluation over 50 episodes: mean_reward = {mean_reward:.2f} ± {std_reward:.2f}")

        os.makedirs(self.video_path, exist_ok=True)

        if self.env_name == "MountainCar-v0" :
            record_env = make_env_Car()
            record_env = RecordVideo(
                record_env,
                self.video_path,
                episode_trigger=lambda e: e == 0,
                name_prefix="A2C_MountainCar-v0"
            )   
                 
        elif self.env_name == "FrozenLake-v1" :
            if self.map_size == 8:
                record_env = make_env_Frozen8()
            else:
                record_env = make_env_Frozen4()

            record_env = RecordVideo(
                record_env,
                self.video_path,
                episode_trigger=lambda e: e == 0,
                name_prefix="A2C_FrozenLake-v1"
            )
            
        obs, _ = record_env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = record_env.step(int(action))
            done = terminated or truncated
        record_env.close()
        print(f"Video recorded to {self.video_path}")

        data = pd.DataFrame

    def plot_results(self, df: pd.DataFrame):

        rewards = np.array(df['Reward'])
        n = len(rewards)
        window = min(50, n)
        if n >= 1:
            sma = np.convolve(rewards, np.ones(window) / window, mode='valid')
            rewards_clipped = np.clip(rewards, None, 100)
            sma_clipped = np.clip(sma, None, 100)
            episodes = np.arange(1, n + 1)
            sma_episodes = np.arange(window, n + 1)

            plt.figure(figsize=(10, 5))
            plt.plot(episodes, rewards_clipped, label='Raw Reward', linewidth=1, alpha=0.2)
            plt.plot(sma_episodes, sma_clipped, label=f'AVG ({window})', linewidth=2.5, alpha=0.9)
            lower = sma_clipped * 0.9
            upper = sma_clipped * 1.1
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
            plt.plot(df['Episode'], df['Loss'], label='Loss', linewidth=1.5, alpha=0.6)
            plt.title(f"Training Loss ({self.algorithm} on {self.env_name})", fontsize=14)
            plt.xlabel("Episode", fontsize=12)
            plt.ylabel("Loss", fontsize=12)
            plt.legend(frameon=True)
            plt.tight_layout()
            plt.savefig(os.path.join(self.plot_path, f"loss_plot_{self.algorithm}_{self.env_name}.png"),
                        format='png', dpi=300, bbox_inches='tight')
            plt.close()
