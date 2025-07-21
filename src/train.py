#train.py
import os
import gc
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from gymnasium.wrappers import RecordVideo
from src.agents.dqn import DQN_Agent
from src.agents.a2c import A2C_Agent
from src.agents.naf import NAF_Agent
from src.agents.sac import SAC_Agent
from src.envs.discrete_envs import step_wrapper
import gymnasium as gym

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Clear memory
gc.collect()
torch.cuda.empty_cache()
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# Seed everything for reproducible results
seed = 2024
np.random.seed(seed)
np.random.default_rng(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class Model_TrainTest:
    def __init__(self, hyperparams, algorithm, env_name):
        """Initialize the training/testing class with hyperparameters, algorithm, and environment."""
        # Common paths & flags
        self.train_mode       = hyperparams["train_mode"]
        self.RL_load_path     = hyperparams["RL_load_path"]
        self.save_path        = hyperparams["save_path"]
        self.plot_path        = hyperparams["plot_path"]
        self.video_path       = hyperparams["video_path"]
        self.render           = hyperparams["render"]
        self.save_interval    = hyperparams["save_interval"]


        # Algorithm-specific hyperparameters
        self.algorithm = algorithm.lower()
        alg = self.algorithm
        self.env_name = env_name

        if alg == "dqn":
            self.clip_grad_norm   = hyperparams["clip_grad_norm"]
            self.learning_rate    = hyperparams["learning_rate"]
            self.discount_factor  = hyperparams["discount_factor"]
            self.batch_size       = hyperparams["batch_size"]
            self.update_frequency = hyperparams["update_frequency"]
            self.epsilon_max      = hyperparams["epsilon_max"]
            self.epsilon_min      = hyperparams["epsilon_min"]
            self.epsilon_decay    = hyperparams["epsilon_decay"]
            self.memory_capacity  = hyperparams["memory_capacity"]
            self.map_size = hyperparams["map_size"]

        elif alg == "a2c":
            # Add other A2C-specific parameters here if needed
            self.learning_rate   = hyperparams["learning_rate"]
            self.gamma           = hyperparams["gamma"]
            self.n_steps         = hyperparams["n_steps"]
            self.gae_lambda      = hyperparams["gae_lambda"]
            self.ent_coef        = hyperparams["ent_coef"]
            self.vf_coef         = hyperparams["vf_coef"]
            self.max_grad_norm   = hyperparams["max_grad_norm"]
            self.verbose         = hyperparams.get("verbose", 1)
            self.policy          = hyperparams["policy"]

        elif alg == "naf":
            # NAF-specific (continuous)
            self.learning_rate = hyperparams["learning_rate"]
            self.total_timesteps = int(hyperparams.get("total_timesteps", 0))
            self.hidden_sizes = hyperparams["hidden_sizes"]
            self.gamma = hyperparams["gamma"]
            self.tau = hyperparams["tau"]
            self.epsilon = hyperparams["epsilon"]
            self.batch_size = hyperparams["batch_size"]
            self.memory_capacity = hyperparams["memory_capacity"]
            self.covariance = hyperparams["covariance"]

        elif alg == "sac":
            # SAC-specific (continuous)
            self.policy          = hyperparams["policy"]
            self.learning_rate   = hyperparams["learning_rate"]
            self.buffer_size     = hyperparams["buffer_size"]
            self.batch_size      = hyperparams["batch_size"]
            self.tau             = hyperparams["tau"]
            self.gamma           = hyperparams["gamma"]
            self.train_freq      = hyperparams["train_freq"]
            self.gradient_steps  = hyperparams["gradient_steps"]
            self.ent_coef        = hyperparams["ent_coef"]
            self.verbose         = hyperparams.get("verbose", 1)
            self.total_timesteps = int(hyperparams.get("total_timesteps", 0))
            self.n_envs        = hyperparams["n_envs"]

        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")

        # Training control parameters
        self.max_episodes = hyperparams.get("max_episodes", 0)
        self.max_steps    = hyperparams.get("max_steps", None)
        self.render_fps   = hyperparams.get("render_fps", None)

        # Create directories for plots and videos
        os.makedirs(self.plot_path, exist_ok=True)
        os.makedirs(self.video_path, exist_ok=True)

        # Initialize base gym environment
        if env_name == "FrozenLake-v1":
            self.map_size    = hyperparams.get("map_size", 8)
            self.num_states  = hyperparams.get("num_states", self.map_size ** 2)
            base_env = gym.make(
                env_name,
                map_name=f"{self.map_size}x{self.map_size}",
                is_slippery=False,
                max_episode_steps=self.max_steps,
                render_mode="rgb_array" if not self.train_mode else None
            )
        elif env_name == "MountainCar-v0":
            base_env = gym.make(
                env_name,
                max_episode_steps=self.max_steps,
                render_mode="rgb_array" if not self.train_mode else None
            )
        elif env_name == "HalfCheetah-v4" :
            base_env = gym.make(
                env_name,
                max_episode_steps=self.max_steps,
                render_mode="rgb_array" if not self.train_mode else None
            )
        elif env_name == "Pendulum-v1" :
            base_env = gym.make(
                env_name,
                max_episode_steps=self.max_steps,
                render_mode="rgb_array" if not self.train_mode else None
            )
        else:
            raise ValueError(f"Unsupported environment: {env_name}")
        base_env.metadata['render_fps'] = self.render_fps

        # Wrap environment
        wrapped_env = step_wrapper(base_env, env_name,
                                   num_states=getattr(self, 'num_states', None))

        # In the __init__ method of Model_TrainTest
        if not self.train_mode:
            self.env = RecordVideo(
                wrapped_env,
                video_folder=self.video_path,
                name_prefix=f"{self.algorithm}_{self.env_name}",
                episode_trigger=lambda episode_id: episode_id == 0  # Record only the first episode
            )
        else:
            self.env = wrapped_env

        # Instantiate agent based on algorithm
        if algorithm.lower() == "dqn":
            self.agent = DQN_Agent(
                env=self.env,
                epsilon_max=self.epsilon_max,
                epsilon_min=self.epsilon_min,
                epsilon_decay=self.epsilon_decay,
                clip_grad_norm=self.clip_grad_norm,
                learning_rate=self.learning_rate,
                discount=self.discount_factor,
                memory_capacity=self.memory_capacity,
                env_name=env_name ,
                batch_size = self.batch_size,
                update_frequency=self.update_frequency,
                plot_path=self.plot_path,
                map_size=self.map_size
            )
        elif alg == "a2c":
            self.agent = A2C_Agent(
                env_name=env_name,
                learning_rate=self.learning_rate,
                gamma=self.gamma,
                n_steps=self.n_steps,
                gae_lambda=self.gae_lambda,
                ent_coef=self.ent_coef,
                vf_coef=self.vf_coef,
                max_grad_norm=self.max_grad_norm,
                verbose=self.verbose,
                policy=self.policy,
                plot_path = self.plot_path,
                RL_load_path = self.RL_load_path,
                video_path = self.video_path,
                max_episodes = self.max_episodes,
                map_size=self.map_size
            )

        elif algorithm.lower() == "naf":
            self.agent = NAF_Agent(
                env_name=env_name,
                env_fn=base_env,
                max_episodes=self.max_episodes,
                total_timesteps=self.total_timesteps,
                plot_path=self.plot_path,
                video_path=self.video_path,
                RL_load_path=self.RL_load_path,
                hidden_sizes=self.hidden_sizes,
                learning_rate=self.learning_rate,
                gamma=self.gamma,
                tau=self.tau,
                epsilon=self.epsilon,
                batch_size=self.batch_size,
                memory_capacity=self.memory_capacity,
                covariance=self.covariance,
                max_steps = self.max_steps,
            )

        elif algorithm.lower() == "sac":
            self.agent = SAC_Agent(
                env_name=env_name,
                policy=self.policy,
                learning_rate=self.learning_rate,
                buffer_size=self.buffer_size,
                batch_size=self.batch_size,
                tau=self.tau,
                gamma=self.gamma,
                train_freq=self.train_freq,
                gradient_steps=self.gradient_steps,
                ent_coef=self.ent_coef,
                verbose=self.verbose,
                plot_path=self.plot_path,
                video_path=self.video_path,
                RL_load_path=self.RL_load_path,
                max_episodes=self.max_episodes,
                total_timesteps=self.total_timesteps,
                n_envs=self.n_envs,
                max_steps=self.max_steps,
                save_interval = self.save_interval
            )
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")

    def train(self):
        """Run training by calling the agent's train method."""
        self.agent.train(
            max_episodes=self.max_episodes,
            max_steps=self.max_steps,
            save_interval=self.save_interval,
            save_path=self.RL_load_path,
        )

    def test(self, max_episodes):
        """Run testing by calling the agent's test method."""
        self.agent.test(
            max_episodes=max_episodes,
            max_steps=self.max_steps,
            load_path=self.RL_load_path
        )

    def run(self, max_episodes):
        """Execute training or testing based on mode."""
        if self.train_mode:
            self.train()
        else:
            self.test(max_episodes)