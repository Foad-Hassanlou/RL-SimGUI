# confing.py

from pprint import pprint

def get_hyperparameters(algorithm, env_name, mode, map_size):
    """
    Returns hyperparameters for the specified algorithm, environment, and mode.
    Supports DQN and A2C for MountainCar-v0 and FrozenLake-v1.
    Supports NAF and SAC for Pendulum-v1 and HalfCheetah-v4.
    """
    base_path = '/path/to/your/project/Project_Code' 

    # Default flags
    hyperparams = {
        "algorithm": algorithm,
        "train_mode": mode == "train",
        "render": mode == "test",
    }

    # MountainCar-v0 (Discrete)
    if env_name == "MountainCar-v0":
        action_space = "Discrete"
        model_dir = f"{base_path}/models/{action_space}/{env_name}/{algorithm}"
        hyperparams.update({
            "RL_load_path": f"{model_dir}/final_weights_{algorithm}_{env_name}_2000.pth",
            "save_path":      f"{model_dir}",
            "plot_path":      f"{base_path}/plots/learning_curves/{action_space}/{env_name}/{algorithm}",
            "video_path":     f"{base_path}/videos/{action_space}/{env_name}/{algorithm}",
        })

        common = {
            "max_episodes": 2000 if mode == "train" else 2,
            "max_steps":    200,
            "render_fps":   60,
            "save_interval": 2000,

        }

        if algorithm == "DQN":
            hyperparams.update({
                **common,
                "clip_grad_norm":   5,
                "learning_rate":    75e-5,
                "discount_factor":  0.96,
                "batch_size":       64,
                "update_frequency": 20,
                "epsilon_max":      0.999 if mode == "train" else -1,
                "epsilon_min":      0.01,
                "epsilon_decay":    0.997,
                "memory_capacity":  125_000 if mode == "train" else 0,
            })
        elif algorithm == "A2C":
            hyperparams.update({
                **common,
                # A2C-specific defaults
                "policy": "MlpPolicy",
                "learning_rate": 1e-3,
                "n_steps" : 5,
                "gamma": 0.96,
                "gae_lambda": 0.95,
                "ent_coef" : 0.07,
                "vf_coef" : 0.5,
                "max_grad_norm": 0.5,
                "verbose" : 1,
                "total_timesteps" : 1e9
            })
        else:
            raise ValueError(f"Unsupported algorithm '{algorithm}' for {env_name}")

    # FrozenLake-v1 (Discrete)
    elif env_name == "FrozenLake-v1":
        action_space = "Discrete"
        model_dir = f"{base_path}/models/{action_space}/{env_name}/{algorithm}/{map_size}x{map_size}"
        hyperparams.update({
            "RL_load_path":   f"{model_dir}/final_weights_{algorithm}_{env_name}_{map_size}x{map_size}_3000.pth",
            "save_path":      f"{model_dir}",
            "plot_path":      f"{base_path}/plots/learning_curves/{action_space}/{env_name}/{algorithm}/{map_size}x{map_size}",
            "video_path":     f"{base_path}/videos/{action_space}/{env_name}/{algorithm}/{map_size}x{map_size}",
            "map_size":       map_size,
            "num_states":     map_size ** 2,
        })

        common = {
            "max_episodes": 3000 if mode == "train" else 5,
            "max_steps":    200,
            "render_fps":   6,
            "save_interval": 3000,
        }

        if algorithm == "DQN":
            hyperparams.update({
                **common,
                "clip_grad_norm":   3,
                "learning_rate":    6e-4,
                "discount_factor":  0.93,
                "batch_size":       32,
                "update_frequency": 10,
                "epsilon_max":      0.999 if mode == "train" else -1,
                "epsilon_min":      0.01,
                "epsilon_decay":    0.999,
                "memory_capacity":  4_000 if mode == "train" else 0,
            })
        elif algorithm == "A2C":
            hyperparams.update({
                **common,
                # A2C-specific defaults
                "policy": "MlpPolicy",
                "learning_rate": 7e-4,
                "n_steps" : 5,
                "gamma": 0.99,
                "gae_lambda": 0.95,
                "ent_coef" : 0.01,
                "vf_coef" : 0.5,
                "max_grad_norm": 0.5,
                "verbose" : 1,
                "total_timesteps" : 1e9
            })
        else:
            raise ValueError(f"Unsupported algorithm '{algorithm}' for {env_name}")

    # Pendulum-v1 (Continuous)
    elif env_name == "Pendulum-v1":
        action_space = "Continuous"
        model_dir = f"{base_path}/models/{action_space}/{env_name}/{algorithm}"
        hyperparams.update({
            "RL_load_path": f"{model_dir}/final_weights_{algorithm}_{env_name}_2000.pth",
            "save_path":      f"{model_dir}",
            "plot_path":      f"{base_path}/plots/learning_curves/{action_space}/{env_name}/{algorithm}",
            "video_path":     f"{base_path}/videos/{action_space}/{env_name}/{algorithm}",
        })

        common = {
            "max_episodes": 2000 if mode == "train" else 5,
            "max_steps": 200,
            "render_fps": 45,
            "save_interval": 2000,
            "total_timesteps": 4e5,
        }

        if algorithm == "NAF":
            hyperparams.update({
                **common,
                # NAF-specific hyperparameters
                "hidden_sizes": [128, 64],
                "learning_rate": 3e-4,
                "gamma": 0.99,
                "tau": 0.005,
                "epsilon": 0.2,
                "batch_size": 64,
                "memory_capacity": 400000,
                "covariance": "identity",
            })
        elif algorithm == "SAC":
            hyperparams.update({
                **common,
                "policy": "MlpPolicy",
                "learning_rate": 3e-4,
                "buffer_size": 400000,
                "batch_size": 256,
                "tau": 0.005,
                "gamma": 0.99,
                "train_freq": 1,
                "gradient_steps": 1,
                "ent_coef": "auto",
                "verbose": 1,
                "n_envs" : 2
            })
        else:
            raise ValueError(f"Unsupported algorithm '{algorithm}' for {env_name}")

    # HalfCheetah-v4 (Continuous)
    elif env_name == "HalfCheetah-v4":
        action_space = "Continuous"
        model_dir = f"{base_path}/models/{action_space}/{env_name}/{algorithm}"
        hyperparams.update({
            "RL_load_path": f"{model_dir}/final_weights_{algorithm}_{env_name}_1500.pth",
            "save_path":      f"{model_dir}",
            "plot_path":      f"{base_path}/plots/learning_curves/{action_space}/{env_name}/{algorithm}",
            "video_path":     f"{base_path}/videos/{action_space}/{env_name}/{algorithm}",
        })

        common = {
            "max_episodes": 1500 if mode == "train" else 5,
            "max_steps": 1000,
            "render_fps": 20,
            "save_interval": 1500,
            "total_timesteps": 1500000,
        }

        if algorithm == "NAF":
            hyperparams.update({
                **common,
                # NAF-specific hyperparameters
                "hidden_sizes": [256, 256],
                "learning_rate": 3e-4,
                "gamma": 0.99,
                "tau": 0.005,
                "epsilon": 0.2,
                "batch_size": 256,
                "memory_capacity": 150000,
                "covariance": "diagonal",
            })
        elif algorithm == "SAC":
            hyperparams.update({
                **common,
                "policy": "MlpPolicy",
                "learning_rate": 3e-4,
                "buffer_size": 1000000,
                "batch_size": 256,
                "tau": 0.005,
                "gamma": 0.99,
                "train_freq": 1,
                "gradient_steps": 2,
                "ent_coef": "auto",
                "verbose": 1,
                "n_envs": 2
            })
        else:
            raise ValueError(f"Unsupported algorithm '{algorithm}' for {env_name}")

    else:
        raise ValueError(f"Unsupported environment: {env_name}")

    return hyperparams


def check(hyperparams):
    """
    Nicely prints storage paths and hyperparameters.
    Tries to use pandas for a tabular view if available, otherwise falls back to pprint.
    """
    try:
        import pandas as pd
        df = pd.DataFrame(list(hyperparams.items()), columns=['Parameter', 'Value'])
        print("\n=== Hyperparameters & Paths ===")
        print(df.to_string(index=False))
    except ImportError:
        print("\n=== Hyperparameters & Paths ===")
        pprint(hyperparams)

def printall():
    # Iterate over all environments, algorithms, and modes
    env_alg_mapping = {
        'MountainCar-v0': ['DQN', 'A2C'],
        'FrozenLake-v1': ['DQN', 'A2C'],
        'Pendulum-v1': ['NAF', 'SAC'],
        'HalfCheetah-v4': ['NAF', 'SAC'],
    }
    modes = ['train', 'test']

    for env_name, algos in env_alg_mapping.items():
        for algorithm in algos:
            for mode in modes:
                print(f"\n## Env: {env_name}, Algorithm: {algorithm}, Mode: {mode}")
                hp = get_hyperparameters(algorithm, env_name, mode)
                check(hp)
