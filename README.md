# Reinforcement Learning Experiment Framework with GUI

This project provides a graphical user interface (GUI) to train and test various reinforcement learning (RL) algorithms on different environments. It supports both discrete and continuous action spaces and includes algorithms like DQN, A2C, NAF, and SAC.

## Features

- **Action Spaces**: Discrete and Continuous
- **Algorithms**:
  - Discrete: DQN, A2C
  - Continuous: NAF, SAC
- **Environments**:
  - Discrete: MountainCar-v0, FrozenLake-v1 (4x4 and 8x8 maps)
  - Continuous: Pendulum-v1, HalfCheetah-v4
- **Modes**: Train and Test
- **GUI Controls**:
  - Run experiments
  - View learning curves and plots
  - Watch videos of agent performance
  - Compare reward curves of different algorithms
  - View hyperparameters for selected configurations
  - View project directory structure

## Requirements

- Python 3.9 or higher
- Dependencies are listed in the following files:
  - `requirements/base.txt`
  - `requirements/dev.txt`
  - `requirements/env.txt`

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/Foad-Hassanlou/RL-SimGUI.git
   cd RL-SimGUI.git
   ```

2. Install the required dependencies using the provided requirements files:
   ```
   pip install -r requirements/base.txt
   pip install -r requirements/dev.txt
   pip install -r requirements/env.txt
   ```

3. **Note**: You may need to adjust the `base_path` in `requirements/config.py` to match your local directory structure.

## Usage

1. Run the GUI:
   ```
   python main.py
   ```

2. In the GUI:
   - Select the action space (Discrete or Continuous).
   - Choose an algorithm from the available options.
   - Select an environment.
   - Choose the mode (Train or Test).
   - If using FrozenLake-v1, select the map size (4x4 or 8x8).
   - Click "Run" to start the experiment.
   - Use other buttons to view plots, videos, compare algorithms, etc.

3. During training, monitor the console output for progress. After training, plots and models will be saved in the respective directories.

4. For testing, the agent will run in the environment, and a video will be recorded.

## Project Structure

```
Project_Code/
├── plots/
│   └── learning_curves/
│   │   ├── Continuous/
│   │   │    ├── Pendulum-v1/
│   │   │    │   ├── SAC/
│   │   │    │   └── NAF/
│   │   │    └── HalfCheetah-v4/
│   │   │       ├── SAC/
│   │   │       └── NAF/
│   │   └── Discrete/
│   │       ├── FrozenLake-v1/
│   │       │   ├── DQN/
│   │       │   │   ├── 4x4
│   │       │   │   └── 8x8
│   │       │   └── A2C/
│   │       │       ├── 4x4
│   │       │       └── 8x8
│   │       └── MountainCar-v0/
│   │           ├── DQN/
│   │           └── A2C/
│   ├── comparison_table.png
│   └── directory_structure.png
├── requirements/
│   ├── base.txt
│   ├── dev.txt
│   ├── env.txt
│   └── config.py
├── src/
│   ├── agents/
│   │   ├── a2c.py
│   │   ├── dqn.py
│   │   ├── naf.py
│   │   └── sac.py
│   ├── envs/
│   │   ├── continuous_envs.py
│   │   └── discrete_envs.py
│   ├── main.py
│   ├── train.py
│   └── utils.py
├── videos/
│   ├── Continuous/
│   │    ├── Pendulum-v1/
│   │    │   ├── SAC/
│   │    │   └── NAF/
│   │    └── HalfCheetah-v4/
│   │       ├── SAC/
│   │       └── NAF/
│   └── Discrete/
│       ├── FrozenLake-v1/
│       │   ├── DQN/
│       │   │   ├── 4x4
│       │   │   └── 8x8
│       │   └── A2C/
│       │       ├── 4x4
│       │       └── 8x8
│       └── MountainCar-v0/
│           ├── DQN/
│           └── A2C/
├── models/
│   ├── Continuous/
│   │    ├── Pendulum-v1/
│   │    │   ├── SAC/
│   │    │   └── NAF/
│   │    └── HalfCheetah-v4/
│   │       ├── SAC/
│   │       └── NAF/
│   └── Discrete/
│       ├── FrozenLake-v1/
│       │   ├── DQN/
│       │   │   ├── 4x4
│       │   │   └── 8x8
│       │   └── A2C/
│       │       ├── 4x4
│       │       └── 8x8
│       └── MountainCar-v0/
│           ├── DQN/
│           └── A2C/
└── README.md
```

This project was developed as part of the Deep Reinforcement Learning course at Sharif University of Technology under the supervision of [Dr. Mohammad Hossein Rohban](https://scholar.google.com/citations?user=pRyJ6FkAAAAJ&hl=en).

## References

- **A2C (Advantage Actor-Critic)**:
  - Mnih, V., et al. (2016). "Asynchronous Methods for Deep Reinforcement Learning." *ICML 2016*. [Paper](https://arxiv.org/abs/1602.01783)
  - Stable-Baselines3 A2C Implementation: [GitHub](https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/a2c/a2c.py)

- **SAC (Soft Actor-Critic)**:
  - Haarnoja, T., et al. (2018). "Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor." *ICML 2018*. [Paper](https://arxiv.org/abs/1801.01290)
  - Stable-Baselines3 SAC Implementation: [GitHub](https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/sac/sac.py)

- **DQN (Deep Q-Network)**:
  - Mnih, V., et al. (2015). "Human-level control through deep reinforcement learning." *Nature*, 518(7540), 529-533. [Paper](https://www.nature.com/articles/nature14236)
  - Stable-Baselines3 DQN Implementation: [GitHub](https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/dqn/dqn.py)
  - DQN for FrozenLake-v1 (used in this project, licensed under MIT): [GitHub](https://github.com/MehdiShahbazi/DQN-Frozenlake-Gymnasium)
  - DQN for MountainCar-v0 (used in this project, licensed under MIT): [GitHub](https://github.com/MehdiShahbazi/DQN-Mountain-Car-Gymnasium)

- **NAF (Normalized Advantage Function)**:
  - Gu, S., et al. (2016). "Continuous Deep Q-Learning with Model-based Acceleration." *ICML 2016*. [Paper](https://arxiv.org/abs/1603.00748)

- **Gymnasium**:
  - Official Gymnasium Documentation: [Gymnasium](https://gymnasium.farama.org/)