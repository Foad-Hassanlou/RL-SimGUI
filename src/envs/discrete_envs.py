# discrete_envs.py
import gymnasium as gym
import numpy as np
import torch
from sympy.codegen.ast import continue_

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class step_wrapper(gym.Wrapper):
    def __init__(self, env, env_name, num_states=None):
        super().__init__(env)
        self.env_name = env_name
        self.num_states = num_states
        if env_name == "MountainCar-v0":
            self.observation_wrapper = observation_wrapper_mountaincar(env)
            self.reward_wrapper = reward_wrapper_mountaincar(env)
        elif env_name == "FrozenLake-v1":
            self.observation_wrapper = observation_wrapper_frozenlake(env, num_states)
            self.reward_wrapper = reward_wrapper_frozenlake(env)
        elif env_name == "Pendulum-v1" or env_name == "HalfCheetah-v4":
            pass

        else:
            raise ValueError(f"Unsupported environment: {env_name}")

    def step(self, action):
        state, reward, done, truncation, info = self.env.step(action)
        modified_state = self.observation_wrapper.observation(state)
        if self.env_name == "MountainCar-v0":
            modified_reward = self.reward_wrapper.reward(modified_state)
        elif self.env_name == "FrozenLake-v1":  # FrozenLake
            modified_reward = reward
        return modified_state, modified_reward, done, truncation, info

    def reset(self, *args, **kwargs):
        state, info = self.env.reset(*args, **kwargs)
        modified_state = self.observation_wrapper.observation(state)
        return modified_state, info

class observation_wrapper_mountaincar(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.min_value = env.observation_space.low
        self.max_value = env.observation_space.high

    def observation(self, state):
        normalized_state = (state - self.min_value) / (self.max_value - self.min_value)
        return normalized_state

class reward_wrapper_mountaincar(gym.RewardWrapper):
    def __init__(self, env):
        super().__init__(env)

    def reward(self, state):
        current_position, current_velocity = state  # extract the position and current velocity based on the state

        # Interpolate the value to the desired range (because the velocity normalized value would be in range of 0 to 1 and now it would be in range of -0.5 to 0.5)
        current_velocity = np.interp(current_velocity, np.array([0, 1]), np.array([-0.5, 0.5]))

        # (1) Calculate the modified reward based on the current position and velocity of the car.
        degree = current_position * 360
        degree2radian = np.deg2rad(degree)
        modified_reward = 0.2 * (np.cos(degree2radian) + 2 * np.abs(current_velocity))

        # (2) Step limitation
        modified_reward -= 0.5  # Subtract 0.5 to adjust the base reward (to limit useless steps).

        # (3) Check if the car has surpassed a threshold of the path and is closer to the goal
        if current_position > 0.98:
            modified_reward += 20  # Add a bonus reward (Reached the goal)
        elif current_position > 0.92:
            modified_reward += 10  # So close to the goal
        elif current_position > 0.82:
            modified_reward += 6  # car is closer to the goal
        elif current_position > 0.65:
            modified_reward += 1 - np.exp(
                -2 * current_position)  # car is getting close. Thus, giving reward based on the position and the further it reached

        # (4) Check if the car is coming down with velocity from left and goes with full velocity to right
        initial_position = 0.40842572  # Normalized value of initial position of the car which is extracted manually

        if current_velocity > 0.3 and current_position > initial_position + 0.1:
            modified_reward += 1 + 2 * current_position  # Add a bonus reward for this desired behavior

        return modified_reward

class observation_wrapper_frozenlake(gym.ObservationWrapper):
    def __init__(self, env, num_states):
        super().__init__(env)
        self.num_states = num_states

    def observation(self, state):
        onehot_vector = torch.zeros(self.num_states, dtype=torch.float32, device=device)
        onehot_vector[state] = 1
        return onehot_vector

class reward_wrapper_frozenlake(gym.RewardWrapper):
    def __init__(self, env):
        super().__init__(env)

    def reward(self, reward):
        return reward