#continuous_envs.py
import gymnasium as gym
import numpy as np

def shaped_reward_Pendulum(state, action, raw_reward):
    """
    Reward shaping for Pendulum-v1.

    Parameters:
    - state: [cos(theta), sin(theta), theta_dot]
    - action: [torque] (action in [-1, 1] before scaling to [-2, 2])
    - raw_reward: Original reward from the environment (-(theta^2 + 0.1*theta_dot^2 + 0.001*action^2))

    Returns:
    - Shaped reward
    """
    # Extract angle (theta) and angular velocity (theta_dot)
    theta = np.arctan2(state[1], state[0])  # Compute angle from cos and sin
    theta_dot = state[2]
    torque = action[0]  # Action is 1D

    # Upright bonus: Encourage pendulum to be upright (theta close to 0)
    upright_bonus = 2.0 * (1.0 + np.cos(theta))  # Max: 4.0 (at theta=0), Min: 0.0 (at theta=pi)

    # Angular velocity penalty: Discourage high angular velocity
    angular_penalty = -0.05 * theta_dot ** 2

    # Control effort penalty: Discourage large torques
    energy_penalty = -0.01 * torque ** 2

    # Combine components
    shaped = upright_bonus + angular_penalty + energy_penalty

    # Blend with raw reward to maintain environment's scale
    return 0.5 * raw_reward + 0.5 * shaped


# Custom wrapper for reward shaping in Pendulum-v1
class PendulumRewardWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env

    def step(self, action):
        # Get the original step results
        obs, reward, terminated, truncated, info = self.env.step(action)
        # Apply shaped reward
        shaped_reward = shaped_reward_Pendulum(obs, action, reward)
        return obs, shaped_reward, terminated, truncated, info