# src/normalization.py
import gymnasium as gym
import numpy as np
from src import config

class NormalizedEnv(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        obs_shape = self.env.observation_space.shape
        self.obs_running_mean = np.zeros(obs_shape, dtype=np.float64)
        self.obs_running_var = np.ones(obs_shape, dtype=np.float64)
        self.obs_count = 1e-4  # Small constant to avoid division by zero
        
        # Configuration flags
        self.normalize_obs = config.OBS_NORMALIZATION
        self.normalize_rew = config.REWARD_NORMALIZATION

    def observation(self, observation):
        if self.normalize_obs:
            self.obs_count += 1
            last_mean = self.obs_running_mean.copy()
            # Update running mean
            self.obs_running_mean += (observation - self.obs_running_mean) / self.obs_count
            # Update running variance
            self.obs_running_var += (observation - last_mean) * (observation - self.obs_running_mean)
            std = np.sqrt(self.obs_running_var / self.obs_count)
            # Normalize observation
            normalized_obs = (observation - self.obs_running_mean) / (std + 1e-8)
            return normalized_obs
        return observation

    def reward(self, reward):
        if self.normalize_rew:
            # Simple reward normalization: scale by a constant factor. 
            # More sophisticated normalization can be implemented as needed.
            return reward / (abs(reward) + 1e-8)
        return reward

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        return self.observation(observation), info

    def step(self, action):
        observation, reward, done, truncated, info = self.env.step(action)
        return self.observation(observation), self.reward(reward), done, truncated, info