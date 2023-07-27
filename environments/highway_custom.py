import numpy as np
from gymnasium.spaces import Box
from mo_gymnasium.envs.highway.highway import MOHighwayEnvFast


class HighwayCustom(MOHighwayEnvFast):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reward_space = Box(low=np.array([0.0, 0.0]), high=np.array([1.0, 1.0]), shape=(2,), dtype=np.float32)
        self.reward_dim = 2

    def step(self, action):
        """Drop the collision reward."""
        obs, vec_reward, terminated, truncated, info = super().step(action)
        return obs, vec_reward[:-1], terminated, truncated, info
