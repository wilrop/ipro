import numbers
import numpy as np
import gymnasium as gym


class OneHotObservations(gym.Wrapper):
    """Return one hot observations in environments with a discrete state space."""

    def __init__(self, env: gym.Env):
        gym.Wrapper.__init__(self, env)
        if isinstance(self.env.observation_space, gym.spaces.Discrete):
            self.env_shape = (self.env.observation_space.n,)
        elif isinstance(self.env.observation_space, gym.spaces.MultiDiscrete):
            self.env_shape = self.env.observation_space.nvec
        elif (
                isinstance(self.env.observation_space, gym.spaces.Box)
                and self.env.observation_space.is_bounded(manner="both")
                and issubclass(self.env.observation_space.dtype.type, numbers.Integral)
        ):
            low_bound = np.array(self.env.observation_space.low)
            high_bound = np.array(self.env.observation_space.high)
            self.env_shape = high_bound - low_bound + 1
        else:
            raise Exception("The one hot observations wrapper only supports discretizable observation spaces.")

        self.num_states = np.prod(self.env_shape)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(self.num_states,), dtype=int)

    def one_hot_encode(self, obs):
        """One-hot encode the given observation.

        Args:
            obs (ndarray): The observation to one-hot encode.

        Returns:
            ndarray: The one-hot encoded observation.
        """
        one_hot_obs = np.zeros(self.num_states, dtype=int)
        one_hot_obs[np.ravel_multi_index(obs, self.env_shape)] = 1
        return one_hot_obs

    def step(self, action):
        """Step the environment."""
        observation, reward, terminated, truncated, info = self.env.step(action)
        return self.one_hot_encode(observation), reward, terminated, truncated, info

    def reset(self, seed=None, **kwargs):
        """Reset the environment."""
        observation, info = self.env.reset(seed=seed, **kwargs)
        return self.one_hot_encode(observation), info
