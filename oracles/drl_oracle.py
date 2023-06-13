import torch
import numpy as np

from oracles.vector_u import create_batched_aasf
from gymnasium.spaces import Box


class DRLOracle:
    def __init__(self,
                 env,
                 aug=0.2,
                 gamma=0.99,
                 one_hot=False,
                 eval_episodes=100):
        self.env = env
        self.aug = aug

        self.num_actions = env.action_space.n
        self.num_objectives = env.reward_space.shape[0]

        if isinstance(self.env.observation_space, Box):
            low_bound = self.env.observation_space.low
            high_bound = self.env.observation_space.high
            self.obs_shape = self.env.observation_space.shape
            if one_hot:
                self.box_shape = (high_bound[0] - low_bound[0] + 1, high_bound[1] - low_bound[1] + 1)
                self.obs_dim = np.prod(self.box_shape)
            else:
                self.obs_dim = np.prod(self.obs_shape)

        self.gamma = gamma
        self.one_hot = one_hot
        self.eval_episodes = eval_episodes
        self.u_func = None
        self.trained_models = {}  # Collection of trained models that can be used for warm-starting.

    def reset(self):
        """Reset the environment and the agent."""
        raise NotImplementedError

    def select_greedy_action(self, obs, accrued_reward):
        """Select the greedy action for the given observation."""
        raise NotImplementedError

    def one_hot_encode(self, obs):
        """One-hot encode the given observation.

        Args:
            obs (ndarray): The observation to one-hot encode.

        Returns:
            ndarray: The one-hot encoded observation.
        """
        dims = obs.ndim
        if dims == 1:
            obs = np.expand_dims(obs, axis=0)
        num_obs = len(obs)
        obs = np.swapaxes(obs, 0, 1)
        flat_obs = np.ravel_multi_index(obs, self.box_shape)
        one_hot_obs = np.zeros((num_obs, self.obs_dim))
        one_hot_obs[np.arange(num_obs), flat_obs] = 1
        if dims == 1:
            one_hot_obs = np.squeeze(one_hot_obs, axis=0)
        return one_hot_obs

    def format_obs(self, obs):
        """Format the given observation.

        Args:
            obs (ndarray): The observation to format.

        Returns:
            ndarray: The formatted observation.
        """
        if self.one_hot:
            return self.one_hot_encode(obs)
        else:
            return obs.flatten()

    def evaluate(self):
        """Evaluate MODQN on the given environment."""
        pareto_point = np.zeros(self.num_objectives)

        for episode in range(self.eval_episodes):
            raw_obs, _ = self.env.reset()
            obs = self.format_obs(raw_obs)
            terminated = False
            truncated = False
            accrued_reward = np.zeros(self.num_objectives)
            timestep = 0

            while not (terminated or truncated):
                action = self.select_greedy_action(obs, accrued_reward)
                next_raw_obs, reward, terminated, truncated, _ = self.env.step(action)
                next_obs = self.format_obs(next_raw_obs)
                accrued_reward += (self.gamma ** timestep) * reward
                obs = next_obs
                timestep += 1

            pareto_point += accrued_reward

        return pareto_point / self.eval_episodes

    def train(self):
        """Train the algorithm on the given environment."""
        raise NotImplementedError

    def get_closest_referent(self, referent):
        """Get the processed referent closest to the given referent.

        Args:
            referent (ndarray): The referent to get the closest processed referent for.

        Returns:
            ndarray: The closest processed referent.
        """
        referents = list(self.trained_models.keys())
        if len(referents) == 0:
            return False
        distances = np.array([np.linalg.norm(np.array(referent) - np.array(r)) for r in referents])
        return referents[np.argmin(distances)]

    def solve(self, referent, ideal):
        """Run the inner loop of the outer loop."""
        referent = torch.tensor(referent)
        ideal = torch.tensor(ideal)
        self.u_func = create_batched_aasf(referent, referent, ideal, aug=self.aug, backend='torch')
        self.train()
        pareto_point = self.evaluate()
        return pareto_point
