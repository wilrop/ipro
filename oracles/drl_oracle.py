import torch
import numpy as np
from oracles.vector_u import create_batched_aasf


class DRLOracle:
    def __init__(self,
                 env,
                 aug=0.2,
                 gamma=0.99,
                 eval_episodes=100):
        self.env = env
        self.aug = aug
        self.num_actions = env.action_space.n
        self.num_objectives = env.reward_space.shape[0]
        self.gamma = gamma
        self.eval_episodes = eval_episodes
        self.u_func = None
        self.trained_models = {}

    def reset(self):
        """Reset the environment and the agent."""
        raise NotImplementedError

    def select_greedy_action(self, state, accrued_reward):
        """Select the greedy action for the given state."""
        raise NotImplementedError

    def evaluate(self):
        """Evaluate MODQN on the given environment."""
        pareto_point = np.zeros(self.num_objectives)

        for episode in range(self.eval_episodes):
            state, _ = self.env.reset()
            terminated = False
            truncated = False
            accrued_reward = np.zeros(self.num_objectives)
            timestep = 0

            while not (terminated or truncated):
                action = self.select_greedy_action(state, accrued_reward)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                accrued_reward += (self.gamma ** timestep) * reward
                state = next_state
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
