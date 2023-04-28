import numpy as np
from mo_gymnasium import LinearReward


class StableBaselines:
    """A linear solver for the outer loop initial phase."""

    def __init__(self, env):
        self.env = env
        self.num_objectives = 2
        self.gamma = 0.99
        self.eval_episodes = 10
        self.buffer = []  # Keep experiences to pre-train the model.

    def train_model(self, env):
        """Train a model on a scalarised reward environment.

        Args:
            env (LinearReward): A scalarised reward environment.

        Returns:
            Model: A trained model.
        """
        model = PPO(env)
        model.learn(total_timesteps=1000)
        trajectories = []
        self.buffer.append(trajectories)
        return model

    def evaluate(self, model):
        """Evaluate the model on the vector environment.

        Args:
            model (Model): A trained model.

        Returns:
            ndarray: A Pareto point.
        """
        pareto_point = np.zeros(self.num_objectives)

        for episode in range(self.eval_episodes):
            state, _ = self.env.reset()
            terminated = False
            truncated = False
            accrued_reward = np.zeros(self.num_objectives)
            timestep = 0

            while not (terminated or truncated):
                action, _ = model.predict(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                accrued_reward += (self.gamma ** timestep) * reward
                state = next_state
                timestep += 1

            pareto_point += accrued_reward

        return pareto_point / self.eval_episodes

    def solve(self, weights):
        """Learn an optimal policy for the given weights.

        Args:
            weights: A list of weights for the objectives.

        Returns:
            A policy that maximizes the weighted sum of the objectives.
        """
        env = LinearReward(self.env, weights)
        model = self.train_model(env)
        pareto_point = self.evaluate(model)
        return pareto_point
