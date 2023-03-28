import numpy as np
import torch

from vector_u import create_fast_translated_rectangle_u


class OnlineMirrorDescent:
    def __init__(self,
                 env,
                 init_policy,
                 init_state_dist,
                 num_states,
                 num_actions,
                 num_objectives,
                 measure_iters=100000,
                 train_iters=500,
                 q_iters=50000,
                 eval_episodes=20,
                 gamma=0.9,
                 q_alpha=0.1,
                 model=None,
                 box=False,
                 env_shape=None,
                 seed=None):
        self.env = env
        self.init_policy = init_policy
        self.init_state_dist = init_state_dist

        self.num_states = num_states
        self.num_actions = num_actions
        self.num_objectives = num_objectives

        self.measure_iters = measure_iters
        self.train_iters = train_iters
        self.q_iters = q_iters
        self.eval_episodes = eval_episodes

        self.gamma = gamma
        self.q_alpha = q_alpha

        self.box = box
        self.env_shape = env_shape

        self.rng = np.random.default_rng(seed)

        if model is None:
            self.reward_matrix = np.zeros((self.num_states, self.num_actions, self.num_objectives))
            self.transition_matrix = np.zeros((self.num_states, self.num_actions, self.num_states))
            self.terminal_states = []
            self.estimate_mdp()
        else:
            self.reward_matrix = model.reward_matrix
            self.transition_matrix = model.transition_matrix
            self.terminal_states = model.terminal_states

    def select_action(self, state, policy):
        """Select an action according to the given policy.

        Args:
            state (int): The current state.
            policy (ndarray): A policy matrix.

        Returns:
            int: The next action.
        """
        return self.rng.choice(self.num_actions, p=policy[state])

    def select_exploratory_action(self, state):
        """Select an action according to the given policy.

        Args:
            state (int): The current state.

        Returns:
            int: The next action.
        """
        action_counters = np.sum(self.transition_matrix[state, :, :], axis=-1)
        explore_actions = np.argwhere(action_counters == min(action_counters)).flatten()
        return self.rng.choice(explore_actions)

    def estimate_mdp(self):
        """Learn a model of the MDP from the environment."""
        if self.box:
            format_state = lambda s: int(np.ravel_multi_index(s, self.env_shape))
        else:
            format_state = lambda s: s

        state, _ = self.env.reset()
        state = format_state(state)
        for i in range(self.measure_iters):
            action = self.select_exploratory_action(state)
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            next_state = format_state(next_state)
            self.transition_matrix[state, action, next_state] += 1
            self.reward_matrix[state, action] += reward
            if terminated or truncated:  # If the episode is done, reset the environment.
                state, _ = self.env.reset()
                state = format_state(state)
            else:
                state = next_state

        self.reward_matrix /= np.sum(self.transition_matrix, axis=-1, keepdims=True)  # Average rewards.
        self.reward_matrix = np.nan_to_num(self.reward_matrix, nan=0)  # Replace NaNs.
        for state in range(self.num_states):
            if np.sum(self.transition_matrix[state]) == 0:
                self.terminal_states.append(state)
                for action in range(self.num_actions):
                    self.transition_matrix[state, action] = np.copy(self.init_state_dist)
        self.transition_matrix = self.transition_matrix / np.sum(self.transition_matrix, axis=-1, keepdims=True)

    def print_model(self):
        """Print the learned model."""
        for state in range(self.num_states):
            for action in range(self.num_actions):
                state_f = np.unravel_index(state, self.env_shape)
                print(f'Reward for state {state_f} and action {action}: {self.reward_matrix[state, action]}')
                next_state = np.argmax(self.transition_matrix[state, action])
                next_state_f = np.unravel_index(next_state, self.env_shape)
                prob = self.transition_matrix[state, action, next_state]
                print(f' - Transitions to {next_state_f} with probability {prob}')

    def print_rewards(self, reward_table):
        """Print the learned model."""
        for state in range(self.num_states):
            for action in range(self.num_actions):
                reward = reward_table[state, action]
                if reward > 0:
                    state_f = np.unravel_index(state, self.env_shape)
                    print(f'Reward for state {state_f} and action {action}: {reward}')

    def compute_occupancy(self, policy):
        """Compute the occupancy measure for the given policy.

        Args:
            policy (ndarray): A policy matrix.

        Returns:
            ndarray, ndarray: The occupancy measure and the policy kernel.
        """
        s0_measure = self.init_state_dist[:, np.newaxis] * policy
        pol_kern = self.compute_policy_kernel(policy)
        dom = (1. - self.gamma) * np.linalg.pinv(np.identity(self.num_states) - self.gamma * pol_kern.T) @ s0_measure
        return dom, pol_kern

    def compute_policy_kernel(self, policy):
        """Compute the transition kernel for the given policy.

        Args:
            policy (ndarray): A policy matrix.

        Returns:
            ndarray: The transition kernel.
        """
        return np.sum(policy[:, :, np.newaxis] * self.transition_matrix, axis=1)

    def compute_reward_table(self, utility_func, occupancy):
        """Compute the reward table for the given utility function and occupancy measure.

        Args:
            utility_func (function): A utility function.
            occupancy (ndarray): An occupancy measure.

        Returns:
            ndarray: The reward table.
        """
        occupancy = torch.tensor(occupancy, requires_grad=True)
        rewards = torch.tensor(self.reward_matrix, requires_grad=False)
        exp_vecs = torch.unsqueeze(occupancy, -1) * rewards
        exp_rew = exp_vecs.sum(dim=(0, 1))
        utility = utility_func(exp_rew)
        utility.backward()
        return occupancy.grad.numpy()

    def compute_q_table(self, reward_table, policy_kernel):
        """Compute the Q-table for the given reward table and policy kernel.

        Args:
            reward_table (ndarray): A reward table.
            policy_kernel (ndarray): A policy kernel.

        Returns:
            ndarray: The Q-table.
        """
        return np.linalg.pinv(np.identity(self.num_states) - self.gamma * policy_kernel) @ reward_table

    def policy_evaluation(self, reward_table, policy):
        """Estimate the Q-table for the given reward table and policy.

        Args:
            reward_table (ndarray): A reward table.
            policy (ndarray): A policy matrix.

        Returns:
            ndarray: The Q-table.
        """
        q_table = np.zeros((self.num_states, self.num_actions))
        state = 0

        for i in range(self.q_iters):
            action = self.select_action(state, policy)
            next_state = self.rng.choice(self.num_states, p=self.transition_matrix[state, action])
            reward = reward_table[state, action]
            q_table[state, action] += self.q_alpha * (
                    reward + self.gamma * np.dot(policy[next_state], q_table[next_state]) - q_table[state, action])
            if next_state in self.terminal_states:  # If the episode is done, reset the environment.
                state = 0
            else:
                state = next_state

        return q_table

    def evaluate_in_model(self, policy):
        """Evaluate the learned policy on the model.

        Args:
            policy (ndarray): A policy matrix.

        Returns:
            ndarray: The Pareto point.
        """
        pareto_point = np.zeros(self.num_objectives)

        for episode in range(self.eval_episodes):
            state = 0
            accrued_reward = np.zeros(self.num_objectives)
            timestep = 0

            while state not in self.terminal_states:
                action = self.select_action(state, policy)
                next_state = self.rng.choice(self.num_states, p=self.transition_matrix[state, action])
                reward = self.reward_matrix[state, action]
                accrued_reward = accrued_reward + (self.gamma ** timestep) * reward
                state = next_state
                timestep += 1

            pareto_point += accrued_reward

        return pareto_point / self.eval_episodes

    def evaluate(self, policy):
        """Evaluate the learned policy on the environment.

        Args:
            policy (ndarray): A policy matrix.

        Returns:
            ndarray: The Pareto point.
        """
        pareto_point = np.zeros(self.num_objectives)

        if self.box:
            format_state = lambda s: int(np.ravel_multi_index(s, self.env_shape))
        else:
            format_state = lambda s: s

        for episode in range(self.eval_episodes):
            state, _ = self.env.reset()
            state = format_state(state)
            terminated = False
            truncated = False
            accrued_reward = np.zeros(self.num_objectives)
            timestep = 0

            while not (terminated or truncated):
                action = self.select_action(state, policy)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                next_state = format_state(next_state)
                accrued_reward = accrued_reward + (self.gamma ** timestep) * reward
                state = next_state
                timestep += 1

            pareto_point += accrued_reward

        return pareto_point / self.eval_episodes

    def create_occupancy_policy(self, occupancy):
        """Create a policy that maximizes the occupancy measure.

        Args:
            occupancy (ndarray): An occupancy measure.

        Returns:
            ndarray: A policy matrix.
        """
        state_dist = np.sum(occupancy, axis=-1, keepdims=True)
        policy = occupancy / state_dist
        return policy

    def set_fixed_policy(self):
        """Set a fixed policy for testing purposes.

        Returns:
            ndarray: A policy matrix.
        """
        policy = np.zeros((self.num_states, self.num_actions))
        actions = [3] * 8 + [1] * 9
        states = [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8), (1, 8), (2, 8), (3, 8),
                  (4, 8), (5, 8), (6, 8), (7, 8), (8, 8)]
        for action, state in zip(actions, states):
            flat_state = np.ravel_multi_index(state, self.env_shape)
            policy[flat_state, :] = 0.
            policy[flat_state, action] = 1.
        return policy

    def solve(self, target, nadir):
        """Solve the convex MDP using the online mirror descent algorithm.

        Args:
            target (ndarray): The target point.
            nadir (ndarray): The nadir point.

        Returns:
            ndarray: The Pareto point.
        """
        u_func = create_fast_translated_rectangle_u(target, nadir, backend='torch')
        policy = np.copy(self.init_policy)
        composite_q = np.zeros((self.num_states, self.num_actions))

        for i in range(self.train_iters):
            occupancy, policy_kernel = self.compute_occupancy(policy)
            reward_table = self.compute_reward_table(u_func, occupancy)
            q_table = self.policy_evaluation(reward_table, policy)
            composite_q += self.q_alpha * q_table
            policy_q = composite_q - np.max(composite_q, axis=-1, keepdims=True)
            policy = np.exp(policy_q) / np.sum(np.exp(policy_q), axis=-1, keepdims=True)

        vec = self.evaluate(policy)
        utility = u_func(torch.tensor(vec))
        return vec, utility
