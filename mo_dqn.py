import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from replay_buffer import AccruedRewardReplayBuffer
from vector_u import create_batched_fast_translated_rectangle_u


class QNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super().__init__()
        self.layers = [nn.Linear(input_dim, hidden_dims[0]), nn.ReLU()]

        for hidden_in, hidden_out in zip(hidden_dims[:-1], hidden_dims[1:]):
            self.layers.extend([nn.Linear(hidden_in, hidden_out), nn.ReLU()])

        self.layers.append(nn.Linear(hidden_dims[-1], output_dim))
        self.layers = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.layers(x)


class MODQN:
    def __init__(self,
                 env,
                 learning_rate,
                 learning_start,
                 train_freq,
                 target_update_interval,
                 epsilon,
                 epsilon_decay,
                 final_epsilon,
                 gamma,
                 tau,
                 hidden_layers,
                 buffer_size,
                 batch_size,
                 num_train_episodes,
                 num_eval_episodes,
                 log_every,
                 seed):
        self.env = env
        self.num_actions = env.action_space.n
        self.num_objectives = env.reward_space.shape[0]

        self.learning_rate = learning_rate
        self.learning_start = learning_start
        self.train_freq = train_freq
        self.target_update_interval = target_update_interval
        self.init_epsilon = epsilon
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon
        self.gamma = gamma
        self.tau = tau

        self.num_train_episodes = num_train_episodes
        self.num_eval_episodes = num_eval_episodes
        self.log_every = log_every

        self.seed = seed
        self.rng = np.random.default_rng(seed=seed)

        self.input_dim = int(env.observation_space.shape[0] + self.num_objectives)
        self.output_dim = int(self.num_actions * self.num_objectives)
        self.hidden_layers = hidden_layers

        self.q_network = None
        self.target_network = None
        self.optimizer = None

        self.buffer_size = buffer_size
        self.replay_buffer = AccruedRewardReplayBuffer(env.observation_space.shape,
                                                       env.action_space.shape,
                                                       rew_dim=self.num_objectives,
                                                       max_size=self.buffer_size,
                                                       action_dtype=np.uint8)
        self.batch_size = batch_size

        self.u_func = None

    def reset(self):
        """Reset the class for a new round of the inner loop."""
        self.q_network = QNetwork(self.input_dim, self.hidden_layers, self.output_dim)
        self.target_network = QNetwork(self.input_dim, self.hidden_layers, self.output_dim)
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        self.replay_buffer.cleanup()
        self.u_func = None
        self.epsilon = self.init_epsilon

    def select_greedy_action(self, state, accrued_reward):
        """Select the greedy action.

        Args:
            state (np.ndarray): The current state of the environment.
            accrued_reward (np.ndarray): The accrued reward so far.

        Returns:
            action (int): The action to take.
        """
        augmented_state = torch.Tensor(np.concatenate((state, accrued_reward)))
        q_values = self.q_network(augmented_state).detach().numpy().reshape(self.num_actions, self.num_objectives)
        expected_returns = accrued_reward + self.gamma * q_values
        utilities = self.u_func(expected_returns)
        return np.argmax(utilities)

    def select_action(self, state, accrued_reward):
        """Select an action using epsilon-greedy exploration.

        Args:
            state (np.ndarray): The current state of the environment.
            accrued_reward (np.ndarray): The accrued reward so far.

        Returns:
            action (int): The action to take.
        """
        if self.rng.uniform() < self.epsilon:
            return self.rng.integers(self.num_actions)
        else:
            return self.select_greedy_action(state, accrued_reward)

    def train_network(self):
        """Train the Q-network using the replay buffer."""
        obs, accrued_rewards, actions, rewards, next_obs, dones = self.replay_buffer.sample(self.batch_size)
        with torch.no_grad():
            next_accr_rews = accrued_rewards + rewards
            augmented_states = torch.Tensor(np.concatenate((next_obs, next_accr_rews), axis=1))
            target_pred = self.target_network(augmented_states).detach().numpy().reshape(-1, self.num_actions,
                                                                                         self.num_objectives)
            total_rewards = np.expand_dims(next_accr_rews, axis=1) + self.gamma * target_pred
            utilities = self.u_func(total_rewards)
            best_actions = np.argmax(utilities, axis=1)
            q_maxs = target_pred[np.arange(self.batch_size), best_actions]
            td_target = torch.Tensor(rewards + self.gamma * q_maxs * (1 - dones))

        augmented_states = torch.Tensor(np.concatenate((obs, accrued_rewards), axis=1))
        old_vals = self.q_network(augmented_states).view(-1, self.num_actions, self.num_objectives)
        actions_idx = torch.LongTensor(actions)
        old_vals = old_vals[torch.arange(self.batch_size), actions_idx]
        loss = F.mse_loss(td_target, old_vals)

        # optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train(self):
        """Train MODQN on the given environment."""
        global_step = 0

        for episode in range(self.num_train_episodes):
            if episode % self.log_every == 0:
                print(f"Training episode {episode + 1}")

            state, _ = self.env.reset()
            terminated = False
            truncated = False
            accrued_reward = np.zeros(self.num_objectives)
            timestep = 0

            while not (terminated or truncated):
                action = self.select_action(state, accrued_reward)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                self.replay_buffer.add(state, accrued_reward, action, reward, next_state, truncated or terminated)
                accrued_reward += (self.gamma ** timestep) * reward
                state = next_state

                if global_step > self.learning_start:
                    if global_step % self.train_freq == 0:
                        self.train_network()
                    if global_step % self.target_update_interval == 0:
                        for target_network_param, q_network_param in zip(self.target_network.parameters(),
                                                                         self.q_network.parameters()):
                            target_network_param.data.copy_(
                                self.tau * q_network_param.data + (1.0 - self.tau) * target_network_param.data)

                timestep += 1
                global_step += 1

            if global_step > self.learning_start:
                self.epsilon = max(self.epsilon * self.epsilon_decay, self.final_epsilon)

    def evaluate(self):
        """Evaluate MODQN on the given environment."""
        pareto_point = np.zeros(self.num_objectives)

        for episode in range(self.num_eval_episodes):
            if episode % self.log_every == 0:
                print(f"Evaluation episode {episode + 1}")

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

        return pareto_point / self.num_eval_episodes

    def solve(self, target, local_nadir):
        """Run the inner loop of the outer loop."""
        self.reset()
        self.u_func = create_batched_fast_translated_rectangle_u(target, local_nadir, backend='numpy')
        self.train()
        pareto_point = self.evaluate()
        return pareto_point
