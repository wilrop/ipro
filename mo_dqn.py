import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from probabilistic_ensemble import ProbabilisticEnsemble
from replay_buffer import PrioritizedAccruedRewardReplayBuffer, AccruedRewardReplayBuffer
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


def linear_schedule(start_val: float, end_val: float, duration: int, t: int):
    """Linear schedule from start_e to end_e over duration steps starting at time t.

    Args:
        start_val (float): The starting value.
        end_val (float): The ending value.
        duration (int): The duration of the schedule.
        t (int): The current time step.

    Returns:
        float: The value of the schedule at time t.
    """
    slope = (end_val - start_val) / duration
    return max(slope * t + start_val, end_val)


class MODQN:
    def __init__(self,
                 env,
                 dqn_lr=0.001,
                 learning_start=1000,
                 train_freq=1,
                 target_update_freq=100,
                 gradient_steps=1,
                 epsilon_start=1.0,
                 epsilon_end=0.01,
                 exploration_frac=0.5,
                 gamma=0.99,
                 tau=1.0,
                 dqn_hidden_layers=(64, 64),
                 model_based=False,
                 model_lr=0.001,
                 model_hidden_layers=(64, 64),
                 model_steps=32,
                 pe_size=5,
                 buffer_size=100000,
                 per=False,
                 alpha_per: float = 0.6,
                 min_priority: float = 1.0,
                 batch_size=32,
                 init_real_frac=0.8,
                 final_real_frac=0.1,
                 model_train_finish=10000,
                 global_steps=100000,
                 eval_episodes=100,
                 log_freq=10,
                 seed=0):
        self.env = env
        self.num_actions = env.action_space.n
        self.num_objectives = env.reward_space.shape[0]

        self.dqn_lr = dqn_lr
        self.learning_start = learning_start
        self.train_freq = train_freq
        self.target_update_freq = target_update_freq
        self.gradient_steps = gradient_steps
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.exploration_frac = exploration_frac
        self.exploration_steps = int(exploration_frac * global_steps)
        self.gamma = gamma
        self.tau = tau

        self.global_steps = global_steps
        self.eval_episodes = eval_episodes
        self.log_freq = log_freq

        self.seed = seed
        self.rng = np.random.default_rng(seed=seed)

        self.input_dim = int(env.observation_space.shape[0] + self.num_objectives)
        self.output_dim = int(self.num_actions * self.num_objectives)
        self.dqn_hidden_layers = dqn_hidden_layers

        self.model_based = model_based
        self.model_lr = model_lr
        self.model_hidden_layers = model_hidden_layers
        self.pe_size = pe_size
        self.model_steps = model_steps

        self.q_network = None
        self.target_network = None
        self.optimizer = None
        self.env_model = None

        self.buffer_size = buffer_size
        self.per = per
        self.alpha_per = alpha_per
        self.min_priority = min_priority
        if self.per:
            self.real_buffer = PrioritizedAccruedRewardReplayBuffer(env.observation_space.shape,
                                                                    env.action_space.shape,
                                                                    rew_dim=self.num_objectives,
                                                                    max_size=self.buffer_size,
                                                                    action_dtype=np.uint8)
            self.model_buffer = PrioritizedAccruedRewardReplayBuffer(env.observation_space.shape,
                                                                     env.action_space.shape,
                                                                     rew_dim=self.num_objectives,
                                                                     max_size=self.buffer_size,
                                                                     action_dtype=np.uint8)
        else:
            self.real_buffer = AccruedRewardReplayBuffer(env.observation_space.shape,
                                                         env.action_space.shape,
                                                         rew_dim=self.num_objectives,
                                                         max_size=self.buffer_size,
                                                         action_dtype=np.uint8)
            self.model_buffer = AccruedRewardReplayBuffer(env.observation_space.shape,
                                                          env.action_space.shape,
                                                          rew_dim=self.num_objectives,
                                                          max_size=self.buffer_size,
                                                          action_dtype=np.uint8)
        self.batch_size = batch_size
        self.init_real_frac = init_real_frac
        self.final_real_frac = final_real_frac
        self.model_train_finish = model_train_finish
        self.model_train_step = 0
        self.u_func = None

        if model_based:
            self._init_env_model()

    def _init_env_model(self):
        """Initialize the environment model."""
        input_dim = self.env.observation_space.shape[0] + self.num_objectives + self.num_actions
        output_dim = input_dim
        self.env_model = ProbabilisticEnsemble(input_dim,
                                               output_dim,
                                               ensemble_size=self.pe_size,
                                               arch=self.model_hidden_layers,
                                               learning_rate=self.model_lr)

    def _get_config(self):
        """Get the configuration of the agent."""
        return {
            'env': self.env,
            'dqn_lr': self.dqn_lr,
            'learning_start': self.learning_start,
            'train_freq': self.train_freq,
            'target_update_freq': self.target_update_freq,
            'gradient_steps': self.gradient_steps,
            'epsilon_start': self.epsilon_start,
            'epsilon_end': self.epsilon_end,
            'exploration_frac': self.exploration_frac,
            'gamma': self.gamma,
            'tau': self.tau,
            'dqn_hidden_layers': self.dqn_hidden_layers,
            'model_based': self.model_based,
            'model_lr': self.model_lr,
            'model_hidden_layers': self.model_hidden_layers,
            'model_steps': self.model_steps,
            'pe_size': self.pe_size,
            'buffer_size': self.buffer_size,
            'per': self.per,
            'alpha_per': self.alpha_per,
            'min_priority': self.min_priority,
            'batch_size': self.batch_size,
            'init_real_frac': self.init_real_frac,
            'final_real_frac': self.final_real_frac,
            'model_train_finish': self.model_train_finish,
            'global_steps': self.global_steps,
            'eval_episodes': self.eval_episodes,
            'log_freq': self.log_freq,
            'seed': self.seed
        }

    def reset(self):
        """Reset the class for a new round of the inner loop."""
        self.q_network = QNetwork(self.input_dim, self.dqn_hidden_layers, self.output_dim)
        self.target_network = QNetwork(self.input_dim, self.dqn_hidden_layers, self.output_dim)
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=self.dqn_lr)
        self.real_buffer.reset()
        self.model_buffer.reset()
        self.u_func = None

    def reset_env_model(self):
        """Reset the environment model."""
        self.model_train_step = 0
        self._init_env_model()

    def select_greedy_action(self, state, accrued_reward, batched=False):
        """Select the greedy action.

        Args:
            state (np.ndarray): The current state of the environment.
            accrued_reward (np.ndarray): The accrued reward so far.

        Returns:
            action (int): The action to take.
        """
        augmented_state = torch.Tensor(np.concatenate((state, accrued_reward)))
        q_values = self.q_network(augmented_state).view(-1, self.num_actions, self.num_objectives)
        expected_returns = accrued_reward + self.gamma * q_values
        utilities = self.u_func(expected_returns)
        if batched:
            return torch.argmax(utilities, dim=-1)
        else:
            return torch.argmax(utilities[0]).item()

    def get_batch_sizes(self):
        """Get the batch sizes for the real and model buffers.

        Returns:
            int, int: The batch sizes for the real and model buffers.
        """
        real_frac = linear_schedule(self.init_real_frac, self.final_real_frac, self.model_train_finish,
                                    self.model_train_step)
        real_batch_size = int(self.batch_size * real_frac)
        model_batch_size = self.batch_size - real_batch_size
        return real_batch_size, model_batch_size

    def select_action(self, state, accrued_reward, epsilon):
        """Select an action using epsilon-greedy exploration.

        Args:
            state (np.ndarray): The current state of the environment.
            accrued_reward (np.ndarray): The accrued reward so far.
            epsilon (float): The probability of selecting a random action.

        Returns:
            action (int): The action to take.
        """
        if self.rng.uniform() < epsilon:
            return self.rng.integers(self.num_actions)
        else:
            return self.select_greedy_action(state, accrued_reward)

    def collect_train_batch(self):
        """Collect a batch of data for training.

        Returns:
            batch (dict): A dictionary containing the batch of data.
        """
        if self.model_based:
            real_batch_size, model_batch_size = self.get_batch_sizes()
            real_batch = self.real_buffer.sample(real_batch_size)
            model_batch = self.model_buffer.sample(model_batch_size)
            return [np.concatenate((real, model)) for real, model in zip(real_batch, model_batch)]
        else:
            batch = self.real_buffer.sample(self.batch_size)
            return batch + (None,)

    def compute_priority(self, state, accrued_reward, action, reward, next_state):
        next_accrued_reward = accrued_reward + reward
        next_q_pred = self.q_network(torch.Tensor(np.concatenate((next_state, next_accrued_reward))))
        u_target = self.u_func(next_accrued_reward + self.gamma * next_q_pred)

        q_pred = self.q_network(torch.Tensor(np.concatenate((state, accrued_reward))))
        u_pred = self.u_func(accrued_reward + self.gamma * q_pred)
        td_error = torch.abs(u_target - u_pred).detach().numpy()
        priority = max(td_error ** self.alpha_per, self.min_priority)
        return priority

    def add_to_real_buffer(self, state, accrued_reward, action, reward, next_state, truncated, terminated):
        if self.per:
            priority = self.compute_priority(state, accrued_reward, action, reward, next_state)
            self.real_buffer.add(state, accrued_reward, action, reward, next_state, truncated or terminated,
                                 priority=priority)
        else:
            self.real_buffer.add(state, accrued_reward, action, reward, next_state, truncated or terminated)

    def update_priorities(self, target_u, q_preds, accrued_rewards, indices):
        """Update the priorities of the transitions in the replay buffer.

        Args:
            target_u (torch.Tensor): The target utilities.
            q_preds (torch.Tensor): The predicted Q-values.
            accrued_rewards (torch.Tensor): The accrued rewards.
            indices (np.ndarray): The indices of the transitions in the replay buffer.
        """
        total_rewards = accrued_rewards.unsqueeze(1) + self.gamma * q_preds
        td_errors = torch.abs(target_u - self.u_func(total_rewards)).detach().numpy()
        priorities = np.maximum(td_errors ** self.alpha_per, self.min_priority)

        real_batch_size, _ = self.get_batch_sizes()
        real_indices = indices[:real_batch_size]
        real_priorities = priorities[:real_batch_size]
        self.real_buffer.update_priorities(real_indices, real_priorities)

        model_indices = indices[real_batch_size:]
        model_priorities = priorities[real_batch_size:]
        self.model_buffer.update_priorities(model_indices, model_priorities)

    def train_network(self):
        """Train the Q-network using the replay buffer."""
        for _ in range(self.gradient_steps):
            obs, accrued_rewards, actions, rewards, next_obs, dones, indices = self.collect_train_batch()

            with torch.no_grad():
                next_accr_rews = accrued_rewards + rewards  # Wrong
                augmented_states = torch.Tensor(np.concatenate((next_obs, next_accr_rews), axis=1))
                target_pred = self.target_network(augmented_states).view(-1, self.num_actions, self.num_objectives)
                total_rewards = next_accr_rews.unsqueeze(1) + self.gamma * target_pred
                utilities = self.u_func(total_rewards)
                best_actions = torch.argmax(utilities, dim=1)
                q_maxs = target_pred[torch.arange(self.batch_size), best_actions]
                td_target = rewards + self.gamma * q_maxs * (1 - dones)

            augmented_states = torch.Tensor(np.concatenate((obs, accrued_rewards), axis=1))
            old_vals = self.q_network(augmented_states).view(-1, self.num_actions, self.num_objectives)
            actions_idx = torch.LongTensor(actions)
            old_vals = old_vals[torch.arange(self.batch_size), actions_idx]
            loss = F.mse_loss(td_target, old_vals)

            # optimize the model
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if self.per:
                self.update_priorities(utilities, old_vals.detach().numpy(), accrued_rewards, indices)

    def update_model(self):
        """Update the environment model."""
        obs, accrued_rewards = self.real_buffer.sample_obs_acc_rews(self.model_steps)
        actions = self.select_greedy_action(obs, accrued_rewards, batched=True)
        model_input = torch.cat((torch.Tensor(obs), accrued_rewards, actions), dim=1)
        pred_next_obs, pred_rewards = self.env_model.predict(model_input)
        pred_next_obs, pred_rewards = pred_next_obs.detach().numpy(), pred_rewards.detach().numpy()
        for i in range(self.model_steps):
            self.model_buffer.add(obs[i], accrued_rewards[i], actions[i], pred_rewards[i], pred_next_obs[i], False)
        self.model_train_step += 1

    def train(self):
        """Train MODQN on the given environment."""
        state, _ = self.env.reset()
        timestep = 0
        accrued_reward = np.zeros(self.num_objectives)

        for global_step in range(self.global_steps):
            if global_step % self.log_freq == 0:
                print(f'Global step: {global_step}')

            epsilon = linear_schedule(self.epsilon_start, self.epsilon_end, self.exploration_steps, global_step)
            action = self.select_action(state, accrued_reward, epsilon)
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            self.add_to_real_buffer(state, accrued_reward, action, reward, next_state, terminated, truncated)
            accrued_reward += (self.gamma ** timestep) * reward
            state = next_state
            timestep += 1

            if self.model_based:  # Training the model can be done from the start as the environment is stationary.
                self.update_model()

            if terminated or truncated:  # If the episode is done, reset the environment and accrued reward.
                state, _ = self.env.reset()
                timestep = 0
                accrued_reward = np.zeros(self.num_objectives)

            if global_step > self.learning_start:
                if global_step % self.train_freq == 0:
                    self.train_network()
                if global_step % self.target_update_freq == 0:
                    for t_params, q_params in zip(self.target_network.parameters(), self.q_network.parameters()):
                        t_params.data.copy_(self.tau * q_params.data + (1.0 - self.tau) * t_params.data)

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

    def solve(self, target, local_nadir):
        """Run the inner loop of the outer loop."""
        self.reset()
        self.u_func = create_batched_fast_translated_rectangle_u(target, local_nadir, backend='torch')
        self.train()
        pareto_point = self.evaluate()
        return pareto_point
