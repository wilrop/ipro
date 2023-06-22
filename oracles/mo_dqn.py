import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from oracles.probabilistic_ensemble import ProbabilisticEnsemble
from oracles.replay_buffer import PrioritizedAccruedRewardReplayBuffer, AccruedRewardReplayBuffer
from oracles.drl_oracle import DRLOracle


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


class MODQN(DRLOracle):
    def __init__(self,
                 env,
                 aug=0.2,
                 lr=0.001,
                 hidden_layers=(64, 64),
                 one_hot=False,
                 learning_start=1000,
                 train_freq=1,
                 target_update_freq=100,
                 gradient_steps=1,
                 epsilon_start=1.0,
                 epsilon_end=0.01,
                 exploration_frac=0.5,
                 gamma=0.99,
                 tau=1.0,
                 model_based=False,
                 model_lr=0.001,
                 model_hidden_layers=(64, 64),
                 model_steps=32,
                 pe_size=5,
                 buffer_size=100000,
                 per=False,
                 alpha_per=0.6,
                 min_priority=1.0,
                 batch_size=32,
                 init_real_frac=0.8,
                 final_real_frac=0.1,
                 model_train_finish=10000,
                 global_steps=100000,
                 eval_episodes=100,
                 log_freq=1000,
                 seed=0):
        super().__init__(env, aug=aug, gamma=gamma, one_hot=one_hot, eval_episodes=eval_episodes)

        self.dqn_lr = lr
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

        self.one_hot = one_hot

        self.input_dim = self.obs_dim + self.num_objectives
        self.output_dim = int(self.num_actions * self.num_objectives)
        self.dqn_hidden_layers = hidden_layers

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
            self.real_buffer = PrioritizedAccruedRewardReplayBuffer((self.obs_dim,),
                                                                    env.action_space.shape,
                                                                    rew_dim=self.num_objectives,
                                                                    max_size=self.buffer_size,
                                                                    action_dtype=np.uint8)
            self.model_buffer = PrioritizedAccruedRewardReplayBuffer((self.obs_dim,),
                                                                     env.action_space.shape,
                                                                     rew_dim=self.num_objectives,
                                                                     max_size=self.buffer_size,
                                                                     action_dtype=np.uint8)
        else:
            self.real_buffer = AccruedRewardReplayBuffer((self.obs_dim,),
                                                         env.action_space.shape,
                                                         rew_dim=self.num_objectives,
                                                         max_size=self.buffer_size,
                                                         action_dtype=np.uint8)
            self.model_buffer = AccruedRewardReplayBuffer((self.obs_dim,),
                                                          env.action_space.shape,
                                                          rew_dim=self.num_objectives,
                                                          max_size=self.buffer_size,
                                                          action_dtype=np.uint8)
        self.batch_size = batch_size
        self.init_real_frac = init_real_frac
        self.final_real_frac = final_real_frac
        self.model_train_finish = model_train_finish
        self.model_train_step = 0

        if model_based:
            self._init_env_model()

    def _init_env_model(self):
        """Initialize the environment model."""
        input_dim = self.obs_dim + self.num_actions
        output_dim = input_dim
        self.env_model = ProbabilisticEnsemble(input_dim,
                                               output_dim,
                                               ensemble_size=self.pe_size,
                                               arch=self.model_hidden_layers,
                                               learning_rate=self.model_lr)

    @staticmethod
    def init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def reset(self):
        """Reset the class for a new round of the inner loop."""
        self.q_network = QNetwork(self.input_dim, self.dqn_hidden_layers, self.output_dim)
        self.target_network = QNetwork(self.input_dim, self.dqn_hidden_layers, self.output_dim)
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=self.dqn_lr)
        self.q_network.apply(self.init_weights)
        self.target_network.apply(self.init_weights)
        self.real_buffer.reset_priorities()
        # self.real_buffer.reset()
        # self.model_buffer.reset()
        self.u_func = None

    def reset_env_model(self):
        """Reset the environment model."""
        self.model_train_step = 0
        self._init_env_model()

    def select_greedy_action(self, obs, accrued_reward, batched=False):
        """Select the greedy action.

        Args:
            obs (np.ndarray): The current observation of the environment.
            accrued_reward (np.ndarray): The accrued reward so far.

        Returns:
            action (int): The action to take.
        """
        augmented_obs = torch.as_tensor(np.concatenate((obs, accrued_reward)), dtype=torch.float)
        q_values = self.q_network(augmented_obs).view(-1, self.num_objectives)
        expected_returns = torch.tensor(accrued_reward) + self.gamma * q_values
        utilities = self.u_func(expected_returns)
        if batched:
            return torch.argmax(utilities.view(-1, self.num_actions, self.num_objectives), dim=-1)
        else:
            return torch.argmax(utilities).item()

    def select_action(self, obs, accrued_reward, epsilon):
        """Select an action using epsilon-greedy exploration.

        Args:
            obs (np.ndarray): The current observation of the environment.
            accrued_reward (np.ndarray): The accrued reward so far.
            epsilon (float): The probability of selecting a random action.

        Returns:
            action (int): The action to take.
        """
        if self.rng.uniform() < epsilon:
            return self.rng.integers(self.num_actions)
        else:
            return self.select_greedy_action(obs, accrued_reward)

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

    def collect_train_batch(self, as_tensor=False):
        """Collect a batch of data for training.

        Args:
            as_tensor (bool, optional): Whether to return the batch as tensors. Defaults to False.

        Returns:
            batch (dict): A dictionary containing the batch of data.
        """
        if self.model_based:
            real_batch_size, model_batch_size = self.get_batch_sizes()
            real_batch = self.real_buffer.sample(real_batch_size)
            model_batch = self.model_buffer.sample(model_batch_size)
            batch = [np.concatenate((real, model)) for real, model in zip(real_batch, model_batch)]
        else:
            batch = self.real_buffer.sample(self.batch_size)

        if as_tensor:
            batch = [torch.as_tensor(data, dtype=torch.float) for data in batch]
        return batch

    def compute_priority(self, u_target, u_pred):
        """Compute the priority of a transition or batch of transitions.

        Args:
            u_target (torch.Tensor): The target utility/utilities.
            u_pred (torch.Tensor): The predicted utility/utilities.

        Returns:
            priority (float): The priority/priorities of the transition.
        """
        td_errors = torch.abs(u_target - u_pred).detach().numpy()
        priority = max(td_errors ** self.alpha_per, self.min_priority)
        return priority

    def add_to_real_buffer(self, obs, accrued_reward, action, reward, next_obs, done, timestep):
        """Add a transition to the real replay buffer.

        Args:
            obs (np.ndarray): The current observation of the environment.
            accrued_reward (np.ndarray): The accrued reward so far.
            action (int): The action taken.
            reward (np.ndarray): The reward received.
            next_obs (np.ndarray): The next observation of the environment.
            done (bool): Whether the episode was completed.
            timestep (int): The timestep of the transition.
        """
        if self.per:
            t_obs = torch.as_tensor(obs, dtype=torch.float)
            t_accrued_reward = torch.as_tensor(accrued_reward, dtype=torch.float)
            t_reward = torch.as_tensor(reward, dtype=torch.float)
            t_next_obs = torch.as_tensor(next_obs, dtype=torch.float)

            # Compute the Q-value and utility of the previous obs-action pair.
            aug_obs = torch.cat((t_obs, t_accrued_reward))
            q_pred = self.q_network(aug_obs)[action]
            u_pred = self.u_func(t_accrued_reward + q_pred * self.gamma)

            # Compute the Q-value and utility of the current obs.
            next_accrued_reward = t_accrued_reward + t_reward * (self.gamma ** timestep)
            aug_next_obs = torch.cat((t_next_obs, next_accrued_reward))
            next_q_pred = self.q_network(aug_next_obs).view(-1, self.num_objectives)
            next_u_pred = self.u_func(next_accrued_reward + next_q_pred * self.gamma)

            # Select the argmax action of the current obs.
            next_action = torch.argmax(next_u_pred).item()

            # Compute the target Q-value of the target network in the current obs using the argmax action.
            q_target = self.target_network(aug_next_obs)[next_action]
            u_target = self.u_func(next_accrued_reward + q_target * self.gamma)

            # Compute the priority.
            priority = self.compute_priority(u_target, u_pred)
            self.real_buffer.add(obs, accrued_reward, action, reward, next_obs, done, timestep, priority=priority)
        else:
            self.real_buffer.add(obs, accrued_reward, action, reward, next_obs, done, timestep)

    def update_priorities(self, target_u, pred_u, indices):
        """Update the priorities of the transitions in the replay buffer.

        Args:
            target_u (torch.Tensor): The target utilities.
            pred_u (torch.Tensor): The predicted utilities.
            indices (np.ndarray): The indices of the transitions in the replay buffer.
        """
        td_errors = torch.abs(target_u - pred_u).detach().numpy()
        priorities = np.maximum(td_errors ** self.alpha_per, self.min_priority)
        if self.model_based:
            real_batch_size, _ = self.get_batch_sizes()

            real_indices = indices[:real_batch_size]
            real_priorities = priorities[:real_batch_size]
            self.real_buffer.update_priorities(real_indices, real_priorities)

            model_indices = indices[real_batch_size:]
            model_priorities = priorities[real_batch_size:]
            self.model_buffer.update_priorities(model_indices, model_priorities)
        else:
            self.real_buffer.update_priorities(indices, priorities)

    def train_network(self):
        """Train the Q-network using the replay buffer."""
        for _ in range(self.gradient_steps):
            batch = self.collect_train_batch(as_tensor=True)

            if self.per:
                obs, accrued_rewards, actions, rewards, next_obs, dones, timesteps, indices = batch
            else:
                obs, accrued_rewards, actions, rewards, next_obs, dones, timesteps = batch

            with torch.no_grad():
                next_accr_rews = accrued_rewards + rewards * (self.gamma ** timesteps)
                augmented_obs = torch.cat((next_obs, next_accr_rews), dim=1)
                target_pred = self.target_network(augmented_obs).view(-1, self.num_actions, self.num_objectives)
                total_rewards = next_accr_rews.unsqueeze(1) + self.gamma * target_pred
                utilities = self.u_func(total_rewards)
                best_actions = torch.argmax(utilities, dim=1)
                target_utilities = utilities[torch.arange(self.batch_size), best_actions]
                q_maxs = target_pred[torch.arange(self.batch_size), best_actions]
                td_target = rewards + self.gamma * q_maxs * (1 - dones)

            augmented_obs = torch.cat((obs, accrued_rewards), dim=1)
            preds = self.q_network(augmented_obs).view(-1, self.num_actions, self.num_objectives)
            action_preds = preds[torch.arange(self.batch_size), actions.type(torch.LongTensor)]
            loss = F.mse_loss(td_target, action_preds)

            # optimize the model
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if self.per:
                pred_utilities = self.u_func(rewards + self.gamma * action_preds).detach().numpy()
                self.update_priorities(target_utilities, pred_utilities, indices.type(torch.int))

    def update_model(self):
        """Update the environment model."""
        m_obs, m_actions, m_rewards, m_next_obs, m_dones, m_timesteps = self.real_buffer.get_all_data()
        one_hot = np.zeros((len(m_obs), self.num_actions))
        one_hot[np.arange(len(m_obs)), m_actions.astype(int).reshape(len(m_obs))] = 1
        X = np.hstack((m_obs, one_hot))
        Y = np.hstack((m_rewards, m_next_obs - m_obs))
        mean_holdout_loss = self.env_model.fit(X, Y)
        return mean_holdout_loss

    def generate_model_samples(self):
        """Update the environment model."""
        obs, accrued_rewards, timesteps = self.real_buffer.sample_obs_acc_rews(self.model_steps)
        actions = self.select_greedy_action(obs, accrued_rewards, batched=True)
        model_input = torch.cat((torch.Tensor(obs), actions), dim=1)
        pred_next_obs, pred_rewards = self.env_model.predict(model_input)
        pred_next_obs, pred_rewards = pred_next_obs.detach().numpy(), pred_rewards.detach().numpy()
        for i in range(self.model_steps):
            self.model_buffer.add(obs[i], accrued_rewards[i], actions[i], pred_rewards[i], pred_next_obs[i], False,
                                  timesteps[i])
        self.model_train_step += 1

    def train(self):
        """Train MODQN on the given environment."""
        raw_obs, _ = self.env.reset()
        obs = self.format_obs(raw_obs)
        timestep = 0
        accrued_reward = np.zeros(self.num_objectives)

        for global_step in range(self.global_steps):
            if global_step % self.log_freq == 0:
                print(f'Global step: {global_step}')

            epsilon = linear_schedule(self.epsilon_start, self.epsilon_end, self.exploration_steps, global_step)
            action = self.select_action(obs, accrued_reward, epsilon)
            next_raw_obs, reward, terminated, truncated, _ = self.env.step(action)
            next_obs = self.format_obs(next_raw_obs)
            self.add_to_real_buffer(obs, accrued_reward, action, reward, next_obs, terminated or truncated, timestep)
            accrued_reward += (self.gamma ** timestep) * reward
            obs = next_obs
            timestep += 1

            if terminated or truncated:  # If the episode is done, reset the environment and accrued reward.
                raw_obs, _ = self.env.reset()
                obs = self.format_obs(raw_obs)
                timestep = 0
                accrued_reward = np.zeros(self.num_objectives)

            if global_step > self.learning_start:
                if global_step % self.train_freq == 0:
                    if self.model_based:
                        self.update_model()
                        self.generate_model_samples()
                    self.train_network()
                if global_step % self.target_update_freq == 0:
                    for t_params, q_params in zip(self.target_network.parameters(), self.q_network.parameters()):
                        t_params.data.copy_(self.tau * q_params.data + (1.0 - self.tau) * t_params.data)

    def load_model(self, referent):
        """Load the model that is closest to the given referent.

        Args:
            referent (ndarray): The referent to load the model for.
        """
        closest_referent = self.get_closest_referent(referent)
        if closest_referent:
            self.target_network.load_state_dict(self.trained_models[tuple(closest_referent)])
            self.q_network.load_state_dict(self.trained_models[tuple(closest_referent)])

    def solve(self, referent, ideal, warm_start=True):
        """Solve for problem for the given referent and ideal.

        Args:
            referent (list): The referent to solve for.
            ideal (list): The ideal to solve for.
            warm_start (bool, optional): Whether to warm start the solver. Defaults to False.

        Returns:
            list: The solution to the problem.
        """
        self.reset()
        if warm_start:
            self.load_model(referent)
        pareto_point = super().solve(referent, ideal)
        self.trained_models[tuple(referent)] = self.q_network.state_dict()
        return pareto_point
