import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ipro.oracles.replay_buffer import PrioritizedAccruedRewardReplayBuffer, AccruedRewardReplayBuffer
from ipro.oracles.drl_oracle import DRLOracle


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
                 gamma=0.99,
                 aug=0.1,
                 scale=100,
                 lr=0.001,
                 hidden_layers=(64, 64),
                 learning_start=1000,
                 train_freq=1,
                 target_update_freq=1,
                 gradient_steps=1,
                 epsilon_start=1.0,
                 epsilon_end=0.01,
                 exploration_frac=0.5,
                 tau=0.1,
                 buffer_size=100000,
                 per=False,
                 alpha_per=0.6,
                 min_priority=1e-3,
                 batch_size=32,
                 global_steps=100000,
                 warm_start=False,
                 eval_episodes=100,
                 log_freq=1000,
                 track=False,
                 seed=0):
        super().__init__(env,
                         gamma=gamma,
                         aug=aug,
                         scale=scale,
                         warm_start=warm_start,
                         eval_episodes=eval_episodes,
                         track=track,
                         seed=seed,
                         alg_name='MO-DQN')
        self.dqn_lr = lr
        self.learning_start = learning_start
        self.train_freq = train_freq
        self.target_update_freq = target_update_freq
        self.gradient_steps = gradient_steps
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.exploration_frac = exploration_frac
        self.exploration_steps = int(exploration_frac * global_steps)
        self.tau = tau

        self.global_steps = int(global_steps)
        self.log_freq = log_freq

        self.output_dim = int(self.num_actions * self.num_objectives)
        self.dqn_hidden_layers = hidden_layers

        self.q_network = None
        self.target_network = None
        self.optimizer = None

        self.buffer_size = buffer_size
        self.per = per
        self.alpha_per = alpha_per
        self.min_priority = min_priority
        if self.per:
            self.replay_buffer = PrioritizedAccruedRewardReplayBuffer((self.aug_obs_dim,),
                                                                      action_shape=self.env.action_space.shape,
                                                                      rew_dim=self.num_objectives,
                                                                      max_size=self.buffer_size,
                                                                      action_dtype=np.uint8,
                                                                      rng=self.np_rng)
        else:
            self.replay_buffer = AccruedRewardReplayBuffer((self.aug_obs_dim,),
                                                           action_shape=self.env.action_space.shape,
                                                           rew_dim=self.num_objectives,
                                                           max_size=self.buffer_size,
                                                           action_dtype=np.uint8,
                                                           rng=self.np_rng)

        self.batch_size = batch_size

    def config(self):
        """Get the config of the algorithm."""
        config = super().config()
        config.update({
            'dqn_lr': self.dqn_lr,
            'learning_start': self.learning_start,
            'train_freq': self.train_freq,
            'target_update_freq': self.target_update_freq,
            'gradient_steps': self.gradient_steps,
            'epsilon_start': self.epsilon_start,
            'epsilon_end': self.epsilon_end,
            'exploration_frac': self.exploration_frac,
            'tau': self.tau,
            'buffer_size': self.buffer_size,
            'per': self.per,
            'alpha_per': self.alpha_per,
            'min_priority': self.min_priority,
            'batch_size': self.batch_size,
            'global_steps': self.global_steps,
            'log_freq': self.log_freq,
        })
        return config

    def reset(self):
        """Reset the class for a new round of the inner loop."""
        self.q_network = QNetwork(self.aug_obs_dim, self.dqn_hidden_layers, self.output_dim)
        self.target_network = QNetwork(self.aug_obs_dim, self.dqn_hidden_layers, self.output_dim)
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=self.dqn_lr)
        self.q_network.apply(self.init_weights)
        self.target_network.apply(self.init_weights)
        self.replay_buffer.reset()
        self.u_func = None

    def select_greedy_action(self, aug_obs, accrued_reward, *args, **kwargs):
        """Select the greedy action.

        Args:
            aug_obs (np.ndarray): The current augmented observation.
            accrued_reward (np.ndarray): The accrued reward so far.

        Returns:
            action (int): The action to take.
        """
        q_values = self.q_network(aug_obs).view(-1, self.num_objectives)
        expected_returns = torch.tensor(accrued_reward) + self.gamma * q_values
        utilities = self.u_func(expected_returns)
        return torch.argmax(utilities).item()

    def select_action(self, aug_obs, accrued_reward, epsilon=0.1, *args, **kwargs):
        """Select an action using epsilon-greedy exploration.

        Args:
            aug_obs (np.ndarray): The current augmented observation of the environment.
            accrued_reward (np.ndarray): The accrued reward so far.
            epsilon (float): The probability of selecting a random action.

        Returns:
            action (int): The action to take.
        """
        if self.np_rng.uniform() < epsilon:
            return self.np_rng.integers(self.num_actions)
        else:
            return self.select_greedy_action(aug_obs, accrued_reward)

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

    def add_to_buffer(self, aug_obs, accrued_reward, action, reward, aug_next_obs, done, timestep):
        """Add a transition to the replay buffer.

        Args:
            aug_obs (np.ndarray): The current observation of the environment.
            accrued_reward (np.ndarray): The accrued reward so far.
            action (int): The action taken.
            reward (np.ndarray): The reward received.
            aug_next_obs (np.ndarray): The next observation of the environment.
            done (bool): Whether the episode was completed.
            timestep (int): The timestep of the transition.
        """
        if self.per:
            with torch.no_grad():
                t_aug_obs = torch.as_tensor(aug_obs, dtype=torch.float)
                t_accrued_reward = torch.as_tensor(accrued_reward, dtype=torch.float)
                t_reward = torch.as_tensor(reward, dtype=torch.float)
                t_aug_next_obs = torch.as_tensor(aug_next_obs, dtype=torch.float)

                # Compute the Q-value and utility of the previous obs-action pair.
                q_pred = self.q_network(t_aug_obs)[action]
                u_pred = self.u_func(t_accrued_reward + q_pred * self.gamma)

                # Compute the Q-value and utility of the current obs.
                next_accrued_reward = t_accrued_reward + t_reward * (self.gamma ** timestep)
                next_q_pred = self.target_network(t_aug_next_obs).view(-1, self.num_objectives)
                next_u_pred = self.u_func(next_accrued_reward + next_q_pred * self.gamma)

                # Select the argmax action of the current obs and take its utility.
                next_action = torch.argmax(next_u_pred).item()
                u_target = next_u_pred[next_action]

                # Compute the priority.
                priority = self.compute_priority(u_target, u_pred)
                self.replay_buffer.add(aug_obs, accrued_reward, action, reward, aug_next_obs, done, timestep,
                                       priority=priority)
        else:
            self.replay_buffer.add(aug_obs, accrued_reward, action, reward, aug_next_obs, done, timestep)

    def update_priorities(self, target_u, pred_u, indices):
        """Update the priorities of the transitions in the replay buffer.

        Args:
            target_u (torch.Tensor): The target utilities.
            pred_u (torch.Tensor): The predicted utilities.
            indices (np.ndarray): The indices of the transitions in the replay buffer.
        """
        td_errors = torch.abs(target_u - pred_u).detach().numpy()
        priorities = np.maximum(td_errors ** self.alpha_per, self.min_priority)
        self.replay_buffer.update_priorities(indices, priorities)

    def train_network(self):
        """Train the Q-network using the replay buffer."""
        for _ in range(self.gradient_steps):
            batch = self.replay_buffer.sample(self.batch_size, to_tensor=True)

            if self.per:
                aug_obs, accrued_rewards, actions, rewards, aug_next_obs, dones, timesteps, indices = batch
            else:
                aug_obs, accrued_rewards, actions, rewards, aug_next_obs, dones, timesteps = batch

            with torch.no_grad():
                next_accr_rews = accrued_rewards + rewards * (self.gamma ** timesteps)
                target_pred = self.target_network(aug_next_obs).view(-1, self.num_actions, self.num_objectives)
                total_rewards = next_accr_rews.unsqueeze(1) + self.gamma * target_pred
                utilities = self.u_func(total_rewards)
                best_actions = torch.argmax(utilities, dim=1)
                target_utilities = utilities[torch.arange(self.batch_size), best_actions]
                q_maxs = target_pred[torch.arange(self.batch_size), best_actions]
                td_target = rewards + self.gamma * q_maxs * (1 - dones)

            preds = self.q_network(aug_obs).view(-1, self.num_actions, self.num_objectives)
            action_preds = preds[torch.arange(self.batch_size), actions.type(torch.LongTensor)]
            loss = F.mse_loss(td_target, action_preds)

            # optimize the model
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if self.per:
                pred_utilities = self.u_func(accrued_rewards + self.gamma * action_preds).detach().numpy()
                self.update_priorities(target_utilities, pred_utilities, indices.type(torch.int))
            return loss.item()

    def train(self):
        """Train MODQN on the given environment."""
        obs, _ = self.env.reset()
        obs = np.nan_to_num(obs, posinf=0)
        timestep = 0
        accrued_reward = np.zeros(self.num_objectives)
        aug_obs = np.hstack((obs, accrued_reward))
        loss = 0

        for global_step in range(self.global_steps):
            if global_step % self.log_freq == 0:
                print(f'Global step: {global_step}')

            epsilon = linear_schedule(self.epsilon_start, self.epsilon_end, self.exploration_steps, global_step)
            with torch.no_grad():
                action = self.select_action(torch.tensor(aug_obs, dtype=torch.float), accrued_reward, epsilon=epsilon)

            next_obs, reward, terminated, truncated, info = self.env.step(action)
            next_obs = np.nan_to_num(next_obs, posinf=0)
            reward = np.nan_to_num(reward, posinf=0)
            next_accrued_reward = accrued_reward + (self.gamma ** timestep) * reward
            aug_next_obs = np.hstack((next_obs, next_accrued_reward))
            self.add_to_buffer(aug_obs, accrued_reward, action, reward, aug_next_obs, terminated, timestep)
            accrued_reward = next_accrued_reward
            aug_obs = aug_next_obs
            timestep += 1

            if terminated or truncated:  # If the episode is done, reset the environment and accrued reward.
                self.save_episode_stats(accrued_reward, timestep)
                obs, _ = self.env.reset()
                obs = np.nan_to_num(obs, posinf=0)
                accrued_reward = np.zeros(self.num_objectives)
                aug_obs = np.hstack((obs, accrued_reward))
                timestep = 0

            if global_step > self.learning_start:
                if global_step % self.train_freq == 0:
                    loss = self.train_network()
                if global_step % self.log_freq == 0:
                    self.log_dqn(global_step, loss)
                if global_step % self.target_update_freq == 0:
                    for t_params, q_params in zip(self.target_network.parameters(), self.q_network.parameters()):
                        t_params.data.copy_(self.tau * q_params.data + (1.0 - self.tau) * t_params.data)

    def solve(self, referent, nadir=None, ideal=None, warm_start=True):
        """Solve for problem for the given referent and ideal.

        Args:
            referent (list): The referent to solve for.
            ideal (list): The ideal to solve for.

        Returns:
            list: The solution to the problem.
        """
        self.reset()
        self.setup_dqn_metrics()
        if self.warm_start:
            _, critic_net = self.load_model(referent)
            if critic_net is not None:
                self.target_network.load_state_dict(critic_net)
                self.q_network.load_state_dict(critic_net)

        pareto_point = super().run_inner_loop(referent, nadir=nadir, ideal=ideal)
        self.save_models(referent, critic=self.q_network)
        return pareto_point, self.q_network
