import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.helpers import load_activation_fn
from oracles.replay_buffer import AccruedRewardReplayBuffer
from oracles.sn_drl_oracle import SNDRLOracle
from oracles.vector_u import aasf


class QNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, activation='relu'):
        super().__init__()
        activation_fn = load_activation_fn(activation)
        self.layers = [nn.Linear(input_dim, hidden_dims[0]), activation_fn()]

        for hidden_in, hidden_out in zip(hidden_dims[:-1], hidden_dims[1:]):
            self.layers.extend([nn.Linear(hidden_in, hidden_out), activation_fn()])

        self.layers.append(nn.Linear(hidden_dims[-1], output_dim))
        self.layers = nn.Sequential(*self.layers)

    def forward(self, obs, ref):
        concat_shape = obs.shape[:-1] + ref.shape[-1:]
        exp_ref = ref.expand(*concat_shape)
        x = torch.cat((obs, exp_ref), dim=-1)
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


class SNMODQN(SNDRLOracle):
    def __init__(self,
                 env,
                 gamma=0.99,
                 aug=0.1,
                 scale=100,
                 lr=0.001,
                 hidden_layers=(64, 64),
                 activation='relu',
                 pre_train_freq=1,
                 online_train_freq=1,
                 target_update_freq=1,
                 gradient_steps=1,
                 pretrain_iters=100,
                 grid_sample=False,
                 num_referents=16,
                 pre_learning_start=1000,
                 pre_epsilon_start=1.0,
                 pre_epsilon_end=0.01,
                 pre_exploration_frac=0.5,
                 pretraining_steps=100000,
                 online_learning_start=1000,
                 online_epsilon_start=0.1,
                 online_epsilon_end=0.01,
                 online_exploration_frac=0.5,
                 online_steps=100000,
                 tau=0.1,
                 buffer_size=100000,
                 batch_size=32,
                 clear_buffer=False,
                 eval_episodes=100,
                 log_freq=1000,
                 track=False,
                 seed=0):
        super().__init__(env,
                         gamma=gamma,
                         aug=aug,
                         scale=scale,
                         pretrain_iters=pretrain_iters,
                         num_referents=num_referents,
                         pretraining_steps=pretraining_steps,
                         grid_sample=grid_sample,
                         online_steps=online_steps,
                         eval_episodes=eval_episodes,
                         track=track,
                         seed=seed,
                         alg_name='SN-MO-DQN')
        self.dqn_lr = lr
        self.online_learning_start = online_learning_start  # The number of steps before learning starts online.
        self.online_epsilon_start = online_epsilon_start  # The starting epsilon for online learning.
        self.online_epsilon_end = online_epsilon_end  # The ending epsilon for online learning.
        self.online_exploration_frac = online_exploration_frac  # The fraction of online learning steps to explore.
        self.online_steps = online_steps  # The number of steps to learn online.

        self.pre_train_freq = pre_train_freq
        self.online_train_freq = online_train_freq
        self.target_update_freq = target_update_freq
        self.gradient_steps = gradient_steps
        self.tau = tau
        self.pre_learning_start = pre_learning_start  # The number of steps to wait before training.
        self.pre_epsilon_start = pre_epsilon_start  # The initial epsilon value.
        self.pre_epsilon_end = pre_epsilon_end  # The final epsilon value.
        self.pre_exploration_frac = pre_exploration_frac  # The fraction of the training steps to explore.

        self.log_freq = log_freq

        self.input_dim = self.aug_obs_dim + self.num_objectives  # Obs + accrued reward + referent.
        self.output_dim = int(self.num_actions * self.num_objectives)
        self.dqn_hidden_layers = hidden_layers
        self.activation = activation

        self.q_network = None
        self.target_network = None
        self.optimizer = None

        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.clear_buffer = clear_buffer
        self.replay_buffer = AccruedRewardReplayBuffer(obs_shape=(self.aug_obs_dim,),
                                                       action_shape=self.env.action_space.shape,
                                                       rew_dim=self.num_objectives,
                                                       max_size=self.buffer_size,
                                                       action_dtype=np.uint8,
                                                       rng=self.np_rng)

    def config(self):
        """Get the config of the algorithm."""
        config = super().config()
        config.update({
            'lr': self.dqn_lr,
            'hidden_layers': self.dqn_hidden_layers,
            'activation': self.activation,
            'pre_train_freq': self.pre_train_freq,
            'online_train_freq': self.online_train_freq,
            'target_update_freq': self.target_update_freq,
            'gradient_steps': self.gradient_steps,
            'pre_learning_start': self.pre_learning_start,
            'pre_epsilon_start': self.pre_epsilon_start,
            'pre_epsilon_end': self.pre_epsilon_end,
            'pre_exploration_frac': self.pre_exploration_frac,
            'online_learning_start': self.online_learning_start,
            'online_epsilon_start': self.online_epsilon_start,
            'online_epsilon_end': self.online_epsilon_end,
            'online_exploration_frac': self.online_exploration_frac,
            'tau': self.tau,
            'buffer_size': self.buffer_size,
            'batch_size': self.batch_size,
            'clear_buffer': self.clear_buffer,
            'log_freq': self.log_freq
        })
        return config

    def reset(self):
        """Reset the class for a new round of the inner loop."""
        self.q_network = QNetwork(self.input_dim, self.dqn_hidden_layers, self.output_dim, activation=self.activation)
        self.target_network = QNetwork(self.input_dim,
                                       self.dqn_hidden_layers,
                                       self.output_dim,
                                       activation=self.activation)
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=self.dqn_lr)
        self.q_network.apply(self.init_weights)
        self.target_network.apply(self.init_weights)
        if self.clear_buffer:
            self.replay_buffer.reset()

    def save_model(self):
        """Save the models."""
        self.pretrained_model = {
            "model_state_dict": self.target_network.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict()
        }

    def load_model(self, _):
        """Load the models."""
        self.q_network.load_state_dict(self.pretrained_model["model_state_dict"])
        self.target_network.load_state_dict(self.pretrained_model["model_state_dict"])
        self.optimizer.load_state_dict(self.pretrained_model["optimizer_state_dict"])

    def select_greedy_action(self, aug_obs, accrued_reward, referent, nadir, ideal):
        """Select the greedy action.

        Args:
            aug_obs (np.ndarray): The current augmented observation.
            accrued_reward (np.ndarray): The accrued reward so far.

        Returns:
            action (int): The action to take.
        """
        q_values = self.q_network(aug_obs, referent).view(-1, self.num_objectives)
        expected_returns = torch.tensor(accrued_reward) + self.gamma * q_values
        utilities = aasf(expected_returns, referent, nadir, ideal, aug=self.aug, scale=self.scale, backend='torch')
        return torch.argmax(utilities).item()

    def select_action(self, aug_obs, accrued_reward, referent, nadir, ideal, epsilon=0.1):
        """Select an action using epsilon-greedy exploration."""
        if self.np_rng.uniform() < epsilon:
            return self.np_rng.integers(self.num_actions)
        else:
            return self.select_greedy_action(aug_obs, accrued_reward, referent, nadir, ideal)

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
        self.replay_buffer.add(aug_obs, accrued_reward, action, reward, aug_next_obs, done, timestep)

    def train_network(self, referent, nadir, ideal, num_referents=1):
        """Train the Q-network using the replay buffer."""
        for _ in range(self.gradient_steps):
            batch = self.replay_buffer.sample(self.batch_size, to_tensor=True)
            aug_obs, accrued_rewards, actions, rewards, aug_next_obs, dones, timesteps = batch
            referents = torch.unsqueeze(referent, dim=0)
            if num_referents > 1:
                additional_referents = self.uniform_sample_referents(num_referents - 1, nadir, ideal)
                referents = torch.cat((referents, additional_referents), dim=0)
            loss = torch.tensor(0, dtype=torch.float)
            for referent in referents:
                referent = referent.expand(self.batch_size, self.num_objectives)
                with torch.no_grad():
                    next_accr_rews = accrued_rewards + rewards * (self.gamma ** timesteps)
                    target_pred = self.target_network(aug_next_obs, referent).view(-1,
                                                                                   self.num_actions,
                                                                                   self.num_objectives)
                    total_rewards = next_accr_rews.unsqueeze(1) + self.gamma * target_pred * (1 - dones.unsqueeze(-1))
                    utilities = aasf(total_rewards,
                                     referent.unsqueeze(1),
                                     nadir,
                                     ideal,
                                     aug=self.aug,
                                     scale=self.scale,
                                     backend='torch')
                    best_actions = torch.argmax(utilities, dim=1)
                    q_maxs = target_pred[torch.arange(self.batch_size), best_actions]
                    td_target = rewards + self.gamma * q_maxs * (1 - dones)

                preds = self.q_network(aug_obs, referent).view(-1, self.num_actions, self.num_objectives)
                action_preds = preds[torch.arange(self.batch_size), actions.type(torch.LongTensor)]
                loss += F.mse_loss(td_target, action_preds)

            # optimize the model
            self.optimizer.zero_grad()
            loss /= num_referents
            loss.backward()
            self.optimizer.step()
            return loss.item()

    def pretrain(self):
        """Pretrain the algorithm."""
        self.reset()
        self.setup_dqn_metrics()
        referents = self.sample_referents(self.pretrain_iters, self.nadir, self.ideal)
        for idx, referent in enumerate(referents):
            print(f"Pretraining on referent {idx + 1} of {self.pretrain_iters}")
            self.train(referent,
                       self.nadir,
                       self.ideal,
                       steps=self.pretraining_steps,
                       train_freq=self.pre_train_freq,
                       learning_start=self.pre_learning_start if idx == 0 else 0,  # Only fill buffer first iteration.
                       epsilon_start=self.pre_epsilon_start,
                       epsilon_end=self.pre_epsilon_end,
                       exploration_frac=self.pre_exploration_frac,
                       num_referents=self.num_referents)

        self.save_model()

    def train(self,
              referent,
              nadir,
              ideal,
              steps=None,
              train_freq=None,
              learning_start=None,
              epsilon_start=None,
              epsilon_end=None,
              exploration_frac=None,
              num_referents=None,
              *args,
              **kwargs):
        """Train MODQN on the given environment."""
        obs, _ = self.env.reset()
        obs = np.nan_to_num(obs, posinf=0)
        timestep = 0
        accrued_reward = np.zeros(self.num_objectives)
        aug_obs = np.hstack((obs, accrued_reward))
        loss = 0

        for step in range(steps):
            if step % self.log_freq == 0:
                print(f'{self.phase} step: {step}')

            epsilon = linear_schedule(epsilon_start, epsilon_end, int(exploration_frac * steps), step)
            with torch.no_grad():
                action = self.select_action(torch.tensor(aug_obs, dtype=torch.float),
                                            accrued_reward,
                                            referent,
                                            nadir,
                                            ideal,
                                            epsilon=epsilon)

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
                self.save_episode_stats(accrued_reward, timestep, referent, nadir, ideal)
                obs, _ = self.env.reset()
                obs = np.nan_to_num(obs, posinf=0)
                accrued_reward = np.zeros(self.num_objectives)
                aug_obs = np.hstack((obs, accrued_reward))
                timestep = 0

            if step > learning_start:
                if step % train_freq == 0:
                    loss = self.train_network(referent, nadir, ideal, num_referents=num_referents)
                if step % self.log_freq == 0:
                    self.log_dqn(step, loss)
                if step % self.target_update_freq == 0:
                    for t_params, q_params in zip(self.target_network.parameters(), self.q_network.parameters()):
                        t_params.data.copy_(self.tau * q_params.data + (1.0 - self.tau) * t_params.data)

    def solve(self, referent, nadir=None, ideal=None, *args, **kwargs):
        """Solve for problem for the given referent."""
        self.reset()
        self.setup_dqn_metrics()
        pareto_point = super().solve(referent,
                                     nadir=nadir,
                                     ideal=ideal,
                                     steps=self.online_steps,
                                     train_freq=self.online_train_freq,
                                     learning_start=self.online_learning_start,
                                     epsilon_start=self.online_epsilon_start,
                                     epsilon_end=self.online_epsilon_end,
                                     exploration_frac=self.online_exploration_frac)
        return pareto_point
