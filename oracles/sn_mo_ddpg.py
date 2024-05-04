import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.helpers import load_activation_fn
from oracles.replay_buffer import AccruedRewardReplayBuffer
from oracles.sn_drl_oracle import SNDRLOracle
from oracles.vector_u import aasf


class Critic(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, activation='relu'):
        super().__init__()
        activation_fn = load_activation_fn(activation)
        self.layers = [nn.Linear(input_dim, hidden_dims[0]), activation_fn()]

        for hidden_in, hidden_out in zip(hidden_dims[:-1], hidden_dims[1:]):
            self.layers.extend([nn.Linear(hidden_in, hidden_out), activation_fn()])

        self.layers.append(nn.Linear(hidden_dims[-1], output_dim))
        self.layers = nn.Sequential(*self.layers)

    def forward(self, obs, ref, action):
        concat_shape = obs.shape[:-1] + ref.shape[-1:]
        exp_ref = ref.expand(*concat_shape)
        x = torch.cat((obs, exp_ref, action), dim=-1)
        return self.layers(x)


class ContinuousActor(nn.Module):
    def __init__(self, input_dim, hidden_dims, env, activation='relu'):
        super().__init__()
        activation_fn = load_activation_fn(activation)
        self.layers = [nn.Linear(input_dim, hidden_dims[0]), activation_fn()]

        for hidden_in, hidden_out in zip(hidden_dims[:-1], hidden_dims[1:]):
            self.layers.extend([nn.Linear(hidden_in, hidden_out), activation_fn()])

        self.layers.append(nn.Linear(hidden_dims[-1], np.prod(env.action_space.shape)))
        self.layers = nn.Sequential(*self.layers)
        # action rescaling
        self.register_buffer(
            "action_scale", torch.tensor((env.action_space.high - env.action_space.low) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor((env.action_space.high + env.action_space.low) / 2.0, dtype=torch.float32)
        )

    def forward(self, obs, ref):
        concat_shape = obs.shape[:-1] + ref.shape[-1:]
        exp_ref = ref.expand(*concat_shape)
        x = torch.cat((obs, exp_ref), dim=-1)
        x = self.layers(x)
        x = torch.tanh(x)
        return x * self.action_scale + self.action_bias


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


class SNMODDPG(SNDRLOracle):
    def __init__(self,
                 env,
                 gamma=0.99,
                 aug=0.1,
                 scale=100,
                 lr_actor=0.001,
                 lr_critic=0.001,
                 actor_hidden=(64, 64),
                 critic_hidden=(64, 64),
                 actor_activation='tanh',
                 critic_activation='tanh',
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
                         alg_name='SN-MO-DDPG')
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.actor_hidden = actor_hidden
        self.critic_hidden = critic_hidden
        self.actor_activation = actor_activation
        self.critic_activation = critic_activation

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
        self.output_dim_actor = 1
        self.output_dim_critic = self.num_objectives

        self.actor = None
        self.critic = None
        self.actor_target = None
        self.critic_target = None
        self.actor_optimizer = None
        self.critic_optimizer = None

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
            'lr_actor': self.lr_actor,
            'lr_critic': self.lr_critic,
            'actor_hidden': self.actor_hidden,
            'critic_hidden': self.critic_hidden,
            'actor_activation': self.actor_activation,
            'critic_activation': self.critic_activation,
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
        self.actor = ContinuousActor(self.input_dim,
                                     self.actor_hidden,
                                     self.env,
                                     activation=self.actor_activation)
        self.actor_target = ContinuousActor(self.input_dim,
                                            self.actor_hidden,
                                            self.env,
                                            activation=self.actor_activation)
        self.critic = Critic(self.input_dim,
                             self.critic_hidden,
                             self.output_dim_critic,
                             activation=self.critic_activation)
        self.critic_target = Critic(self.input_dim,
                                    self.critic_hidden,
                                    self.output_dim_critic,
                                    activation=self.critic_activation)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr_actor)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.lr_critic)
        if self.clear_buffer:
            self.replay_buffer.reset()

    def save_model(self):
        """Save the models."""
        self.pretrained_model = {
            "actor_state_dict": self.actor.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
            "critic_optimizer_state_dict": self.critic_optimizer.state_dict()
        }

    def load_model(self, _):
        """Load the models."""
        self.actor.load_state_dict(self.pretrained_model["actor_state_dict"])
        self.actor_target.load_state_dict(self.pretrained_model["actor_state_dict"])
        self.critic.load_state_dict(self.pretrained_model["critic_state_dict"])
        self.critic_target.load_state_dict(self.pretrained_model["critic_state_dict"])
        self.actor_optimizer.load_state_dict(self.pretrained_model["actor_optimizer_state_dict"])
        self.critic_optimizer.load_state_dict(self.pretrained_model["critic_optimizer_state_dict"])

    def select_greedy_action(self, aug_obs, accrued_reward, referent, nadir, ideal):
        """Select the greedy action."""
        return self.actor(aug_obs, referent).item()

    def select_action(self, aug_obs, accrued_reward, referent, nadir, ideal, epsilon=0.1):
        """Select an action using epsilon-greedy exploration."""
        actions = self.actor(aug_obs, referent)
        actions += torch.normal(0, self.actor.action_scale * epsilon)
        actions = actions.cpu().numpy().clip(self.env.action_space.low, self.env.action_space.high)
        return actions

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

    def train_networks(self, referent, nadir, ideal, step, num_referents=1):
        """Train the actor and critic using the replay buffer."""
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
                    next_state_actions = self.actor_target(aug_next_obs, referent)
                    target_pred = self.critic_target(aug_next_obs, referent, next_state_actions)
                    td_target = rewards + self.gamma * (1 - dones) * target_pred.view(-1, self.num_objectives)

                preds = self.critic(aug_obs, referent, actions).view(-1, self.num_objectives)
                loss += F.mse_loss(td_target, preds)

            # optimize the model
            self.critic_optimizer.zero_grad()
            loss /= num_referents
            loss.backward()
            self.critic_optimizer.step()

            # Update the target networks
            if step % self.target_update_freq == 0:
                actor_loss = -self.critic(aug_obs, referent, self.actor(aug_obs, referent)).mean()
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                for t_params, a_params in zip(self.actor_target.parameters(), self.actor.parameters()):
                    t_params.data.copy_(self.tau * a_params.data + (1.0 - self.tau) * t_params.data)
                for t_params, q_params in zip(self.critic_target.parameters(), self.critic.parameters()):
                    t_params.data.copy_(self.tau * q_params.data + (1.0 - self.tau) * t_params.data)

        return loss.item()

    def pretrain(self):
        """Pretrain the algorithm."""
        self.reset()
        self.setup_ac_metrics()
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
              actor_train_freq=None,
              critic_train_freq=None,
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
                loss = self.train_networks(referent, nadir, ideal, step, num_referents=num_referents)
                if step % self.log_freq == 0:
                    self.log_pg(step, loss, 0, 0, 0)

    def solve(self, referent, nadir=None, ideal=None, *args, **kwargs):
        """Solve for problem for the given referent."""
        self.reset()
        self.setup_ac_metrics()
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
