import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from oracles.policy import Categorical
from oracles.drl_oracle import DRLOracle
from oracles.replay_buffer import RolloutBuffer


class Actor(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super().__init__()
        self.layers = [nn.Linear(input_dim, hidden_dims[0]), nn.Tanh()]

        for hidden_in, hidden_out in zip(hidden_dims[:-1], hidden_dims[1:]):
            self.layers.extend([nn.Linear(hidden_in, hidden_out), nn.Tanh()])

        self.layers.append(nn.Linear(hidden_dims[-1], output_dim))
        self.layers = nn.Sequential(*self.layers)

    def forward(self, x):
        """Forward pass through the network."""
        x = self.layers(x)
        x = F.log_softmax(x, dim=-1)
        return x


class Critic(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super().__init__()
        self.layers = [nn.Linear(input_dim, hidden_dims[0]), nn.Tanh()]

        for hidden_in, hidden_out in zip(hidden_dims[:-1], hidden_dims[1:]):
            self.layers.extend([nn.Linear(hidden_in, hidden_out), nn.Tanh()])

        self.layers.append(nn.Linear(hidden_dims[-1], output_dim))
        self.layers = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.layers(x)


class MOA2C(DRLOracle):
    def __init__(self,
                 env,
                 gamma=0.99,
                 aug=0.1,
                 scale=100,
                 lr_actor=0.001,
                 lr_critic=0.001,
                 actor_hidden=(64, 64),
                 critic_hidden=(64, 64),
                 e_coef=0.01,
                 v_coef=0.5,
                 max_grad_norm=0.5,
                 normalize_advantage=False,
                 n_steps=10,
                 gae_lambda=0.5,
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
                         seed=seed)
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.e_coef = e_coef
        self.v_coef = v_coef
        self.s0 = None

        self.global_steps = int(global_steps)
        self.log_freq = log_freq

        self.output_dim_actor = int(self.num_actions)
        self.output_dim_critic = int(self.num_objectives)
        self.actor_hidden = actor_hidden
        self.critic_hidden = critic_hidden

        self.actor = None
        self.critic = None
        self.policy = None
        self.actor_optimizer = None
        self.critic_optimizer = None
        self.max_grad_norm = max_grad_norm
        self.normalize_advantage = normalize_advantage
        self.gae_lambda = gae_lambda

        self.n_steps = n_steps
        self.rollout_buffer = RolloutBuffer((self.aug_obs_dim,),
                                            self.env.action_space.shape,
                                            rew_dim=(self.num_objectives,),
                                            max_size=self.n_steps,
                                            action_dtype=int,
                                            rng=self.np_rng)

    def config(self):
        """Get the config of the algorithm."""
        config = super().config()
        config.update({
            'lr_actor': self.lr_actor,
            'lr_critic': self.lr_critic,
            'e_coef': self.e_coef,
            'v_coef': self.v_coef,
            'global_steps': self.global_steps,
            'log_freq': self.log_freq,
            'actor_hidden': self.actor_hidden,
            'critic_hidden': self.critic_hidden,
            'max_grad_norm': self.max_grad_norm,
            'normalize_advantage': self.normalize_advantage,
            'n_steps': self.n_steps,
            'gae_lambda': self.gae_lambda,
        })
        return config

    def reset(self):
        """Reset the actor and critic networks, optimizers and policy."""
        self.actor = Actor(self.aug_obs_dim, self.actor_hidden, self.output_dim_actor)
        self.critic = Critic(self.aug_obs_dim, self.critic_hidden, self.output_dim_critic)
        self.actor.apply(self.init_weights)
        self.critic.apply(self.init_weights)
        self.policy = Categorical()
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr_actor)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.lr_critic)

    def calc_generalised_advantages(self, rewards, dones, values, v_next):
        """Compute the advantages for the rollouts.

        Args:
            rewards (Tensor): The rewards.
            dones (Tensor): The dones.
            values (Tensor): The values.
            v_next (Tensor): The value of the next state.

        Returns:
            Tensor: The advantages.
        """
        v_comb = torch.cat((values, v_next), dim=0)
        td_errors = rewards + self.gamma * (1 - dones) * v_comb[1:] - v_comb[:-1]
        advantages = torch.zeros_like(td_errors)
        advantages[-1] = td_errors[-1]
        for t in reversed(range(len(td_errors) - 1)):
            advantages[t] = td_errors[t] + self.gamma * self.gae_lambda * advantages[t + 1] * (1 - dones[t])
        return advantages

    def update_policy(self):
        """Update the policy using the rollout buffer."""
        aug_obs, actions, rewards, aug_next_obs, dones = self.rollout_buffer.get_all_data(to_tensor=True)
        with torch.no_grad():
            v_s0 = self.critic(self.s0)  # Value of s0.
        v_s0.requires_grad = True
        self.u_func(v_s0).backward()  # Gradient of utility function w.r.t. values.

        values = self.critic(aug_obs)  # Predict values of observations.
        with torch.no_grad():
            v_next = self.critic(aug_next_obs[-1:])  # Predict values of next observations.
            advantages = self.calc_generalised_advantages(rewards, dones, values, v_next)
            returns = advantages + values

        if self.normalize_advantage:
            advantages = (advantages - advantages.mean(dim=0)) / (advantages.std(dim=0) + 1e-8)

        actor_out = self.actor(aug_obs)  # Predict logits of actions.
        log_prob, entropy = self.policy.evaluate_actions(actor_out, actions)  # Evaluate actions.
        pg_loss = (advantages * log_prob).mean(dim=0)  # Policy gradient loss with advantage as baseline.
        policy_loss = -torch.dot(v_s0.grad, pg_loss)  # Gradient update rule for SER.
        entropy_loss = -torch.mean(entropy)  # Compute entropy bonus.
        value_loss = F.mse_loss(returns, values)

        loss = policy_loss + self.v_coef * value_loss + self.e_coef * entropy_loss
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        self.critic_optimizer.step()
        self.actor_optimizer.step()

        return loss.item(), policy_loss.item(), value_loss.item(), entropy_loss.item()

    def reset_env(self):
        """Reset the environment.

        Returns:
            Tuple: The initial observation, accrued reward, augmented observation and timestep.
        """
        obs, _ = self.env.reset()
        obs = np.nan_to_num(obs, posinf=0)
        accrued_reward = np.zeros(self.num_objectives)
        aug_obs = torch.tensor(np.concatenate((obs, accrued_reward)), dtype=torch.float)  # Create the augmented state.
        timestep = 0
        return aug_obs, accrued_reward, timestep

    def select_action(self, aug_obs, accrued_reward):
        """Select an action from the policy.

        Args:
            aug_obs (Tensor): The augmented observation.
            accrued_reward (ndarray): The accrued reward. This is not used in this algorithm.

        Returns:
            int: The action.
        """
        log_probs = self.actor(aug_obs)  # Logprobs for the actions.
        action = self.policy(log_probs, aug_obs=aug_obs).item()  # Sample an action from the distribution.
        return action

    def select_greedy_action(self, aug_obs, accrued_reward):
        """Select a greedy action. Used by the solve method in the super class.

        Args:
            aug_obs (Tensor): The augmented observation.
            accrued_reward (ndarray): The accrued reward. This is not used in this algorithm.

        Returns:
            int: The action.
        """
        log_probs = self.actor(aug_obs)  # Logprobs for the actions.
        action = self.policy.greedy(log_probs).item()  # Sample an action from the distribution.
        return action

    def train(self):
        """Train the agent."""
        aug_obs, accrued_reward, timestep = self.reset_env()
        self.s0 = aug_obs
        loss = 0
        pg_l = 0
        v_l = 0
        e_l = 0

        for global_step in range(self.global_steps):
            if global_step % self.log_freq == 0:
                print(f'Global step: {global_step}')

            with torch.no_grad():
                action = self.select_action(aug_obs, accrued_reward)

            next_obs, reward, terminated, truncated, info = self.env.step(action)
            next_obs = np.nan_to_num(next_obs, posinf=0)
            reward = np.nan_to_num(reward, posinf=0)
            accrued_reward += (self.gamma ** timestep) * reward  # Update the accrued reward.
            aug_next_obs = torch.tensor(np.concatenate((next_obs, accrued_reward)), dtype=torch.float)
            self.rollout_buffer.add(aug_obs, action, reward, aug_next_obs, terminated)

            if (global_step + 1) % self.n_steps == 0:
                loss, pg_l, v_l, e_l = self.update_policy()
                self.rollout_buffer.reset()

            if (global_step + 1) % self.log_freq == 0:
                self.log_pg(global_step, loss, pg_l, v_l, e_l)

            aug_obs = aug_next_obs
            timestep += 1

            if terminated or truncated:  # If the episode is done, reset the environment and accrued reward.
                self.save_episode_stats(accrued_reward, timestep)
                aug_obs, accrued_reward, timestep = self.reset_env()

    def solve(self, referent, nadir=None, ideal=None, warm_start=True):
        """Train the algorithm on the given environment."""
        self.reset()
        self.setup_ac_metrics()
        if warm_start:
            actor_net, critic_net = self.load_model(referent)
            if actor_net is not None:
                self.actor.load_state_dict(actor_net)
            if critic_net is not None:
                self.critic.load_state_dict(critic_net)
        pareto_point = super().solve(referent, nadir=nadir, ideal=ideal)
        self.save_models(referent, actor=self.actor, critic=self.critic)
        return pareto_point
