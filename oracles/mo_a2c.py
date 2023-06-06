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
        self.layers = [nn.Linear(input_dim, hidden_dims[0]), nn.ReLU()]

        for hidden_in, hidden_out in zip(hidden_dims[:-1], hidden_dims[1:]):
            self.layers.extend([nn.Linear(hidden_in, hidden_out), nn.ReLU()])

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
        self.layers = [nn.Linear(input_dim, hidden_dims[0]), nn.ReLU()]

        for hidden_in, hidden_out in zip(hidden_dims[:-1], hidden_dims[1:]):
            self.layers.extend([nn.Linear(hidden_in, hidden_out), nn.ReLU()])

        self.layers.append(nn.Linear(hidden_dims[-1], output_dim))
        self.layers = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.layers(x)


class MOA2C(DRLOracle):
    def __init__(self,
                 env,
                 aug=0.2,
                 lrs=(0.001, 0.001),
                 hidden_layers=((64, 64), (64, 64)),
                 one_hot=False,
                 e_coef=0.01,
                 v_coef=0.5,
                 gamma=0.99,
                 tau=1.0,
                 max_grad_norm=0.5,
                 normalize_advantage=True,
                 n_steps=10,
                 gae_lambda=0.5,
                 global_steps=100000,
                 eval_episodes=100,
                 log_freq=1000,
                 seed=0):
        super().__init__(env, aug=aug, gamma=gamma, one_hot=one_hot, eval_episodes=eval_episodes)

        if len(lrs) == 1:  # Use same learning rate for all models.
            lrs = (lrs[0], lrs[0])

        if len(hidden_layers) == 1:  # Use same hidden layers for all models.
            hidden_layers = (hidden_layers[0], hidden_layers[0])

        self.actor_lr, self.critic_lr = lrs
        self.e_coef = e_coef
        self.v_coef = v_coef
        self.gamma = gamma
        self.tau = tau
        self.s0 = None

        self.global_steps = global_steps
        self.eval_episodes = eval_episodes
        self.log_freq = log_freq

        self.seed = seed
        self.rng = np.random.default_rng(seed=seed)

        self.one_hot = one_hot
        self.input_dim = self.obs_dim + self.num_objectives
        self.actor_output_dim = int(self.num_actions)
        self.output_dim_critic = int(self.num_objectives)
        self.actor_layers, self.critic_layers = hidden_layers

        self.actor = None
        self.critic = None
        self.policy = None
        self.actor_optimizer = None
        self.critic_optimizer = None
        self.max_grad_norm = max_grad_norm
        self.normalize_advantage = normalize_advantage
        self.gae_lambda = gae_lambda

        self.n_steps = n_steps
        self.rollout_buffer = RolloutBuffer((self.obs_dim,),
                                            env.action_space.shape,
                                            rew_dim=self.num_objectives,
                                            max_size=self.n_steps,
                                            action_dtype=int,
                                            aug_obs=True)

    def reset(self):
        """Reset the actor and critic networks, optimizers and policy."""
        self.actor = Actor(self.input_dim, self.actor_layers, self.actor_output_dim)
        self.critic = Critic(self.input_dim, self.critic_layers, self.output_dim_critic)
        self.policy = Categorical()
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.critic_lr)

    def get_config(self):
        return {
            "aug": self.aug,
            "lrs": (self.actor_lr, self.critic_lr),
            "hidden_layers": (self.actor_layers, self.critic_layers),
            "e_coef": self.e_coef,
            "v_coef": self.v_coef,
            "gamma": self.gamma,
            "tau": self.tau,
            "max_grad_norm": self.max_grad_norm,
            "normalize_advantage": self.normalize_advantage,
            "n_steps": self.n_steps,
            "gae_lambda": self.gae_lambda,
            "global_steps": self.global_steps,
            "eval_episodes": self.eval_episodes,
            "log_freq": self.log_freq,
            "seed": self.seed
        }

    def calc_returns(self, rewards, dones, v_preds):
        """Compute the returns from the rewards and values.

        Args:
            rewards (Tensor): The rewards.
            dones (Tensor): The dones.
            v_preds (Tensor): The predicted values for the observations and final next observation.

        Returns:
            Tensor: The returns.
        """
        returns = torch.zeros_like(rewards)
        returns[-1] = rewards[-1] + self.gamma * v_preds[-1] * (1 - dones[-1])
        for t in reversed(range(len(rewards) - 1)):
            returns[t] = rewards[t] + self.gamma * (1 - dones[t]) * returns[t + 1]
        return returns

    def calc_generalised_advantages(self, rewards, dones, v_preds):
        """Compute the advantages for the rollouts.

        Args:
            rewards (Tensor): The rewards.
            dones (Tensor): The dones.
            v_preds (Tensor): The predicted values for the observations and final next observation.

        Returns:
            Tensor: The advantages.
        """
        td_errors = rewards + self.gamma * (1 - dones) * v_preds[1:] - v_preds[:-1]
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

        v_preds = self.critic(torch.cat((aug_obs, aug_next_obs[-1:]), dim=0))  # Predict values of observations.
        returns = self.calc_returns(rewards, dones, v_preds)
        advantages = self.calc_generalised_advantages(rewards, dones, v_preds)

        if self.normalize_advantage:
            advantages = (advantages - advantages.mean(dim=0)) / (advantages.std(dim=0) + 1e-8)

        log_prob = self.policy.log_prob(actions.unsqueeze(dim=-1), self.actor(aug_obs))
        pg_loss = -(advantages * log_prob).mean(dim=0)  # Policy gradient loss with advantage as baseline.
        policy_loss = torch.dot(v_s0.grad, pg_loss)  # Gradient update rule for SER.
        entropy_loss = -torch.mean(-log_prob)
        value_loss = F.mse_loss(returns, v_preds[:-1])

        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        loss = policy_loss + self.v_coef * value_loss + self.e_coef * entropy_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        self.actor_optimizer.step()
        self.critic_optimizer.step()

    def reset_env(self):
        """Reset the environment.

        Returns:
            Tuple: The initial observation, accrued reward, augmented observation and timestep.
        """
        raw_obs, _ = self.env.reset()
        obs = self.format_obs(raw_obs)
        accrued_reward = np.zeros(self.num_objectives)
        aug_obs = torch.tensor(np.concatenate((obs, accrued_reward)), dtype=torch.float)  # Create the augmented state.
        timestep = 0
        self.s0 = aug_obs
        return obs, accrued_reward, aug_obs, timestep

    def select_action(self, aug_obs):
        """Select an action from the policy.

        Args:
            aug_obs (Tensor): The augmented observation.

        Returns:
            int: The action.
        """
        log_probs = self.actor(aug_obs)  # Logprobs for the actions.
        action = self.policy(log_probs).item()  # Sample an action from the distribution.
        return action

    def select_greedy_action(self, obs, accrued_reward):
        """Select a greedy action. Used by the solve method in the super class.

        Args:
            obs (Tensor): The observation.
            accrued_reward (Tensor): The accrued reward.

        Returns:
            int: The action.
        """
        aug_obs = torch.tensor(np.concatenate((obs, accrued_reward)), dtype=torch.float)
        log_probs = self.actor(aug_obs)  # Logprobs for the actions.
        action = self.policy.greedy(log_probs).item()  # Sample an action from the distribution.
        return action

    def train(self):
        """Train the agent."""
        obs, accrued_reward, aug_obs, timestep = self.reset_env()

        for global_step in range(self.global_steps):
            if global_step % self.log_freq == 0:
                print(f'Global step: {global_step}')

            with torch.no_grad():
                action = self.select_action(aug_obs)

            next_raw_obs, reward, terminated, truncated, _ = self.env.step(action)
            next_obs = self.format_obs(next_raw_obs)
            aug_next_obs = torch.tensor(np.concatenate((next_obs, accrued_reward)), dtype=torch.float)
            accrued_reward += (self.gamma ** timestep) * reward  # Update the accrued reward.
            self.rollout_buffer.add(aug_obs, action, reward, aug_next_obs, terminated or truncated)

            if (global_step + 1) % self.n_steps == 0:
                self.update_policy()
                self.rollout_buffer.reset()

            aug_obs = aug_next_obs
            timestep += 1

            if terminated or truncated:  # If the episode is done, reset the environment and accrued reward.
                obs, accrued_reward, aug_obs, timestep = self.reset_env()

    def load_model(self, referent, load_actor=False, load_critic=True):
        """Load the model that is closest to the given referent.

        Args:
            referent (ndarray): The referent to load the model for.
            load_actor (bool, optional): Whether to load the actor. The rationale to not loading the actor is that the
                policy might already be very close to deterministic making it very hard to escape. Defaults to False.
            load_critic (bool, optional): Whether to load the critic. The rationale to loading the critic is that the
                value function estimates may help the policy navigate to better states faster. Defaults to True.
        """
        closest_referent = self.get_closest_referent(referent)
        if closest_referent:
            actor_net, critic_net = self.trained_models[tuple(closest_referent)]
            if load_actor:
                self.actor.load_state_dict(actor_net)
            if load_critic:
                self.critic.load_state_dict(critic_net)

    def solve(self, referent, ideal, warm_start=True):
        """Train the algorithm on the given environment."""
        self.reset()
        if warm_start:
            self.load_model(referent)
        pareto_point = super().solve(referent, ideal)
        self.trained_models[tuple(referent)] = (self.actor.state_dict(), self.critic.state_dict())
        return pareto_point
