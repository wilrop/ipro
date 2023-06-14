import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical as CDist

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


class MOPPO(DRLOracle):
    def __init__(self,
                 envs,
                 aug=0.2,
                 gamma=0.99,
                 lrs=(2.5e-4, 2.5e-4),
                 eps=1e-5,
                 hidden_layers=((64, 64), (64, 64)),
                 one_hot=False,
                 e_coef=0.01,
                 v_coef=0.5,
                 num_envs=4,
                 num_minibatches=4,
                 update_epochs=4,
                 max_grad_norm=0.5,
                 normalize_advantage=True,
                 clip_coef=0.2,
                 clip_vloss=False,
                 gae_lambda=0.95,
                 n_steps=128,
                 global_steps=500000,
                 eval_episodes=100,
                 log_freq=1000,
                 seed=0):
        super().__init__(envs.envs[0], aug=aug, gamma=gamma, one_hot=one_hot, eval_episodes=eval_episodes)

        if len(lrs) == 1:  # Use same learning rate for all models.
            lrs = (lrs[0], lrs[0])

        if len(hidden_layers) == 1:  # Use same hidden layers for all models.
            hidden_layers = (hidden_layers[0], hidden_layers[0])

        self.envs = envs
        self.actor_lr, self.critic_lr = lrs
        self.eps = eps
        self.s0 = None

        self.e_coef = e_coef
        self.v_coef = v_coef
        self.num_envs = num_envs
        self.num_minibatches = num_minibatches
        self.update_epochs = update_epochs
        self.max_grad_norm = max_grad_norm
        self.normalize_advantage = normalize_advantage
        self.clip_coef = clip_coef
        self.clip_vloss = clip_vloss
        self.gae_lambda = gae_lambda

        self.n_steps = n_steps
        self.global_steps = global_steps
        self.eval_episodes = eval_episodes

        self.batch_size = int(self.num_envs * self.n_steps)
        self.minibatch_size = int(self.batch_size // self.num_minibatches)
        self.num_updates = int(self.global_steps // self.batch_size)

        self.seed = seed
        self.np_rng = np.random.default_rng(seed=seed)
        self.torch_rng = torch.Generator()
        self.torch_rng.manual_seed(seed)

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
        self.actor_scheduler = None
        self.critic_scheduler = None

        self.rollout_buffer = RolloutBuffer((self.obs_dim,),
                                            envs.single_action_space.shape,
                                            rew_dim=self.num_objectives,
                                            max_size=self.batch_size,
                                            action_dtype=int,
                                            aug_obs=True)

        self.log_freq = log_freq
        self.capture_video = False

    def reset(self):
        """Reset the actor and critic networks, optimizers and policy."""
        self.actor = Actor(self.input_dim, self.actor_layers, self.actor_output_dim)
        self.actor.apply(self.init_weights)
        self.critic = Critic(self.input_dim, self.critic_layers, self.output_dim_critic)
        self.critic.apply(self.init_weights)
        self.policy = Categorical()
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.actor_lr, eps=self.eps)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.critic_lr, eps=self.eps)

    @staticmethod
    def init_weights(m, std=np.sqrt(2), bias_const=0.0):
        if isinstance(m, nn.Linear):
            torch.nn.init.orthogonal(m.weight, std)
            torch.nn.init.constant_(m.bias, bias_const)

    def get_config(self):
        return {
            "aug": self.aug,
            "gamma": self.gamma,
            "lrs": (self.actor_lr, self.critic_lr),
            "hidden_layers": (self.actor_layers, self.critic_layers),
            "one_hot": self.one_hot,
            "e_coef": self.e_coef,
            "v_coef": self.v_coef,
            "num_envs": self.num_envs,
            "num_minibatches": self.num_minibatches,
            "update_epochs": self.update_epochs,
            "max_grad_norm": self.max_grad_norm,
            "normalize_advantage": self.normalize_advantage,
            "clip_coef": self.clip_coef,
            "clip_vloss": self.clip_vloss,
            "gae_lambda": self.gae_lambda,
            "n_steps": self.n_steps,
            "global_steps": self.global_steps,
            "eval_episodes": self.eval_episodes,
            "log_freq": self.log_freq,
            "seed": self.seed
        }

    def calc_returns(self, advantages, values):
        """Compute the returns from the advantages and values.

        Notes:
            This method is specific to PPO.
            (see: https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/)

        Args:
            advantages (Tensor): The computed advantages.
            values (Tensor): The predicted values for the observations and final next observation.

        Returns:
            Tensor: The returns.
        """
        return advantages + values[:-1]

    def calc_generalised_advantages(self, rewards, dones, values):
        """Compute the advantages for the rollouts.

        Args:
            rewards (Tensor): The rewards.
            dones (Tensor): The dones.
            values (Tensor): The predicted values for the observations and final next observation.

        Returns:
            Tensor: The advantages.
        """
        td_errors = rewards + self.gamma * (1 - dones) * values[1:] - values[:-1]
        advantages = torch.zeros_like(td_errors)
        advantages[-1] = td_errors[-1]
        for t in reversed(range(len(td_errors) - 1)):
            advantages[t] = td_errors[t] + self.gamma * self.gae_lambda * advantages[t + 1] * (1 - dones[t])
        return advantages

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        probs = CDist(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)

    def update_policy(self):
        """Update the policy using the rollout buffer."""
        with torch.no_grad():
            aug_obs, actions, rewards, aug_next_obs, dones = self.rollout_buffer.get_all_data(to_tensor=True)
            values = self.critic(torch.cat((aug_obs, aug_next_obs[-1:]), dim=0))  # Predict values of observations.
            advantages = self.calc_generalised_advantages(rewards, dones, values)  # Calculate the advantages.
            returns = self.calc_returns(advantages, values)  # Calculate the returns.
            log_prob = CDist(logits=self.actor(aug_obs)).log_prob(actions)

        for epoch in range(self.update_epochs):
            for mb_inds in torch.chunk(torch.randperm(self.batch_size), self.num_minibatches):
                # Get the minibatch data.
                mb_aug_obs = aug_obs[mb_inds]
                mb_actions = actions[mb_inds]
                mb_values = values[mb_inds]
                mb_advantages = advantages[mb_inds]
                mb_returns = returns[mb_inds]
                mb_logprobs = log_prob[mb_inds]

                # Get the current policy log probabilities and values.
                _, newlogprob, entropy, newvalue = self.get_action_and_value(mb_aug_obs, mb_actions)
                logratio = newlogprob - mb_logprobs
                ratio = logratio.exp().unsqueeze(dim=-1)  # Ratio is the same for all objectives.

                if self.normalize_advantage:
                    mb_advantages = (mb_advantages - mb_advantages.mean(dim=0)) / (mb_advantages.std(dim=0) + 1e-8)

                # Compute the policy loss.
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
                pg_loss3 = torch.maximum(pg_loss1, pg_loss2).mean(dim=0)

                with torch.no_grad():
                    v_s0 = self.critic(self.s0)  # Value of s0.
                v_s0.requires_grad = True
                self.u_func(v_s0).backward()  # Gradient of utility function w.r.t. values.

                pg_loss = torch.dot(pg_loss3, v_s0.grad)  # The policy loss for nonlinear utility functions.

                # Compute the value loss
                if self.clip_vloss:
                    values_pred = mb_values + torch.clamp(newvalue - mb_values, -self.clip_coef, self.clip_coef)
                else:
                    values_pred = newvalue
                value_loss = F.mse_loss(mb_returns, values_pred)

                entropy_loss = -entropy.mean()
                loss = pg_loss + self.v_coef * value_loss + self.e_coef * entropy_loss  # The total loss.

                # Update the actor and critic networks.
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()
                self.critic_optimizer.step()

    def select_action(self, aug_obs):
        """Select an action from the policy.

        Args:
            aug_obs (Tensor): The augmented observation.

        Returns:
            int: The action.
        """
        log_probs = self.actor(aug_obs)  # Logprobs for the actions.
        actions = self.policy(log_probs).squeeze()  # Sample an action from the distribution.
        return np.array(actions)

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
        global_step = 0
        raw_obs, _ = self.envs.reset()
        obs = torch.tensor(self.format_obs(raw_obs), dtype=torch.float)
        acs = torch.zeros((self.num_envs, self.num_objectives), dtype=torch.float)
        aug_obs = torch.hstack((obs, acs))
        self.s0 = aug_obs[0].clone()
        timesteps = torch.zeros((self.num_envs, 1))

        for update in range(self.num_updates):
            # Update the learning rate.
            lr_frac = 1. - update / self.num_updates
            self.actor_optimizer.param_groups[0]['lr'] = lr_frac * self.actor_lr
            self.critic_optimizer.param_groups[0]['lr'] = lr_frac * self.critic_lr

            # Perform rollouts in the environments.
            for step in range(self.n_steps):
                if global_step % self.log_freq == 0:
                    print(f'Global step: {global_step}')
                global_step += self.num_envs  # The global step is 1 * the number of environments.

                with torch.no_grad():
                    actions = self.select_action(aug_obs)

                next_raw_obs, rewards, terminateds, truncateds, _ = self.envs.step(actions)
                next_obs = torch.tensor(self.format_obs(next_raw_obs))
                acs += (self.gamma ** timesteps) * rewards  # Update the accrued reward.
                aug_next_obs = torch.tensor(np.hstack((next_obs, acs)), dtype=torch.float)
                dones = np.expand_dims(terminateds | truncateds, axis=1)
                self.rollout_buffer.add(aug_obs, actions, rewards, aug_next_obs, dones, size=self.num_envs)

                aug_obs = aug_next_obs
                timesteps = (timesteps + 1) * (1 - dones)

            self.update_policy()

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
