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


class MOPPO(DRLOracle):
    def __init__(self,
                 envs,
                 gamma,
                 track=False,
                 aug=0.2,
                 scale=1000,
                 lr_actor=2.5e-4,
                 lr_critic=2.5e-4,
                 eps=1e-8,
                 actor_hidden=(64, 64),
                 critic_hidden=(64, 64),
                 one_hot=False,
                 anneal_lr=False,
                 e_coef=0.01,
                 v_coef=0.5,
                 num_envs=4,
                 num_minibatches=4,
                 update_epochs=4,
                 max_grad_norm=0.5,
                 target_kl=None,
                 normalize_advantage=True,
                 clip_coef=0.2,
                 clip_range_vf=None,
                 gae_lambda=0.95,
                 n_steps=128,
                 global_steps=500000,
                 eval_episodes=100,
                 log_freq=1000,
                 window_size=100,
                 seed=0):
        super().__init__(envs.envs[0],
                         track=track,
                         aug=aug,
                         scale=scale,
                         gamma=gamma,
                         one_hot=one_hot,
                         eval_episodes=eval_episodes,
                         window_size=window_size, )
        self.envs = envs
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.eps = eps
        self.s0 = None

        self.anneal_lr = anneal_lr
        self.e_coef = e_coef
        self.v_coef = v_coef
        self.num_envs = num_envs
        self.num_minibatches = num_minibatches
        self.update_epochs = update_epochs
        self.max_grad_norm = max_grad_norm
        self.target_kl = target_kl
        self.normalize_advantage = normalize_advantage
        self.clip_coef = clip_coef
        self.clip_range_vf = clip_range_vf
        self.gae_lambda = gae_lambda

        self.n_steps = n_steps
        self.global_steps = int(global_steps)
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
        self.actor_hidden = actor_hidden
        self.critic_hidden = critic_hidden

        self.actor = None
        self.critic = None
        self.policy = None
        self.actor_optimizer = None
        self.critic_optimizer = None
        self.actor_scheduler = None
        self.critic_scheduler = None

        self.rollout_buffer = RolloutBuffer((self.num_envs, self.obs_dim),
                                            (self.num_envs,) + envs.single_action_space.shape,
                                            rew_dim=(self.num_envs, self.num_objectives),
                                            dones_dim=(self.num_envs, 1),
                                            max_size=self.batch_size,
                                            action_dtype=int,
                                            aug_obs=True)

        self.log_freq = log_freq

    def config(self):
        return {
            "gamma": self.gamma,
            "track": self.track,
            "aug": self.aug,
            "scale": self.scale,
            "lr_actor": self.lr_actor,
            "lr_critic": self.lr_critic,
            "eps": self.eps,
            "actor_hidden": self.actor_hidden,
            "critic_hidden": self.critic_hidden,
            "one_hot": self.one_hot,
            "anneal_lr": self.anneal_lr,
            "e_coef": self.e_coef,
            "v_coef": self.v_coef,
            "num_envs": self.num_envs,
            "num_minibatches": self.num_minibatches,
            "update_epochs": self.update_epochs,
            "max_grad_norm": self.max_grad_norm,
            "target_kl": self.target_kl,
            "normalize_advantage": self.normalize_advantage,
            "clip_coef": self.clip_coef,
            "clip_range_vf": self.clip_range_vf,
            "gae_lambda": self.gae_lambda,
            "n_steps": self.n_steps,
            "global_steps": self.global_steps,
            "eval_episodes": self.eval_episodes,
            "log_freq": self.log_freq,
            "window_size": self.window_size,
            "seed": self.seed
        }

    def reset(self):
        """Reset the actor and critic networks, optimizers and policy."""
        self.actor = Actor(self.input_dim, self.actor_hidden, self.actor_output_dim)
        self.actor.apply(self.init_weights)
        self.critic = Critic(self.input_dim, self.critic_hidden, self.output_dim_critic)
        self.critic.apply(self.init_weights)
        self.policy = Categorical()
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr_actor, eps=self.eps)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.lr_critic, eps=self.eps)
        self.policy_returns = []
        self.rollout_buffer.reset()

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

    def update_policy(self):
        """Update the policy using the rollout buffer."""
        with torch.no_grad():
            aug_obs, actions, rewards, aug_next_obs, dones = self.rollout_buffer.get_all_data(to_tensor=True)
            values = self.critic(torch.cat((aug_obs, aug_next_obs[-1:]), dim=0))  # Predict values of observations.
            advantages = self.calc_generalised_advantages(rewards, dones, values)  # Calculate the advantages.
            returns = advantages + values[:-1]  # Calculate the returns.
            actor_out = self.actor(aug_obs)
            log_prob, _ = self.policy.evaluate_actions(actor_out, actions)

            # Flatten the data.
            aug_obs = aug_obs.view(-1, self.input_dim)
            actions = actions.view(-1)
            log_prob = log_prob.view(-1, 1)
            values = values.view(-1, self.num_objectives)
            advantages = advantages.view(-1, self.num_objectives)
            returns = returns.view(-1, self.num_objectives)

        for epoch in range(self.update_epochs):
            shuffled_inds = torch.randperm(self.batch_size, generator=self.torch_rng)
            for mb_inds in torch.chunk(shuffled_inds, self.num_minibatches):
                # Get the minibatch data.
                mb_aug_obs = aug_obs[mb_inds]
                mb_actions = actions[mb_inds]
                mb_values = values[mb_inds]
                mb_advantages = advantages[mb_inds]
                mb_returns = returns[mb_inds]
                mb_logprobs = log_prob[mb_inds]

                # Get the current policy log probabilities and values.
                actor_out = self.actor(mb_aug_obs)
                newlogprob, entropy = self.policy.evaluate_actions(actor_out, mb_actions)  # Evaluate actions.
                newvalue = self.critic(mb_aug_obs)
                logratio = newlogprob - mb_logprobs
                ratio = logratio.exp()  # Ratio is the same for all objectives.

                if self.normalize_advantage:
                    mb_advantages = (mb_advantages - mb_advantages.mean(dim=0)) / (mb_advantages.std(dim=0) + 1e-8)

                # Compute the policy loss.
                pg_loss1 = mb_advantages * ratio
                pg_loss2 = mb_advantages * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)  # PPO loss.
                pg_loss3 = -torch.min(pg_loss1, pg_loss2).mean(dim=0)

                with torch.no_grad():
                    v_s0 = self.critic(self.s0)  # Value of s0.

                v_s0.requires_grad = True
                self.u_func(v_s0).backward()  # Gradient of utility function w.r.t. values.

                pg_loss = torch.dot(v_s0.grad, pg_loss3)  # The policy loss for nonlinear utility functions.

                # Compute the value loss
                if self.clip_range_vf is not None:
                    values_pred = mb_values + torch.clamp(newvalue - mb_values, -self.clip_range_vf, self.clip_range_vf)
                else:
                    values_pred = newvalue
                value_loss = F.mse_loss(mb_returns, values_pred)

                entropy_loss = -torch.mean(entropy)
                loss = pg_loss + self.v_coef * value_loss + self.e_coef * entropy_loss  # The total loss.

                # Update the actor and critic networks.
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()
                self.critic_optimizer.step()

            if self.target_kl is not None:
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - logratio).mean()
                if approx_kl > self.target_kl:
                    break

        return loss.item(), pg_loss.item(), value_loss.item(), entropy_loss.item()

    def select_action(self, aug_obs, acs):
        """Select an action from the policy.

        Args:
            aug_obs (Tensor): The augmented observation.
            acs (ndarray): The accrued rewards. This is not used in this algorithm.

        Returns:
            int: The action.
        """
        log_probs = self.actor(aug_obs)  # Logprobs for the actions.
        actions = self.policy(log_probs, aug_obs=aug_obs)  # Sample an action from the distribution.
        if len(actions) == 1:
            return actions.item()
        else:
            return np.array(actions.squeeze())

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
        global_step = 0
        raw_obs, _ = self.envs.reset()
        obs = torch.tensor(self.format_obs(np.nan_to_num(raw_obs), vectorized=True), dtype=torch.float)
        acs = torch.zeros((self.num_envs, self.num_objectives), dtype=torch.float)
        aug_obs = torch.hstack((obs, acs))
        self.s0 = aug_obs[0].detach()
        timesteps = torch.zeros((self.num_envs, 1))
        steps_since_log = 0

        for update in range(self.num_updates):
            if self.anneal_lr:  # Update the learning rate.
                lr_frac = 1. - update / self.num_updates
                self.actor_optimizer.param_groups[0]['lr'] = lr_frac * self.lr_actor
                self.critic_optimizer.param_groups[0]['lr'] = lr_frac * self.lr_critic

            # Perform rollouts in the environments.
            for step in range(self.n_steps):
                if global_step % self.log_freq == 0:
                    print(f'Global step: {global_step}')

                with torch.no_grad():
                    actions = self.select_action(aug_obs, acs)

                next_raw_obs, rewards, terminateds, truncateds, info = self.envs.step(actions)
                dones = np.expand_dims(terminateds | truncateds, axis=1)
                next_obs = self.format_obs(np.nan_to_num(next_raw_obs), vectorized=True)
                acs = (acs + (self.gamma ** timesteps) * rewards) * (1 - dones)  # Update the accrued reward.
                aug_next_obs = torch.tensor(np.hstack((next_obs, acs)), dtype=torch.float)

                self.rollout_buffer.add(aug_obs, actions, rewards, aug_next_obs, np.expand_dims(terminateds, axis=-1))

                aug_obs = aug_next_obs
                timesteps = (timesteps + 1) * (1 - dones)

                self.save_vectorized_episodic_stats(info, dones)

                global_step += self.num_envs  # The global step is 1 * the number of environments.
                steps_since_log += self.num_envs

            loss, pg_l, v_l, e_l = self.update_policy()
            if steps_since_log - self.log_freq >= 0:
                self.log_pg(global_step, loss, pg_l, v_l, e_l)
                steps_since_log = 0
            self.rollout_buffer.reset()

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
        self.setup_ac_metrics()
        if warm_start:
            self.load_model(referent)
        pareto_point = super().solve(referent, ideal)
        self.trained_models[tuple(referent)] = (self.actor.state_dict(), self.critic.state_dict())
        return pareto_point
