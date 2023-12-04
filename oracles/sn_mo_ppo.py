import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from oracles.policy import Categorical
from oracles.sn_drl_oracle import SNDRLOracle
from oracles.replay_buffer import RolloutBuffer
from oracles.vector_u import aasf


class Actor(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super().__init__()
        self.layers = [nn.Linear(input_dim, hidden_dims[0]), nn.Tanh()]

        for hidden_in, hidden_out in zip(hidden_dims[:-1], hidden_dims[1:]):
            self.layers.extend([nn.Linear(hidden_in, hidden_out), nn.Tanh()])

        self.layers.append(nn.Linear(hidden_dims[-1], output_dim))
        self.layers = nn.Sequential(*self.layers)

    def forward(self, obs, ref):
        concat_shape = obs.shape[:-1] + ref.shape[-1:]
        exp_ref = ref.expand(*concat_shape)
        x = torch.cat((obs, exp_ref), dim=-1)
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

    def forward(self, obs, ref):
        concat_shape = obs.shape[:-1] + ref.shape[-1:]
        exp_ref = ref.expand(*concat_shape)
        x = torch.cat((obs, exp_ref), dim=-1)
        return self.layers(x)


class SNMOPPO(SNDRLOracle):
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
                 pretrain_iters=100,
                 num_referents=16,
                 pretraining_steps=500000,
                 online_steps=500000,
                 eval_episodes=100,
                 log_freq=1000,
                 seed=0):
        super().__init__(envs.unwrapped.envs[0],
                         aug=aug,
                         scale=scale,
                         gamma=gamma,
                         pretrain_iters=pretrain_iters,
                         num_referents=num_referents,
                         pretraining_steps=pretraining_steps,
                         online_steps=online_steps,
                         eval_episodes=eval_episodes,
                         track=track,
                         seed=seed)

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
        self.log_freq = log_freq

        self.batch_size = int(self.num_envs * self.n_steps)
        self.minibatch_size = int(self.batch_size // self.num_minibatches)
        self.num_pretrain_updates = int(self.pretraining_steps // self.batch_size)
        self.num_online_updates = int(self.online_steps // self.batch_size)

        self.input_dim = self.aug_obs_dim + self.num_objectives  # Obs + accrued reward + referent.
        self.output_dim_actor = int(self.num_actions)
        self.output_dim_critic = int(self.num_objectives)
        self.actor_hidden = actor_hidden
        self.critic_hidden = critic_hidden

        self.actor = None
        self.critic = None
        self.policy = None
        self.actor_optimizer = None
        self.critic_optimizer = None
        self.rollout_buffer = RolloutBuffer((self.num_envs, self.aug_obs_dim),
                                            (self.num_envs,) + self.envs.single_action_space.shape,
                                            rew_dim=(self.num_envs, self.num_objectives),
                                            dones_dim=(self.num_envs, 1),
                                            max_size=self.batch_size,
                                            action_dtype=int,
                                            rng=self.np_rng)

    def config(self):
        """Return the configuration of the oracle."""
        config = super().config()
        config.update({
            'lr_actor': self.lr_actor,
            'lr_critic': self.lr_critic,
            'eps': self.eps,
            'actor_hidden': self.actor_hidden,
            'critic_hidden': self.critic_hidden,
            'anneal_lr': self.anneal_lr,
            'e_coef': self.e_coef,
            'v_coef': self.v_coef,
            'num_envs': self.num_envs,
            'num_minibatches': self.num_minibatches,
            'update_epochs': self.update_epochs,
            'max_grad_norm': self.max_grad_norm,
            'target_kl': self.target_kl,
            'normalize_advantage': self.normalize_advantage,
            'clip_coef': self.clip_coef,
            'clip_range_vf': self.clip_range_vf,
            'gae_lambda': self.gae_lambda,
            'n_steps': self.n_steps,
            'pretrain_iters': self.pretrain_iters,
            'num_referents': self.num_referents,
            'pretraining_steps': self.pretraining_steps,
            'online_steps': self.online_steps,
            'eval_episodes': self.eval_episodes,
            'log_freq': self.log_freq,
            'seed': self.seed
        })
        return config

    def reset(self):
        """Reset the actor and critic networks, optimizers and policy."""
        self.actor = Actor(self.input_dim, self.actor_hidden, self.output_dim_actor)
        self.actor.apply(self.init_weights)
        self.critic = Critic(self.input_dim, self.critic_hidden, self.output_dim_critic)
        self.critic.apply(self.init_weights)
        self.policy = Categorical()
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr_actor, eps=self.eps)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.lr_critic, eps=self.eps)
        self.rollout_buffer.reset()

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
        self.critic.load_state_dict(self.pretrained_model["critic_state_dict"])
        self.actor_optimizer.load_state_dict(self.pretrained_model["actor_optimizer_state_dict"])
        self.critic_optimizer.load_state_dict(self.pretrained_model["critic_optimizer_state_dict"])

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

    def perform_update(self,
                       referent,
                       nadir,
                       ideal,
                       mb_aug_obs,
                       mb_actions,
                       mb_values,
                       mb_returns,
                       mb_advantages,
                       mb_logprobs):
        """Perform an update step."""
        # Get the current policy log probabilities and values.
        actor_out = self.actor(mb_aug_obs, referent)
        newlogprob, entropy = self.policy.evaluate_actions(actor_out, mb_actions)  # Evaluate actions.
        newvalue = self.critic(mb_aug_obs, referent)
        logratio = newlogprob - mb_logprobs
        ratio = logratio.exp()  # Ratio is the same for all objectives.

        if self.normalize_advantage:
            mb_advantages = (mb_advantages - mb_advantages.mean(dim=0)) / (mb_advantages.std(dim=0) + 1e-8)

        # Compute the policy loss.
        pg_loss1 = mb_advantages * ratio
        pg_loss2 = mb_advantages * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)  # PPO loss.
        pg_loss3 = -torch.min(pg_loss1, pg_loss2).mean(dim=0)

        with torch.no_grad():
            v_s0 = self.critic(self.s0, referent)  # Value of s0.

        v_s0.requires_grad = True
        aasf(v_s0,
             referent,
             nadir,
             ideal,
             aug=self.aug,
             scale=self.scale,
             backend='torch').backward()  # Gradient of utility function w.r.t. values.

        pg_loss = torch.dot(v_s0.grad, pg_loss3)  # The policy loss for nonlinear utility functions.

        # Compute the value loss
        if self.clip_range_vf is not None:
            values_pred = mb_values + torch.clamp(newvalue - mb_values, -self.clip_range_vf, self.clip_range_vf)
        else:
            values_pred = newvalue
        value_loss = F.mse_loss(mb_returns, values_pred)

        entropy_loss = -torch.mean(entropy)
        loss = pg_loss + self.v_coef * value_loss + self.e_coef * entropy_loss  # The total loss.
        return loss

    def update_policy(self, referent, nadir, ideal, num_referents=1):
        """Update the policy using the rollout buffer."""
        with torch.no_grad():
            aug_obs, actions, rewards, aug_next_obs, dones = self.rollout_buffer.get_all_data(to_tensor=True)
            all_obs = torch.cat((aug_obs, aug_next_obs[-1:]), dim=0)
            values = self.critic(all_obs, referent)
            advantages = self.calc_generalised_advantages(rewards, dones, values)  # Calculate the advantages.
            returns = advantages + values[:-1]  # Calculate the returns.
            actor_out = self.actor(aug_obs, referent)
            log_prob, _ = self.policy.evaluate_actions(actor_out, actions)

            # Flatten the data.
            aug_obs = aug_obs.view(-1, self.aug_obs_dim)
            actions = actions.view(-1)
            log_prob = log_prob.view(-1, 1)
            values = values.view(-1, self.num_objectives)
            advantages = advantages.view(-1, self.num_objectives)
            returns = returns.view(-1, self.num_objectives)

        referents = torch.unsqueeze(referent, dim=0)
        if num_referents > 1:
            additional_referents = self.sample_referents(num_referents - 1, nadir, ideal)
            referents = torch.cat((referents, additional_referents), dim=0)

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

                loss = torch.tensor(0., dtype=torch.float)

                for ref in referents:
                    ref_loss = self.perform_update(ref,
                                                   nadir,
                                                   ideal,
                                                   mb_aug_obs,
                                                   mb_actions,
                                                   mb_values,
                                                   mb_returns,
                                                   mb_advantages,
                                                   mb_logprobs)
                    loss += ref_loss

                loss /= num_referents

                # Update the actor and critic networks.
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()
                self.critic_optimizer.step()

        return loss.item()

    def select_action(self, aug_obs, accrued_reward, referent, nadir, ideal):
        """Select an action from the policy."""
        log_probs = self.actor(aug_obs, referent)  # Logprobs for the actions.
        actions = self.policy(log_probs, aug_obs=aug_obs)  # Sample an action from the distribution.
        if len(actions) == 1:
            return actions.item()
        else:
            return np.array(actions.squeeze())

    def select_greedy_action(self, aug_obs, accrued_reward, referent, nadir, ideal):
        """Select a greedy action."""
        log_probs = self.actor(aug_obs, referent)  # Logprobs for the actions.
        action = self.policy.greedy(log_probs).item()  # Sample an action from the distribution.
        return action

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
                       num_referents=self.num_referents,
                       num_updates=self.num_pretrain_updates)

        self.save_model()

    def train(self,
              referent,
              nadir,
              ideal,
              steps=None,
              num_referents=None,
              num_updates=None,
              *args,
              **kwargs):
        """Train the agent."""
        global_step = 0
        obs, _ = self.envs.reset()
        obs = torch.tensor(np.nan_to_num(obs, posinf=0))
        acs = torch.zeros((self.num_envs, self.num_objectives), dtype=torch.float)
        aug_obs = torch.hstack((obs, acs))
        self.s0 = aug_obs[0].detach()
        timesteps = torch.zeros((self.num_envs, 1))
        steps_since_log = 0

        for update in range(num_updates):
            if self.anneal_lr:  # Update the learning rate.
                lr_frac = 1. - update / num_updates
                self.actor_optimizer.param_groups[0]['lr'] = lr_frac * self.lr_actor
                self.critic_optimizer.param_groups[0]['lr'] = lr_frac * self.lr_critic

            # Perform rollouts in the environments.
            for step in range(self.n_steps):
                if global_step % self.log_freq == 0:
                    print(f'Global step: {global_step}')

                with torch.no_grad():
                    actions = self.select_action(aug_obs, acs, referent, nadir, ideal)

                next_obs, rewards, terminateds, truncateds, info = self.envs.step(actions)
                dones = np.expand_dims(terminateds | truncateds, axis=1)
                next_obs = np.nan_to_num(next_obs, posinf=0)
                rewards = np.nan_to_num(rewards, posinf=0)
                acs = (acs + (self.gamma ** timesteps) * rewards) * (1 - dones)  # Update the accrued reward.
                aug_next_obs = torch.tensor(np.hstack((next_obs, acs)), dtype=torch.float)
                self.rollout_buffer.add(aug_obs, actions, rewards, aug_next_obs, np.expand_dims(terminateds, axis=-1))

                aug_obs = aug_next_obs
                timesteps = (timesteps + 1) * (1 - dones)

                self.save_vectorized_episodic_stats(info, dones, referent, nadir, ideal)

                global_step += self.num_envs  # The global step is 1 * the number of environments.
                steps_since_log += self.num_envs

            loss = self.update_policy(referent, nadir, ideal, num_referents=num_referents)
            if steps_since_log - self.log_freq >= 0:
                self.log_pg(global_step, loss, 0, 0, 0)  # Don't log the constituent losses.
                steps_since_log = 0
            self.rollout_buffer.reset()

    def solve(self, referent, nadir=None, ideal=None, *args, **kwargs):
        """Train the algorithm on the given environment."""
        self.reset()
        self.setup_ac_metrics()
        pareto_point = super().solve(referent,
                                     nadir=nadir,
                                     ideal=ideal,
                                     steps=self.online_steps,
                                     num_updates=self.num_online_updates)
        return pareto_point
