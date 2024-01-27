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


class SNMOA2C(SNDRLOracle):
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
                 e_coef=0.01,
                 v_coef=0.5,
                 max_grad_norm=0.5,
                 normalize_advantage=False,
                 n_steps=16,
                 gae_lambda=0.5,
                 pretrain_iters=100,
                 grid_sample=False,
                 num_referents=16,
                 pretraining_steps=100000,
                 online_steps=100000,
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
                         alg_name='SN-MO-A2C')

        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.actor_hidden = actor_hidden
        self.critic_hidden = critic_hidden
        self.actor_activation = actor_activation
        self.critic_activation = critic_activation
        self.e_coef = e_coef
        self.v_coef = v_coef
        self.max_grad_norm = max_grad_norm
        self.normalize_advantage = normalize_advantage
        self.n_steps = n_steps
        self.gae_lambda = gae_lambda
        self.log_freq = log_freq

        self.input_dim = self.aug_obs_dim + self.num_objectives  # Obs + accrued reward + referent.
        self.output_dim_actor = int(self.num_actions)
        self.output_dim_critic = int(self.num_objectives)
        self.actor = None
        self.critic = None
        self.policy = None
        self.actor_optimizer = None
        self.critic_optimizer = None
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
            'actor_hidden': self.actor_hidden,
            'critic_hidden': self.critic_hidden,
            'actor_activation': self.actor_activation,
            'critic_activation': self.critic_activation,
            'e_coef': self.e_coef,
            'v_coef': self.v_coef,
            'max_grad_norm': self.max_grad_norm,
            'normalize_advantage': self.normalize_advantage,
            'n_steps': self.n_steps,
            'gae_lambda': self.gae_lambda,
            'log_freq': self.log_freq
        })
        return config

    def reset(self):
        """Reset the actor and critic networks, optimizers and policy."""
        self.actor = Actor(self.input_dim,
                           self.actor_hidden,
                           self.output_dim_actor,
                           activation=self.actor_activation)
        self.critic = Critic(self.input_dim,
                             self.critic_hidden,
                             self.output_dim_critic,
                             activation=self.critic_activation)
        self.policy = Categorical()
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr_actor)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.lr_critic)
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

    def perform_update(self, referent, nadir, ideal, aug_obs, actions, rewards, aug_next_obs, dones, b_log_probs):
        """Perform an update step."""
        with torch.no_grad():
            v_s0 = self.critic(self.s0, referent)
        v_s0.requires_grad = True
        aasf(v_s0,
             referent,
             nadir,
             ideal,
             aug=self.aug,
             scale=self.scale,
             backend='torch').backward()  # Gradient of utility function w.r.t. values.

        values = self.critic(aug_obs, referent)
        actor_out = self.actor(aug_obs, referent)  # Predict logits of actions.
        ref_log_probs, entropy = self.policy.evaluate_actions(actor_out, actions)  # Evaluate actions.

        with torch.no_grad():
            v_next = self.critic(aug_next_obs[-1:], referent[None, ...])  # Predict values of next observations.
            advantages = self.calc_generalised_advantages(rewards, dones, values, v_next)
            returns = advantages + values

            # Compute importance sampling ratios.
            is_ratios = torch.exp(ref_log_probs - b_log_probs)

        if self.normalize_advantage:
            advantages = (advantages - advantages.mean(dim=0)) / (advantages.std(dim=0) + 1e-8)

        pg_loss = (advantages * ref_log_probs * is_ratios).mean(dim=0)  # PG loss corrected with importance sampling.
        policy_loss = -torch.dot(v_s0.grad, pg_loss)  # Gradient update rule for SER.
        entropy_loss = -torch.mean(entropy)  # Compute entropy bonus.
        value_loss = F.mse_loss(returns, values)

        loss = policy_loss + self.v_coef * value_loss + self.e_coef * entropy_loss
        return loss

    def update_policy(self, referent, nadir, ideal, num_referents=1):
        """Update the policy using the rollout buffer."""
        aug_obs, actions, rewards, aug_next_obs, dones = self.rollout_buffer.get_all_data(to_tensor=True)
        referents = torch.unsqueeze(referent, dim=0)
        if num_referents > 1:
            additional_referents = self.uniform_sample_referents(num_referents - 1, nadir, ideal)
            referents = torch.cat((referents, additional_referents), dim=0)

        with torch.no_grad():
            actor_out = self.actor(aug_obs, referent)
            log_probs, _ = self.policy.evaluate_actions(actor_out, actions)

        loss = torch.tensor(0, dtype=torch.float)

        for ref in referents:
            loss += self.perform_update(ref, nadir, ideal, aug_obs, actions, rewards, aug_next_obs, dones, log_probs)

        loss /= num_referents

        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        self.critic_optimizer.step()
        self.actor_optimizer.step()

        return loss.item()

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

    def select_action(self, aug_obs, accrued_reward, referent, nadir, ideal):
        """Select an action from the policy.

        Args:
            aug_obs (Tensor): The augmented observation.
            accrued_reward (ndarray): The accrued reward. This is not used in this algorithm.
            referent (Tensor): The referent.
            nadir (Tensor): The nadir point.
            ideal (Tensor): The ideal point.

        Returns:
            int: The action.
        """
        log_probs = self.actor(aug_obs, referent)  # Logprobs for the actions.
        action = self.policy(log_probs, aug_obs=aug_obs).item()  # Sample an action from the distribution.
        return action

    def select_greedy_action(self, aug_obs, accrued_reward, referent, nadir, ideal):
        """Select a greedy action. Used by the solve method in the super class.

        Args:
            aug_obs (Tensor): The augmented observation.
            accrued_reward (ndarray): The accrued reward. This is not used in this algorithm.
            referent (Tensor): The referent.
            nadir (Tensor): The nadir point.
            ideal (Tensor): The ideal point.

        Returns:
            int: The action.
        """
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
                       num_referents=self.num_referents)

        self.save_model()

    def train(self,
              referent,
              nadir,
              ideal,
              steps=None,
              num_referents=None,
              *args,
              **kwargs):
        """Train the agent."""
        aug_obs, accrued_reward, timestep = self.reset_env()
        self.s0 = aug_obs
        loss = 0
        pg_l = 0
        v_l = 0
        e_l = 0

        for step in range(steps):
            if step % self.log_freq == 0:
                print(f'{self.phase} step: {step}')

            with torch.no_grad():
                action = self.select_action(aug_obs, accrued_reward, referent, nadir, ideal)

            next_obs, reward, terminated, truncated, info = self.env.step(action)
            next_obs = np.nan_to_num(next_obs, posinf=0)
            reward = np.nan_to_num(reward, posinf=0)
            accrued_reward += (self.gamma ** timestep) * reward  # Update the accrued reward.
            aug_next_obs = torch.tensor(np.concatenate((next_obs, accrued_reward)), dtype=torch.float)
            self.rollout_buffer.add(aug_obs, action, reward, aug_next_obs, terminated)

            if (step + 1) % self.n_steps == 0:
                loss = self.update_policy(referent, nadir, ideal, num_referents=num_referents)
                self.rollout_buffer.reset()

            if (step + 1) % self.log_freq == 0:
                self.log_pg(step, loss, pg_l, v_l, e_l)

            aug_obs = aug_next_obs
            timestep += 1

            if terminated or truncated:  # If the episode is done, reset the environment and accrued reward.
                self.save_episode_stats(accrued_reward, timestep, referent, nadir, ideal)
                aug_obs, accrued_reward, timestep = self.reset_env()

    def solve(self, referent, nadir=None, ideal=None, *args, **kwargs):
        """Train the algorithm on the given environment."""
        self.reset()
        self.setup_ac_metrics()
        pareto_point = super().solve(referent,
                                     nadir=nadir,
                                     ideal=ideal,
                                     steps=self.online_steps)
        return pareto_point
