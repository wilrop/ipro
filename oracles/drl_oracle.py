import torch
import wandb
import numpy as np

from gymnasium.spaces import Box
from collections import deque
import torch.nn as nn

from oracles.vector_u import create_batched_aasf


class DRLOracle:
    def __init__(self,
                 env,
                 aug=0.2,
                 scale=100,
                 gamma=0.99,
                 one_hot=False,
                 warm_start=False,
                 eval_episodes=100,
                 window_size=100,
                 track=False):
        self.env = env
        self.aug = aug
        self.scale = scale

        self.num_actions = env.action_space.n
        self.num_objectives = env.reward_space.shape[0]

        if isinstance(self.env.observation_space, Box):
            low_bound = self.env.observation_space.low
            high_bound = self.env.observation_space.high
            self.obs_shape = self.env.observation_space.shape
            if one_hot:
                self.box_shape = (high_bound[0] - low_bound[0] + 1, high_bound[1] - low_bound[1] + 1)
                self.obs_dim = np.prod(self.box_shape)
            else:
                self.obs_dim = np.prod(self.obs_shape)

        self.gamma = gamma
        self.one_hot = one_hot
        self.eval_episodes = eval_episodes
        self.u_func = None
        self.trained_models = {}  # Collection of trained models that can be used for warm-starting.

        self.iteration = 0

        self.warm_start = warm_start

        self.window_size = window_size
        self.episodic_returns = deque(maxlen=window_size)
        self.episodic_lengths = deque(maxlen=window_size)

        self.track = track

    @staticmethod
    def _compute_grad_norm(model):
        """Compute the gradient norm of the model parameters."""
        total_norm = 0
        for p in model.parameters():
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
        return total_norm ** (1. / 2)

    @staticmethod
    def init_weights(m, std=np.sqrt(2), bias_const=0.01):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight, gain=std)
            torch.nn.init.constant_(m.bias, bias_const)

    def config(self):
        """Return the configuration of the algorithm."""
        raise NotImplementedError

    def reset(self):
        """Reset the environment and the agent."""
        raise NotImplementedError

    def reset_stats(self):
        """Reset the agent's statistics."""
        self.episodic_returns.clear()
        self.episodic_lengths.clear()

    def select_greedy_action(self, aug_obs, accrued_reward):
        """Select the greedy action for the given observation."""
        raise NotImplementedError

    def select_action(self, aug_obs, accrued_reward):
        """Select an action for the given observation."""
        raise NotImplementedError

    def setup_chart_metrics(self):
        """Set up the chart metrics for logging."""
        if self.track:
            wandb.define_metric(f'charts/utility_{self.iteration}', step_metric=f'global_step_{self.iteration}')
            wandb.define_metric(f'charts/episodic_length_{self.iteration}', step_metric=f'global_step_{self.iteration}')
            wandb.define_metric(f'charts/distance_{self.iteration}', step_metric=f'global_step_{self.iteration}')

    def setup_dqn_metrics(self):
        """Set up the metrics for logging."""
        if self.track:
            wandb.define_metric(f'global_step_{self.iteration}')
            wandb.define_metric(f'losses/loss_{self.iteration}', step_metric=f'global_step_{self.iteration}')
            self.setup_chart_metrics()

    def setup_ac_metrics(self):
        """Set up the metrics for logging."""
        if self.track:
            wandb.define_metric(f'global_step_{self.iteration}')
            wandb.define_metric(f'losses/loss_{self.iteration}', step_metric=f'global_step_{self.iteration}')
            wandb.define_metric(f'losses/pg_loss_{self.iteration}', step_metric=f'global_step_{self.iteration}')
            wandb.define_metric(f'losses/value_loss_{self.iteration}', step_metric=f'global_step_{self.iteration}')
            wandb.define_metric(f'losses/entropy_loss_{self.iteration}', step_metric=f'global_step_{self.iteration}')
            wandb.define_metric(f'losses/actor_grad_norm_{self.iteration}', step_metric=f'global_step_{self.iteration}')
            wandb.define_metric(f'losses/critic_grad_norm_{self.iteration}',
                                step_metric=f'global_step_{self.iteration}')
            self.setup_chart_metrics()

    def one_hot_encode(self, obs):
        """One-hot encode the given observation.

        Args:
            obs (ndarray): The observation to one-hot encode.

        Returns:
            ndarray: The one-hot encoded observation.
        """
        dims = obs.ndim
        if dims == 1:
            obs = np.expand_dims(obs, axis=0)
        num_obs = len(obs)
        obs = np.swapaxes(obs, 0, 1)
        flat_obs = np.ravel_multi_index(obs, self.box_shape)
        one_hot_obs = np.zeros((num_obs, self.obs_dim))
        one_hot_obs[np.arange(num_obs), flat_obs] = 1
        if dims == 1:
            one_hot_obs = np.squeeze(one_hot_obs, axis=0)
        return one_hot_obs

    def format_obs(self, obs, vectorized=False):
        """Format the given observation.

        Args:
            obs (ndarray): The observation to format.
            vectorized (bool): Whether the observation is vectorized or not. Defaults to False.

        Returns:
            ndarray: The formatted observation.
        """
        if self.one_hot:
            return self.one_hot_encode(obs)
        elif vectorized:
            return obs.reshape((obs.shape[0], -1))
        else:
            return obs.flatten()

    def evaluate(self, eval_episodes=100, deterministic=True):
        """Evaluate the agent on the environment.

        Args:
            deterministic (bool): Whether to use a deterministic policy or not.

        Returns:
            ndarray: The average reward over the evaluation episodes.
        """
        if deterministic:
            policy = self.select_greedy_action
        else:
            policy = self.select_action

        pareto_point = np.zeros(self.num_objectives)

        for episode in range(eval_episodes):
            raw_obs, _ = self.env.reset()
            obs = self.format_obs(raw_obs)
            terminated = False
            truncated = False
            accrued_reward = np.zeros(self.num_objectives)
            timestep = 0

            while not (terminated or truncated):
                aug_obs = torch.tensor(np.concatenate((obs, accrued_reward)), dtype=torch.float)
                with torch.no_grad():
                    action = policy(aug_obs, accrued_reward)
                next_raw_obs, reward, terminated, truncated, _ = self.env.step(action)
                next_obs = self.format_obs(next_raw_obs)
                accrued_reward += (self.gamma ** timestep) * reward
                obs = next_obs
                timestep += 1

            pareto_point += accrued_reward

        return pareto_point / eval_episodes

    def train(self):
        """Train the algorithm on the given environment."""
        raise NotImplementedError

    def get_episode_stats(self, global_step):
        """Get the episode statistics."""
        curr_exp_ret = np.mean(self.episodic_returns, axis=0)
        with torch.no_grad():
            utility = self.u_func(torch.tensor(curr_exp_ret, dtype=torch.float))
        episodic_length = np.mean(self.episodic_lengths)
        return {
            f'charts/utility_{self.iteration}': utility,
            f'charts/episodic_length_{self.iteration}': episodic_length,
            f'global_step_{self.iteration}': global_step,
        }

    def save_episode_stats(self, episodic_return, episodic_length):
        """Save the episodic statistics for a single environment."""
        self.episodic_returns.append(episodic_return)
        self.episodic_lengths.append(episodic_length)

    def save_vectorized_episodic_stats(self, info, dones):
        """Save the episodic statistics for vectorized environments."""
        for k, v in info.items():
            if k == "episode":
                episodic_returns = v["r"]
                episodic_lengths = v["l"]
                for episodic_return, episodic_length, done in zip(episodic_returns, episodic_lengths, dones):
                    if done:
                        self.save_episode_stats(episodic_return, episodic_length)

    def log_pg(self, global_step, loss, pg_l, v_l, e_l):
        """Log the loss and episode statistics for PPO and A2C."""""
        log_dict = {
            f'losses/loss_{self.iteration}': loss,
            f'losses/policy_gradient_loss_{self.iteration}': pg_l,
            f'losses/value_loss_{self.iteration}': v_l,
            f'losses/entropy_loss_{self.iteration}': e_l,
            **self.get_episode_stats(global_step),
        }
        self.log_wandb(log_dict)

    def log_dqn(self, global_step, loss):
        """Log the loss and episode statistics for DQN."""
        log_dict = {
            f'losses/loss_{self.iteration}': loss,
            **self.get_episode_stats(global_step),
        }
        self.log_wandb(log_dict)

    def log_wandb(self, log_dict):
        """Log a dictionary to wandb."""
        if self.track:
            try:
                wandb.log(log_dict)
            except Exception as e:
                print(e)
                print(log_dict)
        else:
            print(log_dict)

    def get_closest_referent(self, referent):
        """Get the processed referent closest to the given referent.

        Args:
            referent (ndarray): The referent to get the closest processed referent for.

        Returns:
            ndarray: The closest processed referent.
        """
        referents = list(self.trained_models.keys())
        if len(referents) == 0:
            return False
        distances = np.array([np.linalg.norm(np.array(referent) - np.array(r)) for r in referents])
        return referents[np.argmin(distances)]

    def load_model(self, referent):
        """Load the model that is closest to the given referent.

        Args:
            referent (ndarray): The referent to load the model for.
        """
        closest_referent = self.get_closest_referent(referent)
        if closest_referent:
            return self.trained_models[tuple(closest_referent)]
        else:
            return None, None

    def save_models(self, referent, actor=None, critic=None):
        """Save the models for the given referent.

        Args:
            referent (ndarray): The referent to save the models for.
            actor (nn.Module, optional): The actor network to save. Defaults to None.
            critic (nn.Module, optional): The critic network to save. Defaults to None.
        """
        if actor is not None:
            actor = actor.state_dict()
        if critic is not None:
            critic = critic.state_dict()
        self.trained_models[tuple(referent)] = (actor, critic)

    def solve(self, referent, ideal):
        """Run the inner loop of the outer loop."""
        self.reset_stats()
        referent = torch.tensor(referent)
        ideal = torch.tensor(ideal)
        self.u_func = create_batched_aasf(referent, referent, ideal, aug=self.aug, scale=self.scale, backend='torch')
        self.train()
        pareto_point = self.evaluate(eval_episodes=self.eval_episodes, deterministic=True)
        self.iteration += 1
        return pareto_point
