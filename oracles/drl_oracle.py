import torch
import wandb
import numpy as np

from gymnasium.spaces import Discrete
from collections import deque
import torch.nn as nn

from oracles.vector_u import create_batched_aasf


class DRLOracle:
    """The base class for deep reinforcement learning oracles that execute independent learning."""

    def __init__(self,
                 env,
                 aug=0.2,
                 scale=100,
                 gamma=0.99,
                 warm_start=False,
                 vary_nadir=False,
                 vary_ideal=False,
                 eval_episodes=100,
                 deterministic_eval=True,
                 window_size=100,
                 track=False,
                 seed=0):
        self.seed = seed
        self.np_rng = np.random.default_rng(seed=seed)
        self.torch_rng = torch.Generator()
        self.torch_rng.manual_seed(seed)

        self.env = env
        self.aug = aug
        self.scale = scale

        self.num_actions = env.action_space.n
        self.num_objectives = env.reward_space.shape[0]
        self.flat_obs_dim = np.prod(self.env.observation_space.shape)
        self.aug_obs_dim = self.flat_obs_dim + self.num_objectives

        self.gamma = gamma
        self.eval_episodes = eval_episodes
        self.deterministic_eval = deterministic_eval
        self.u_func = None
        self.trained_models = {}  # Collection of trained models that can be used for warm-starting.

        self.nadir = None
        self.ideal = None
        self.vary_nadir = vary_nadir
        self.vary_ideal = vary_ideal

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

    def init_oracle(self, nadir=None, ideal=None):
        """Initialise the oracle."""
        self.nadir = nadir
        self.ideal = ideal

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

    def select_greedy_action(self, aug_obs, accrued_reward, *args, **kwargs):
        """Select the greedy action for the given observation."""
        raise NotImplementedError

    def select_action(self, aug_obs, accrued_reward, *args, **kwargs):
        """Select an action for the given observation."""
        raise NotImplementedError

    def setup_chart_metrics(self):
        """Set up the chart metrics for logging."""
        if self.track:
            wandb.define_metric(f'charts/utility_{self.iteration}', step_metric=f'global_step_{self.iteration}')
            wandb.define_metric(f'charts/episodic_length_{self.iteration}', step_metric=f'global_step_{self.iteration}')

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

    def evaluate(self, eval_episodes=100, deterministic=True):
        """Evaluate the agent on the environment.

        Args:
            eval_episodes (int, optional): The number of episodes to evaluate the agent for. Defaults to 100.
            deterministic (bool, optional): Whether to use a deterministic policy or not. Defaults to True.

        Returns:
            ndarray: The average reward over the evaluation episodes.
        """
        if deterministic:
            policy = self.select_greedy_action
        else:
            policy = self.select_action

        pareto_point = np.zeros(self.num_objectives)

        for episode in range(eval_episodes):
            obs, _ = self.env.reset()
            terminated = False
            truncated = False
            accrued_reward = np.zeros(self.num_objectives)
            timestep = 0

            while not (terminated or truncated):
                aug_obs = torch.tensor(np.concatenate((obs, accrued_reward)), dtype=torch.float)
                with torch.no_grad():
                    action = policy(aug_obs, accrued_reward)
                next_obs, reward, terminated, truncated, _ = self.env.step(action)
                accrued_reward += (self.gamma ** timestep) * reward
                obs = next_obs
                timestep += 1

            pareto_point += accrued_reward

        return pareto_point / eval_episodes

    def train(self, *args, **kwargs):
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

    def save_episode_stats(self, episodic_return, episodic_length, *args, **kwargs):
        """Save the episodic statistics for a single environment."""
        self.episodic_returns.append(episodic_return)
        self.episodic_lengths.append(episodic_length)

    def save_vectorized_episodic_stats(self, info, dones, *args, **kwargs):
        """Save the episodic statistics for vectorized environments."""
        for k, v in info.items():
            if k == "episode":
                episodic_returns = v["r"]
                episodic_lengths = v["l"]
                for episodic_return, episodic_length, done in zip(episodic_returns, episodic_lengths, dones):
                    if done:
                        self.save_episode_stats(episodic_return, episodic_length, *args, **kwargs)

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

    def solve(self, referent, nadir=None, ideal=None):
        """Run the inner loop of the outer loop."""
        self.reset_stats()

        # Determine boundaries of the utility function.
        nadir = nadir if nadir is not None and self.vary_nadir else self.nadir
        ideal = ideal if ideal is not None and self.vary_ideal else self.ideal

        # Make vectors tensors.
        referent = torch.tensor(referent, dtype=torch.float32)
        nadir = torch.tensor(nadir, dtype=torch.float32)
        ideal = torch.tensor(ideal, dtype=torch.float32)

        self.u_func = create_batched_aasf(referent, nadir, ideal, aug=self.aug, scale=self.scale, backend='torch')
        self.train()
        pareto_point = self.evaluate(eval_episodes=self.eval_episodes, deterministic=self.deterministic_eval)
        self.iteration += 1
        return pareto_point
