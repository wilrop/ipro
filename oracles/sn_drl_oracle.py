import wandb
import torch
import numpy as np

from collections import deque

from oracles.drl_oracle import DRLOracle
from oracles.vector_u import aasf


class SNDRLOracle(DRLOracle):
    """The base class for a single-network deep reinforcement learning oracle."""

    def __init__(self,
                 env,
                 aug=0.2,
                 scale=100,
                 gamma=0.99,
                 vary_nadir=False,
                 vary_ideal=False,
                 pretrain_iters=100,
                 num_referents=16,
                 pre_learning_start=1000,
                 pre_epsilon_start=1.0,
                 pre_epsilon_end=0.01,
                 pre_exploration_frac=0.5,
                 pretraining_steps=100000,
                 online_steps=10000,
                 eval_episodes=100,
                 deterministic_eval=True,
                 window_size=100,
                 track=False,
                 seed=0):
        super().__init__(env,
                         aug=aug,
                         scale=scale,
                         gamma=gamma,
                         warm_start=False,
                         vary_nadir=vary_nadir,
                         vary_ideal=vary_ideal,
                         eval_episodes=eval_episodes,
                         deterministic_eval=deterministic_eval,
                         window_size=window_size,
                         track=track,
                         seed=seed)
        self.pretrained_model = None  # The pretrained model.
        self.phase = 'pretrain'  # The phase of the algorithm. Either 'pre' or 'online_{iteration}'.

        self.pretrain_iters = pretrain_iters  # The number of iterations to pretrain for.
        self.num_referents = num_referents  # The number of referents to train on.
        self.pre_learning_start = pre_learning_start  # The number of steps to wait before training.
        self.pre_epsilon_start = pre_epsilon_start  # The initial epsilon value.
        self.pre_epsilon_end = pre_epsilon_end  # The final epsilon value.
        self.pre_exploration_frac = pre_exploration_frac  # The fraction of the training steps to explore.
        self.pretraining_steps = pretraining_steps  # The number of training steps.
        self.online_steps = online_steps

        self.episodic_utility = deque(maxlen=window_size)  # The episodic utility of the agent rather than the returns.
        self.episodic_lengths = deque(maxlen=window_size)

    def config(self):
        """Get the config of the algorithm."""
        return {
            'pretrain_iters': self.pretrain_iters,
            'num_referents': self.num_referents,
            'pre_learning_start': self.pre_learning_start,
            'pre_epsilon_start': self.pre_epsilon_start,
            'pre_epsilon_end': self.pre_epsilon_end,
            'pre_exploration_frac': self.pre_exploration_frac,
            'pretraining_steps': self.pretraining_steps,
            'online_steps': self.online_steps,
        }

    # noinspection PyMethodOverriding
    def select_greedy_action(self, aug_obs, accrued_reward, referent, nadir, ideal):
        """Select the greedy action for the given observation."""
        raise NotImplementedError

    # noinspection PyMethodOverriding
    def select_action(self, aug_obs, accrued_reward, referent, nadir, ideal):
        """Select an action for the given observation."""
        raise NotImplementedError

    # noinspection PyMethodOverriding
    def train(self, referent, nadir, ideal, *args, **kwargs):
        """Train the algorithm on the given environment."""
        raise NotImplementedError

    def pretrain(self):
        """Pretrain the algorithm."""
        self.reset()
        self.setup_dqn_metrics()
        referents = torch.rand(size=(self.pretrain_iters, self.num_objectives),
                               dtype=torch.float,
                               generator=self.torch_rng) * (self.nadir - self.ideal) + self.ideal
        for idx, referent in enumerate(referents):
            self.train(referent,
                       self.nadir,
                       self.ideal,
                       self.pretraining_steps,
                       self.pre_learning_start if idx == 0 else 0,  # Only fill the buffer on the first iteration.
                       self.pre_epsilon_start,
                       self.pre_epsilon_end,
                       self.pre_exploration_frac,
                       self.num_referents)

    def init_oracle(self, nadir=None, ideal=None):
        """Initialise the oracle."""
        self.nadir = torch.tensor(nadir, dtype=torch.float)
        self.ideal = torch.tensor(ideal, dtype=torch.float)
        self.pretrain()

    def setup_chart_metrics(self):
        """Set up the chart metrics for logging."""
        if self.track:
            wandb.define_metric(f'charts/{self.phase}_utility', step_metric=f'{self.phase}_step')
            wandb.define_metric(f'charts/{self.phase}_episodic_length', step_metric=f'{self.phase}_step')

    def setup_dqn_metrics(self):
        """Set up the metrics for logging."""
        if self.track:
            wandb.define_metric(f'{self.phase}_step')
            wandb.define_metric(f'losses/{self.phase}_loss', step_metric=f'{self.phase}_step')
            self.setup_chart_metrics()

    def setup_ac_metrics(self):
        """Set up the metrics for logging."""
        if self.track:
            wandb.define_metric(f'{self.phase}_step')
            wandb.define_metric(f'losses/{self.phase}_loss', step_metric=f'{self.phase}_step')
            wandb.define_metric(f'losses/{self.phase}_pg_loss', step_metric=f'{self.phase}_step')
            wandb.define_metric(f'losses/{self.phase}_value_loss', step_metric=f'{self.phase}_step')
            wandb.define_metric(f'losses/{self.phase}_entropy_loss', step_metric=f'{self.phase}_step')
            wandb.define_metric(f'losses/{self.phase}_actor_grad_norm', step_metric=f'{self.phase}_step')
            wandb.define_metric(f'losses/{self.phase}_critic_grad_norm', step_metric=f'{self.phase}_step')
            self.setup_chart_metrics()

    def get_episode_stats(self, step):
        """Get the episode statistics."""
        utility = np.mean(self.episodic_utility)
        episodic_length = np.mean(self.episodic_lengths)
        return {
            f'charts/{self.phase}_utility': utility,
            f'charts/{self.phase}_episodic_length': episodic_length,
            f'{self.phase}_step': step,
        }

    # noinspection PyMethodOverriding
    def save_episode_stats(self, episodic_return, episodic_length, referent, nadir, ideal, *args, **kwargs):
        """Save the episodic statistics for a single environment."""
        episodic_utility = aasf(episodic_return,
                                referent,
                                nadir,
                                ideal,
                                aug=self.aug,
                                scale=self.scale,
                                backend='torch')
        self.episodic_returns.append(episodic_utility)
        self.episodic_lengths.append(episodic_length)

    def log_pg(self, step, loss, pg_l, v_l, e_l):
        """Log the loss and episode statistics for PPO and A2C."""""
        log_dict = {
            f'losses/{self.phase}_loss': loss,
            f'losses/{self.phase}_policy_gradient_loss': pg_l,
            f'losses/{self.phase}_value_loss': v_l,
            f'losses/{self.phase}_entropy_loss': e_l,
            **self.get_episode_stats(step),
        }
        self.log_wandb(log_dict)

    def log_dqn(self, step, loss):
        """Log the loss and episode statistics for DQN."""
        log_dict = {
            f'losses/{self.phase}_loss': loss,
            **self.get_episode_stats(step),
        }
        self.log_wandb(log_dict)

    def solve(self, referent, nadir=None, ideal=None, *args, **kwargs):
        """Run the inner loop of the outer loop."""
        self.reset_stats()
        self.phase = f'online_{self.iteration}'

        # Determine boundaries of the utility function.
        nadir = nadir if nadir is not None and self.vary_nadir else self.nadir
        ideal = ideal if ideal is not None and self.vary_ideal else self.ideal

        # Make vectors tensors.
        referent = torch.tensor(referent, dtype=torch.float32)
        nadir = torch.tensor(nadir, dtype=torch.float32)
        ideal = torch.tensor(ideal, dtype=torch.float32)

        self.train(referent,
                   nadir,
                   ideal,
                   num_referents=1,  # Only train on the given referent.
                   *args,
                   **kwargs)
        pareto_point = self.evaluate(eval_episodes=self.eval_episodes, deterministic=self.deterministic_eval)
        self.iteration += 1
        return pareto_point
