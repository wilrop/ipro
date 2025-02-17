import wandb
import torch
import numpy as np

from collections import deque

from ipro.oracles.drl_oracle import DRLOracle
from ipro.oracles.vector_u import aasf


class SNDRLOracle(DRLOracle):
    """The base class for a single-network deep reinforcement learning oracle."""

    def __init__(self,
                 env,
                 gamma=0.99,
                 aug=0.1,
                 scale=100,
                 vary_nadir=False,
                 vary_ideal=False,
                 pretrain_iters=100,
                 grid_sample=False,
                 num_referents=16,
                 pretraining_steps=100000,
                 online_steps=10000,
                 eval_episodes=100,
                 deterministic_eval=True,
                 track=False,
                 seed=0,
                 alg_name=None):
        super().__init__(env,
                         gamma=gamma,
                         aug=aug,
                         scale=scale,
                         warm_start=False,
                         vary_nadir=vary_nadir,
                         vary_ideal=vary_ideal,
                         eval_episodes=eval_episodes,
                         deterministic_eval=deterministic_eval,
                         track=track,
                         seed=seed,
                         alg_name=alg_name)
        self.pretrained_model = None  # The pretrained model.
        self.phase = 'pretrain'  # The phase of the algorithm. Either 'pre' or 'online_{iteration}'.

        self.pretrain_iters = int(pretrain_iters)  # The number of iterations to pretrain for.
        self.grid_sample = grid_sample  # Whether to sample a grid of referents or uniformly sample.
        self.num_referents = int(num_referents)  # The number of referents to train on.
        self.pretraining_steps = int(pretraining_steps)  # The number of training steps.
        self.online_steps = int(online_steps)

        self.episodic_utility = deque(maxlen=eval_episodes)  # Episodic utility of the agent rather than the returns.

    def config(self):
        """Get the config of the algorithm."""
        config = super().config()
        config.update({
            'pretrain_iters': self.pretrain_iters,
            'grid_sample': self.grid_sample,
            'num_referents': self.num_referents,
            'pretraining_steps': self.pretraining_steps,
            'online_steps': self.online_steps,
        })
        return config

    # noinspection PyMethodOverriding
    def select_greedy_action(self, aug_obs, accrued_reward, referent, nadir, ideal):
        """Select the greedy action for the given observation."""
        raise NotImplementedError

    # noinspection PyMethodOverriding
    def select_action(self, aug_obs, accrued_reward, referent, nadir, ideal):
        """Select an action for the given observation."""
        raise NotImplementedError

    def save_model(self):
        """Save the models."""
        raise NotImplementedError

    def pretrain(self):
        """Pretrain the algorithm."""
        raise NotImplementedError

    # noinspection PyMethodOverriding
    def train(self, referent, nadir, ideal, *args, **kwargs):
        """Train the algorithm on the given environment."""
        raise NotImplementedError

    def grid_sample_referents(self, num_referents, nadir, ideal):
        """Sample a grid of evenly space points from a box described by nadir and ideal."""
        atoms_per_dim = int(num_referents ** (1 / self.num_objectives))
        dim_atoms = [atoms_per_dim] * self.num_objectives
        linspaces = []
        for d in range(self.num_objectives):
            linspace = torch.linspace(nadir[d], ideal[d], dim_atoms[d] + 1)[:-1]
            linspaces.append(linspace)
        mesh = torch.meshgrid(*linspaces)
        coordinates = torch.stack(mesh, dim=-1)
        grid = coordinates.view(-1, self.num_objectives)
        return grid

    def uniform_sample_referents(self, num_referents, nadir, ideal):
        """Sample a batch of referents uniformly.

        Args:
            num_referents (int): The number of referents to sample.
            nadir (torch.Tensor): The nadir point.
            ideal (torch.Tensor): The ideal point.

        Returns:
            torch.Tensor: The sampled referents.
        """
        referents = torch.rand(size=(num_referents, self.num_objectives),
                               dtype=torch.float,
                               generator=self.torch_rng) * (nadir - ideal) + ideal
        return referents

    def sample_referents(self, num_referents, nadir, ideal):
        """Sample referents between the given nadir and ideal points.

        Args:
            num_referents (int): The number of referents to sample.
            nadir (torch.Tensor): The nadir point.
            ideal (torch.Tensor): The ideal point.

        Returns:
            torch.Tensor: The sampled referents.
        """
        if self.grid_sample:
            return self.grid_sample_referents(num_referents, nadir, ideal)
        else:
            return self.uniform_sample_referents(num_referents, nadir, ideal)

    def init_oracle(self, nadir=None, ideal=None):
        """Initialise the oracle."""
        super().init_oracle(nadir, ideal)
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
        utility = np.mean(list(self.episodic_utility))
        episodic_length = np.mean(list(self.episodic_lengths))
        return {
            f'charts/{self.phase}_utility': utility,
            f'charts/{self.phase}_episodic_length': episodic_length,
            f'{self.phase}_step': step,
        }

    # noinspection PyMethodOverriding
    def save_episode_stats(self, episodic_return, episodic_length, referent, nadir, ideal, *args, **kwargs):
        """Save the episodic statistics for a single environment."""
        episodic_return = torch.tensor(episodic_return, dtype=torch.float32, requires_grad=False)
        episodic_utility = aasf(episodic_return,
                                referent,
                                nadir,
                                ideal,
                                aug=self.aug,
                                scale=self.scale,
                                backend='torch')
        self.episodic_utility.append(episodic_utility)
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

    def run_inner_loop(self, referent, nadir=None, ideal=None, *args, **kwargs):
        """Run the inner loop of the outer loop."""
        self.reset_stats()
        self.phase = f'online_{self.iteration}'

        referent, nadir, ideal = self.get_asf_params(referent, nadir, ideal, backend='torch')
        self.load_model(None)  # We always load the pretrained model.
        self.train(
            referent,
            nadir,
            ideal,
            num_referents=1,  # Only train on the given referent.
            *args,
            **kwargs
        )
        pareto_point = self.evaluate(
            eval_episodes=self.eval_episodes,
            deterministic=self.deterministic_eval,
            referent=referent,
            nadir=nadir,
            ideal=ideal
        )
        self.iteration += 1
        return pareto_point
