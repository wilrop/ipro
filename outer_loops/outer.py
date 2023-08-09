import wandb
import numpy as np
import pygmo as pg


class OuterLoop:
    def __init__(self,
                 problem,
                 dimensions,
                 oracle,
                 linear_solver,
                 method="priol",
                 ref_point=None,
                 offset=1,
                 tolerance=1e-1,
                 max_steps=None,
                 warm_start=False,
                 track=False,
                 exp_name=None,
                 wandb_project_name=None,
                 wandb_entity=None,
                 seed=None):
        self.problem = problem
        self.dim = dimensions
        self.oracle = oracle
        self.linear_solver = linear_solver
        self.method = method
        self.ref_point = ref_point
        self.offset = offset
        self.tolerance = tolerance
        self.max_steps = max_steps if max_steps is not None else np.inf
        self.warm_start = warm_start

        self.bounding_box = None
        self.ideal = None
        self.nadir = None
        self.pf = np.empty((0, self.dim))
        self.robust_points = np.empty((0, self.dim))
        self.completed = np.empty((0, self.dim))

        self.hv = 0
        self.total_hv = 0
        self.dominated_hv = 0
        self.discarded_hv = 0
        self.coverage = 0
        self.error = np.inf

        self.track = track
        self.run_id = None
        self.exp_name = exp_name
        self.wandb_project_name = wandb_project_name
        self.wandb_entity = wandb_entity

        self.seed = seed

    def config(self):
        """Get the config of the algorithm."""
        return {
            "method": self.method,
            "env_id": self.problem.env_id,
            "dimensions": self.dim,
            "warm_start": self.warm_start,
            "tolerance": self.tolerance,
            "max_steps": self.max_steps,
            "seed": self.seed,
        }

    def setup_wandb(self, cluster=True):
        """Setup wandb."""
        if self.track:
            config = self.config()
            config.update(self.oracle.config())
            if cluster:
                wandb.init(
                    settings=wandb.Settings(log_internal=str('/scratch/brussel/103/vsc10340/wandb/null'), ),
                    project=self.wandb_project_name,
                    entity=self.wandb_entity,
                    config=config,
                    name=self.exp_name,
                )
            else:
                wandb.init(
                    project=self.wandb_project_name,
                    entity=self.wandb_entity,
                    config=config,
                    name=self.exp_name,
                )

            wandb.define_metric('iteration')
            wandb.define_metric('outer/hypervolume', step_metric='iteration')
            wandb.define_metric('outer/dominated_hv', step_metric='iteration')
            wandb.define_metric('outer/discarded_hv', step_metric='iteration')
            wandb.define_metric('outer/coverage', step_metric='iteration')
            wandb.define_metric('outer/error', step_metric='iteration')
            self.run_id = wandb.run.id

    def close_wandb(self):
        """Close wandb."""
        if self.track:
            wandb.finish()

    def log_iteration(self, iteration):
        """Log the iteration."""
        if self.track:
            wandb.log({
                'outer/hypervolume': self.hv,
                'outer/dominated_hv': self.dominated_hv,
                'outer/discarded_hv': self.discarded_hv,
                'outer/coverage': self.coverage,
                'outer/error': self.error,
                'iteration': iteration
            })

    def compute_hypervolume(self, points, ref):
        """Compute the hypervolume of a set of points.

        Note:
            This computes the hypervolume assuming all objectives are to be minimized.

        Args:
            points (array_like): List of points.
            ref (np.array): Reference point.

        Returns:
            float: The computed hypervolume.
        """
        return pg.hypervolume(points).compute(ref)
