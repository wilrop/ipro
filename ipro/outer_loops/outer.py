import os
import time
import random
import wandb
import platform
import numpy as np

from ipro.utils.pareto import extreme_prune
from ipro.utils.hypervolume import compute_hypervolume


class OuterLoop:
    def __init__(
            self,
            problem_id,
            dimensions,
            oracle,
            linear_solver,
            method="IPRO",
            ref_point=None,
            offset=1,
            tolerance=1e-1,
            max_iterations=None,
            known_pf=None,
            track=False,
            exp_name=None,
            wandb_project_name=None,
            wandb_entity=None,
            seed=None,
            extra_config=None
    ):
        self.problem_id = problem_id
        self.dim = dimensions
        self.oracle = oracle
        self.linear_solver = linear_solver
        self.method = method
        self.ref_point = ref_point
        self.offset = offset
        self.tolerance = tolerance
        self.max_iterations = max_iterations if max_iterations is not None else np.inf
        self.known_pf = known_pf

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
        self.replay_triggered = 0

        self.track = track
        self.run_id = None
        self.exp_name = exp_name
        self.wandb_project_name = wandb_project_name
        self.wandb_entity = wandb_entity

        self.seed = seed

        self.extra_config = extra_config

    def reset(self):
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
        self.replay_triggered = 0

    def finish(self, start_time, iteration):
        """Finish the algorithm."""
        self.pf = extreme_prune(np.vstack((self.pf, self.robust_points)))
        self.dominated_hv = compute_hypervolume(-self.pf, -self.nadir)
        self.hv = compute_hypervolume(-self.pf, -self.ref_point)
        self.log_iteration(iteration + 1)

        end_str = f'Iterations {iteration + 1} | Time {time.time() - start_time:.2f} | '
        end_str += f'HV {self.hv:.2f} | PF size {len(self.pf)} |'
        print(end_str)

        self.close_wandb()

    def config(self):
        """Get the config of the algorithm."""
        extra_config = self.extra_config if self.extra_config is not None else {}
        return {
            "method": self.method,
            "problem_id": self.problem_id,
            "dimensions": self.dim,
            "tolerance": self.tolerance,
            "max_iterations": self.max_iterations,
            "seed": self.seed,
            **extra_config
        }

    def setup(self, mode='online'):
        """Setup wandb."""
        config = self.config()
        config.update(self.oracle.config())

        print(f'Running with config: {config}')

        if self.track:
            location = platform.platform()

            if location.startswith('Linux-6.6.22-frehi12'):  # Hack to check where the code is running.
                location = 'ailab'
            elif location.startswith('macOS'):
                location = 'mac'
            else:
                location = 'vub'

            if location == 'vub':
                wandb.init(
                    settings=wandb.Settings(log_internal=str('/scratch/brussel/103/vsc10340/wandb/null'), ),
                    project=self.wandb_project_name,
                    entity=self.wandb_entity,
                    config=config,
                    name=self.exp_name,
                    mode=mode,
                )
            else:
                wandb.init(
                    project=self.wandb_project_name,
                    entity=self.wandb_entity,
                    config=config,
                    name=self.exp_name,
                    mode=mode,
                )

            wandb.define_metric('iteration')
            wandb.define_metric('outer/hypervolume', step_metric='iteration')
            wandb.define_metric('outer/dominated_hv', step_metric='iteration')
            wandb.define_metric('outer/discarded_hv', step_metric='iteration')
            wandb.define_metric('outer/coverage', step_metric='iteration')
            wandb.define_metric('outer/error', step_metric='iteration')
            self.run_id = wandb.run.id

        return time.time()

    def close_wandb(self):
        """Close wandb."""
        if self.track:
            pf_table = wandb.Table(data=self.pf, columns=[f'obj_{i}' for i in range(self.dim)])
            wandb.run.log({'pareto_front': pf_table})
            wandb.run.summary['PF_size'] = len(self.pf)
            wandb.finish()

    def log_iteration(self, iteration, referent=None, ideal=None, pareto_point=None):
        """Log the iteration."""
        if self.track:
            while True:
                try:
                    wandb.log({
                        'outer/hypervolume': self.hv,
                        'outer/dominated_hv': self.dominated_hv,
                        'outer/discarded_hv': self.discarded_hv,
                        'outer/coverage': self.coverage,
                        'outer/error': self.error,
                        'iteration': iteration
                    })
                    break
                except wandb.Error as e:
                    print(f"wandb got error {e}")
                    time.sleep(random.randint(10, 100))

            if referent is not None:
                wandb.run.summary[f"referent_{iteration}"] = referent
                wandb.run.summary[f"ideal_{iteration}"] = ideal
                wandb.run.summary[f"pareto_point_{iteration}"] = pareto_point

            wandb.run.summary['hypervolume'] = self.hv
            wandb.run.summary['PF_size'] = len(self.pf)
            wandb.run.summary['replay_triggered'] = self.replay_triggered
