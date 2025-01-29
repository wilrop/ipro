import time
import random
import wandb
import platform
import numpy as np

from typing import Optional, Iterable, Any
from ipro.outer_loops.typing import Subproblem, Subsolution, IPROCallback

from pymoo.indicators.hv import Hypervolume
from pymoo.config import Config

from ipro.oracles.oracle import Oracle
from ipro.linear_solvers.linear_solver import LinearSolver
from ipro.utils.pareto import (
    strict_pareto_dominates,
    batched_strict_pareto_dominates,
    extreme_prune,
    batched_pareto_dominates
)

Config.warnings['not_compiled'] = False


class OuterLoop:
    def __init__(
            self,
            problem_id: str,
            dimensions: int,
            oracle: Oracle,
            linear_solver: LinearSolver,
            method: str = "IPRO",
            direction: str = "maximize",  # "minimize" or "maximize
            ref_point: Optional[np.ndarray] = None,
            offset: float = 1,
            tolerance: float = 1e-1,
            max_iterations: Optional[int] = None,
            known_pf: Optional[np.ndarray] = None,
            track: bool = False,
            exp_name: Optional[str] = None,
            wandb_project_name: Optional[str] = None,
            wandb_entity: Optional[str] = None,
            seed: Optional[int] = None,
            extra_config: Optional[dict] = None
    ):
        self.problem_id = problem_id
        self.dim = dimensions
        self.oracle = oracle
        self.linear_solver = linear_solver
        self.method = method
        self.direction = direction
        self.ref_point = ref_point
        self.offset = offset
        self.tolerance = tolerance
        self.max_iterations = max_iterations if max_iterations is not None else np.inf
        self.known_pf = known_pf

        self.sign = 1 if direction == "maximize" else -1
        self.bounding_box = None
        self.ideal = None
        self.nadir = None
        self.pf = None
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

    def config(self) -> dict:
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

    def setup(self, mode: str = 'online') -> float:
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

    def get_pareto_set(self, subsolutions: list[Subsolution]) -> list[tuple[np.ndarray, Any]]:
        """Get the Pareto set from the subsolutions."""
        pareto_set = []
        for subsolution in subsolutions:
            if np.any(np.all(subsolution[1] == self.pf, axis=1)):
                pareto_set.append((subsolution[1], subsolution[2]))
        return pareto_set

    def finish(self, start_time: float, iteration: int):
        """Finish the algorithm."""
        self.pf = extreme_prune(np.vstack((self.pf, self.robust_points)))
        self.dominated_hv = self.compute_hypervolume(-self.pf, -self.nadir)
        self.hv = self.compute_hypervolume(-self.pf, -self.ref_point)
        self.log_iteration(iteration + 1)

        end_str = f'Iterations {iteration + 1} | Time {time.time() - start_time:.2f} | '
        end_str += f'HV {self.hv:.2f} | PF size {len(self.pf)} |'
        print(end_str)

        self.close_wandb()

    def close_wandb(self):
        """Close wandb."""
        if self.track:
            pf_table = wandb.Table(data=self.pf, columns=[f'obj_{i}' for i in range(self.dim)])
            wandb.run.log({'pareto_front': pf_table})
            wandb.run.summary['PF_size'] = len(self.pf)
            wandb.finish()

    def log_iteration(
            self,
            iteration: int,
            subproblem: Optional[Subproblem] = None,
            pareto_point: Optional[np.ndarray] = None
    ):
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

            if subproblem is not None:
                wandb.run.summary[f"referent_{iteration}"] = self.sign * subproblem.referent
                wandb.run.summary[f"ideal_{iteration}"] = self.sign * subproblem.ideal
                wandb.run.summary[f"pareto_point_{iteration}"] = self.sign * pareto_point

            wandb.run.summary['hypervolume'] = self.hv
            wandb.run.summary['PF_size'] = len(self.pf)
            wandb.run.summary['replay_triggered'] = self.replay_triggered

    def compute_hypervolume(self, points: np.ndarray, ref: np.ndarray) -> float:
        """Compute the hypervolume of a set of points.

        Note:
            This computes the hypervolume assuming all objectives are to be minimized.

        Args:
            points (array_like): List of points.
            ref (np.array): Reference point.

        Returns:
            float: The computed hypervolume.
        """
        points = points[batched_pareto_dominates(ref, points)]
        if points.size == 0:
            return 0
        ind = Hypervolume(ref_point=ref)
        return ind(points)

    def init_phase(self) -> tuple[list[Subsolution], bool]:
        """Initialize the outer loop."""
        raise NotImplementedError

    def is_done(self, step: int) -> bool:
        """Check if the algorithm is done."""
        return 1 - self.coverage <= self.tolerance or step >= self.max_iterations

    def decompose_problem(self, iteration: int, method: str = 'first') -> Subproblem:
        """Decompose the problem into a subproblem."""
        raise NotImplementedError

    def update_found(self, subproblem: Subproblem, vec: np.ndarray):
        """The update that is called when a Pareto optimal solution is found."""
        raise NotImplementedError

    def update_not_found(self, subproblem: Subproblem, vec: np.ndarray):
        """The update that is called when no Pareto optimal solution is found."""
        raise NotImplementedError

    def update_excluded_volume(self):
        """Update the dominated and infeasible sets."""
        raise NotImplementedError

    def estimate_error(self):
        """Estimate the error of the algorithm."""
        raise NotImplementedError

    def get_iterable_for_replay(self) -> Iterable[Any]:
        raise NotImplementedError

    def maybe_add_solution(
            self,
            subproblem: Subproblem,
            vec: np.ndarray,
            item: Any,
    ) -> Subproblem | bool:
        raise NotImplementedError

    def maybe_add_completed(
            self,
            subproblem: Subproblem,
            vec: np.ndarray,
            item: Any,
    ) -> Subproblem | bool:
        raise NotImplementedError

    def replay(
            self,
            vec: np.ndarray,
            sol: Any,
            iter_pairs: list[Subsolution]
    ) -> tuple[list[Subsolution], list[Subsolution]]:
        """Replay the algorithm while accounting for the non-optimal Pareto oracle.

        Note:
            This reexecutes the initialisation phase which may trigger expensive compute again. However, we always use
            a given box in the experiments, so this makes no difference **in this specific case**.

        Args:
            vec (ndarray): The vector that causes the conflict.
            iter_pairs (list[Subsolution]): A list of subsolutions.

        Returns:
            An updated list of referent point tuples.
        """
        print('REPLAY TRIGGERED')
        replay_triggered = self.replay_triggered
        self.reset()
        self.replay_triggered = replay_triggered + 1
        new_init_subsolutions, _ = self.init_phase()
        idx = 0
        new_subsolutions = []

        for old_subproblem, old_vec, old_sol in iter_pairs:  # Replay the points that were added correctly
            idx += 1
            if strict_pareto_dominates(old_vec, old_subproblem.referent):
                if strict_pareto_dominates(vec, old_vec):
                    self.update_found(old_subproblem, vec)
                    new_subsolutions.append((old_subproblem, vec, sol))
                    break
                else:
                    self.update_found(old_subproblem, old_vec)
                    new_subsolutions.append((old_subproblem, old_vec, old_sol))
            else:
                if strict_pareto_dominates(vec, old_subproblem.referent):
                    self.update_found(old_subproblem, vec)
                    new_subsolutions.append((old_subproblem, vec, sol))
                    break
                else:
                    self.update_not_found(old_subproblem, old_vec)
                    new_subsolutions.append((old_subproblem, old_vec, old_vec))

        for old_subproblem, old_vec, old_sol in iter_pairs[idx:]:  # Process the remaining points to see if we can still add them.
            items = self.get_iterable_for_replay()
            if strict_pareto_dominates(old_vec, old_subproblem.referent):
                maybe_add = self.maybe_add_solution
            else:
                maybe_add = self.maybe_add_completed
            for item in items:
                res = maybe_add(old_subproblem, old_vec, item)
                if res:
                    new_subsolutions.append((res, old_vec, old_sol))
                    break

        return new_init_subsolutions, new_subsolutions

    def solve(self, callback: Optional[IPROCallback] = None) -> list[tuple[np.ndarray, Any]]:
        """Solve the problem."""
        start = self.setup()
        linear_subsolutions, done = self.init_phase()
        iteration = 0

        if done:
            print('The problem is solved in the initial phase.')
            pareto_set = self.get_pareto_set(linear_subsolutions)
            return pareto_set

        self.log_iteration(iteration)
        subsolutions = []

        while not self.is_done(iteration):
            begin_loop = time.time()
            print(f'Iter {iteration} - Covered {self.coverage:.5f}% - Error {self.error:.5f}')

            subproblem = self.decompose_problem(iteration)
            vec, sol = self.oracle.solve(
                self.sign * subproblem.referent,
                nadir=self.sign * subproblem.nadir,
                ideal=self.sign * subproblem.ideal
            )
            vec *= self.sign

            if strict_pareto_dominates(vec, subproblem.referent):
                if np.any(batched_strict_pareto_dominates(vec, np.vstack((self.pf, self.completed)))):
                    linear_subsolutions, subsolutions = self.replay(vec, sol, subsolutions)
                else:
                    self.update_found(subproblem, vec)
                    subsolutions.append((subproblem, vec, sol))
            else:
                if np.any(batched_strict_pareto_dominates(vec, self.completed)):
                    linear_subsolutions, subsolutions = self.replay(vec, sol, subsolutions)
                else:
                    self.update_not_found(subproblem, vec)
                    subsolutions.append((subproblem, vec, sol))

            self.update_excluded_volume()
            self.estimate_error()
            self.coverage = (self.dominated_hv + self.discarded_hv) / self.total_hv
            self.hv = self.compute_hypervolume(-self.sign * self.pf, -self.sign * self.ref_point)

            iteration += 1
            self.log_iteration(iteration, subproblem=subproblem, pareto_point=vec)

            if callback is not None:
                callback(iteration, self.hv, self.dominated_hv, self.discarded_hv, self.coverage, self.error)

            duration = time.time() - begin_loop
            print(f'Ref {self.sign * subproblem.referent} - Found {self.sign * vec} - Time {duration:.2f}s')
            print('---------------------')

        self.finish(start, iteration)
        pareto_set = self.get_pareto_set(linear_subsolutions + subsolutions)
        return pareto_set
