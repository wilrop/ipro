import time

import numpy as np
import pygmo as pg
from outer_loops.box import Box
from utils.pareto import strict_pareto_dominates, extreme_prune


class Priol:
    """An inner-outer loop method for solving multi-objective problems."""

    def __init__(self,
                 problem,
                 dimensions,
                 oracle,
                 linear_solver,
                 warm_start=False,
                 tolerance=1e-1,
                 max_steps=5000,
                 rng=None,
                 seed=None,
                 approx=False,
                 ref_offset=0.1,
                 hv_eps=0.1,
                 hv_delta=0.1,
                 save_figs=False,
                 log_dir=None
                 ):
        self.problem = problem
        self.dim = dimensions

        self.oracle = oracle
        self.linear_solver = linear_solver

        self.approx = approx
        self.ref_offset = ref_offset
        self.approx_hv = pg.bf_fpras(eps=hv_eps, delta=hv_delta, seed=seed)  # Polynomial time approx hypervolume.

        self.warm_start = warm_start
        self.tol = tolerance
        self.max_steps = max_steps

        self.bounding_box = None
        self.total_hv = 0
        self.nadir = None
        self.ideal = None
        self.pf = []
        self.robust_points = np.empty((0, self.dim))
        self.completed = np.empty((0, self.dim))
        self.lower_points = []
        self.upper_points = []
        self.dominated_hv = 0
        self.dominating_hv = 0
        self.error_estimates = []
        self.covered_volume = 0

        self.save_figs = save_figs
        self.log_dir = log_dir

        self.seed = seed
        self.rng = rng if rng is not None else np.random.default_rng(seed)

    def reset(self):
        """Reset the algorithm to its initial state."""
        self.bounding_box = None
        self.total_hv = 0
        self.nadir = None
        self.ideal = None
        self.pf = []
        self.robust_points = np.empty((0, self.dim))
        self.completed = np.empty((0, self.dim))
        self.lower_points = []
        self.upper_points = []
        self.dominated_hv = 0
        self.dominating_hv = 0
        self.error_estimates = []
        self.covered_volume = 0

    def config(self):
        """Get the configuration of the algorithm."""
        return {
            'dimensions': self.dim,
            'tolerance': self.tol,
            'max_steps': self.max_steps,
            'seed': self.seed,
            'approx': self.approx,
            'hv_eps': self.approx_hv.eps,
            'hv_delta': self.approx_hv.delta,
            'save_figs': self.save_figs,
            'log_dir': self.log_dir
        }

    def init_phase(self):
        """Run the initialisation phase of the algorithm.

        This phase computes the bounding box of the Pareto front by solving the maximisation and minimisation problems
        for all objectives. For the ideal, this is exact and for the nadir this is guaranteed to be a pessimistic
        estimate. The lower points are then computed and initial hypervolume improvements.

        Returns:
            bool: True if the Pareto front is the ideal point, False otherwise.
        """
        nadir = np.zeros(self.dim)
        ideal = np.zeros(self.dim)
        pf = []

        for i in range(self.dim):
            weight_vec = np.zeros(self.dim)
            weight_vec[i] = 1
            max_i = self.linear_solver.solve(weight_vec)
            min_i = self.linear_solver.solve(-1 * weight_vec)
            nadir[i] = min_i[i]
            ideal[i] = max_i[i]
            pf.append(max_i)

        self.pf = extreme_prune(np.array(pf))
        if len(self.pf) == 1:  # If the Pareto front is the ideal.
            return True

        nadir = nadir - 1  # Necessary to ensure every Pareto optimal point strictly dominates the nadir.
        self.bounding_box = Box(nadir - self.ref_offset, ideal + self.ref_offset)
        self.nadir = np.copy(nadir)
        self.ideal = np.copy(ideal)
        self.total_hv = self.bounding_box.volume
        self.lower_points = np.array([nadir])

        for point in self.pf:  # Initialise the lower points.
            self.update_lower_points(np.array(point))

        self.upper_points = np.array([ideal])  # Initialise the upper points.
        self.error_estimates.append(max(ideal - nadir))
        self.compute_hvis()

        return False

    def compute_hypervolume(self, points, ref):
        """Compute the hypervolume of a set of points.

        Args:
            points (array_like): List of points.
            ref (np.array): Reference point.

        Returns:
            float: The computed hypervolume.
        """
        ref = ref + self.ref_offset
        if self.approx:
            return pg.hypervolume(points).compute(ref, hv_algo=self.approx_hv)
        else:
            return pg.hypervolume(points).compute(ref)

    def compute_hvis(self, num=50):
        """Compute the hypervolume improvements of the lower points.

        Note:
            An optional num parameter can be given as computing the hypervolume for a large number of potential points
            is expensive.

        Args:
            num (int, optional): The number of lower points to compute hypervolume improvements for. Defaults to 50.

        Returns:
            np.array: The hypervolume improvements of the lower points.
        """
        point_set = np.vstack((self.pf, self.completed))
        hvis = np.zeros(len(self.lower_points))

        for lower_id in self.rng.choice(len(self.lower_points), min(num, len(self.lower_points)), replace=False):
            hv = self.compute_hypervolume(np.vstack((point_set, self.lower_points[lower_id])), self.ideal)
            hvis[lower_id] = hv  # We don't have to compute the difference as it is proportional to the hypervolume.

        sorted_args = np.argsort(hvis)[::-1]
        self.lower_points = self.lower_points[sorted_args]

    def max_hypervolume_improvement(self):
        """Recompute the hypervolume improvements and return the point that maximises it..

        Returns:
            np.array: The point that maximises the hypervolume improvement.
        """
        self.compute_hvis()
        return self.lower_points[0]

    def estimate_error(self):
        """Estimate the error of the algorithm."""
        if len(self.upper_points) == 0:
            error = 0
        else:
            pf = np.array(list(self.pf))
            diffs = self.upper_points[:, None, :] - pf[None, :, :]
            error = np.max(np.min(np.max(diffs, axis=2), axis=1))
        self.error_estimates.append(error)

    def batched_strict_pareto_dominates(self, vec, points):
        """Check if a vector strictly dominates a set of points.

        Args:
            vec (np.array): The vector.
            points (np.array): The set of points.

        Returns:
            np.array: A boolean array indicating whether each point is dominated.
        """
        return np.all(vec > points, axis=-1)

    def update_upper_points(self, vec):
        """Update the upper set.

        Args:
            vec (np.array): The point that is added to the boundary of the dominating space.
        """
        strict_dominates = self.batched_strict_pareto_dominates(self.upper_points, vec)
        to_keep = self.upper_points[strict_dominates == 0]
        shifted = np.stack([self.upper_points[strict_dominates == 1]] * self.dim)
        shifted[range(self.dim), :, range(self.dim)] = np.expand_dims(vec, -1)
        shifted = shifted.reshape(-1, self.dim)
        shifted = shifted[np.all(shifted > self.nadir, axis=-1)]

        new_upper_points = np.vstack((to_keep, shifted))
        self.upper_points = extreme_prune(new_upper_points)

    def update_lower_points(self, vec):
        """Update the upper set.

        Args:
            vec (np.array): The point that is added to the boundary of the dominated space.
        """
        strict_dominates = self.batched_strict_pareto_dominates(vec, self.lower_points)
        to_keep = self.lower_points[strict_dominates == 0]
        shifted = np.stack([self.lower_points[strict_dominates == 1]] * self.dim)
        shifted[range(self.dim), :, range(self.dim)] = np.expand_dims(vec, -1)
        shifted = shifted.reshape(-1, self.dim)
        shifted = shifted[np.all(self.ideal > shifted, axis=-1)]

        new_lower_points = np.vstack((to_keep, shifted))
        self.lower_points = -extreme_prune(-new_lower_points)

    def update_dominating_hv(self):
        """Update the hypervolume of the dominating space."""
        self.dominating_hv = self.compute_hypervolume(np.vstack((self.pf, self.completed)), self.ideal)

    def update_dominated_hv(self):
        """Update the hypervolume of the dominated space."""
        self.dominated_hv = self.compute_hypervolume(-self.pf, -self.nadir)

    def select_referent(self, method='random'):
        if method == 'random':
            return self.lower_points[self.rng.integers(0, len(self.lower_points))]
        if method == 'first':
            return self.lower_points[0]
        elif method == 'max_hvi':
            return self.compute_hvis()
        else:
            raise ValueError(f'Unknown method {method}')

    def solve(self, log_freq=1, update_freq=50):
        """Solve the problem.

        Args:
            log_freq (int, optional): The frequency of logging the progress of the algorithm. Defaults to 10.

        Returns:
            set: The Pareto front.
        """
        start = time.time()
        done = self.init_phase()
        step = 0

        if done:
            print('The problem is solved in the initial phase.')
            print(self.pf)
            return {tuple(vec) for vec in self.pf}

        while self.error_estimates[-1] > self.tol and step < self.max_steps:
            if step % log_freq == 0:
                print(f'Step {step}')
                print(f'↪ Covered volume: {self.covered_volume * 100:.5f}%')
                print(f'↪ Error estimate: {self.error_estimates[-1]:.5f}')

            referent = self.select_referent(method='first')
            vec = self.oracle.solve(np.copy(referent), np.copy(self.ideal), warm_start=self.warm_start)
            print(f'Referent {referent} -> Vec {vec}')

            if strict_pareto_dominates(vec, referent):
                self.pf = np.vstack((self.pf, vec))
                self.update_lower_points(vec)
                self.update_upper_points(vec)
            else:
                self.completed = np.vstack((self.completed, referent))
                self.lower_points = self.lower_points[np.any(self.lower_points != referent, axis=1)]
                self.update_upper_points(referent)
                self.robust_points = np.vstack((self.robust_points, vec))

            if step % update_freq == 0:
                self.compute_hvis()
                self.update_dominated_hv()
                self.update_dominating_hv()
                self.covered_volume = (self.dominated_hv + self.dominating_hv) / self.total_hv

            self.estimate_error()
            step += 1

        pf = {tuple(vec) for vec in extreme_prune(np.vstack((self.pf, self.robust_points)))}

        print(f'Algorithm finished in {time.time() - start:.2f} seconds.')
        return pf
