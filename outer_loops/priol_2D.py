import time
import wandb

import numpy as np
import pygmo as pg

from sortedcontainers import SortedKeyList

from outer_loops.outer import OuterLoop
from outer_loops.box import Box
from utils.pareto import extreme_prune, strict_pareto_dominates


class Priol2D(OuterLoop):
    """An inner-outer loop method for solving 2D multi-objective problems."""

    def __init__(self,
                 problem,
                 oracle,
                 linear_solver,
                 track=False,
                 exp_name=None,
                 wandb_project_name=None,
                 wandb_entity=None,
                 warm_start=False,
                 tolerance=1e-6,
                 seed=None):
        super().__init__(oracle,
                         track=track,
                         exp_name=exp_name,
                         wandb_project_name=wandb_project_name,
                         wandb_entity=wandb_entity)

        self.problem = problem
        self.dim = 2
        self.oracle = oracle
        self.linear_solver = linear_solver
        self.warm_start = warm_start
        self.tolerance = tolerance
        self.seed = seed  # Not used in this algorithm.

        self.bounding_box = None
        self.ideal = None
        self.nadir = None
        self.box_queue = SortedKeyList([], key=lambda x: x.volume)
        self.pf = np.empty((0, self.dim))
        self.robust_points = np.empty((0, self.dim))
        self.completed = np.empty((0, self.dim))

        self.total_hv = 0
        self.dominated_hv = 0
        self.discarded_hv = 0
        self.coverage = 0
        self.error = np.inf

        self.setup_wandb(self.config())

    def config(self):
        """Get the config of the algorithm."""
        return {
            "method": "priol_2D",
            "env_id": self.problem.env_id,
            "warm_start": self.warm_start,
            "tolerance": self.tolerance,
            "seed": self.seed
        }

    def estimate_error(self):
        """Estimate the error of the algorithm."""
        if len(self.box_queue) == 0:
            self.error = 0
        else:
            self.error = max(box.max_dist for box in self.box_queue)

    def split_box(self, box, point):
        """Split a box into two new boxes.

        Args:
            box (Box): The box to split.
            point (np.array): The point to split the box on.

        Returns:
            tuple(Box, Box): The two new boxes.
        """
        nadir1 = np.array([box.nadir[0], point[1]])
        ideal1 = np.array([point[0], box.ideal[1]])
        new_box1 = Box(ideal1, nadir1)

        nadir2 = np.array([point[0], box.nadir[1]])
        ideal2 = np.array([box.ideal[0], point[1]])
        new_box2 = Box(ideal2, nadir2)

        self.dominated_hv += Box(box.nadir, point).volume
        self.discarded_hv += Box(point, box.ideal).volume
        return new_box1, new_box2

    def update_box_queue(self, box, point):
        """Update the algorithm with a new point.

        Args:
            box (Box): The box that was searched.
            point (np.array): The new point.
        """
        new_box1, new_box2 = self.split_box(box, point)

        if new_box1.volume > self.tolerance:
            self.box_queue.add(new_box1)

        if new_box2.volume > self.tolerance:
            self.box_queue.add(new_box2)

    def get_outer_points(self):
        """Get the outer points of the problem.

        Returns:
            np.array: The outer points.
        """
        outer_points = []
        for d in range(2):
            weights = np.zeros(2)
            weights[d] = 1
            outer_points.append(self.linear_solver.solve(weights))
        return np.array(outer_points)

    def init_phase(self):
        """The initial phase in solving the problem."""
        outer_points = self.get_outer_points()
        self.bounding_box = Box(np.min(outer_points, axis=0), np.max(outer_points, axis=0))
        self.ideal = np.copy(self.bounding_box.ideal)
        self.nadir = np.copy(self.bounding_box.nadir)
        self.pf = np.vstack((self.pf, outer_points))
        self.box_queue.add(self.bounding_box)
        self.estimate_error()
        self.total_hv = self.bounding_box.volume

    def compute_hypervolume(self, points, ref, ref_offset=0.1):
        """Compute the hypervolume of a set of points.

        Args:
            points (array_like): List of points.
            ref (np.array): Reference point.

        Returns:
            float: The computed hypervolume.
        """
        ref = ref + ref_offset
        return pg.hypervolume(points).compute(ref)

    def get_next_box(self):
        """Get the next box to search."""
        if self.box_queue:
            return self.box_queue.pop(-1)
        else:
            raise Exception('No more boxes to split.')

    def is_done(self):
        """Check if the algorithm is done."""
        return not self.box_queue or 1 - self.coverage <= self.tolerance

    def solve(self, callback=None):
        """Solve the problem.

        Returns:
            ndarray: The Pareto front.
        """
        start = time.time()
        self.init_phase()
        iteration = 0
        self.log_iteration(iteration, self.dominated_hv, self.discarded_hv, self.coverage, self.error)

        while not self.is_done():
            begin_loop = time.time()
            print(f'Step {iteration} - Covered {self.coverage:.5f}% - Error {self.error:.5f}')

            box = self.get_next_box()
            ideal = np.copy(box.ideal)
            referent = np.copy(box.nadir)
            vec = self.oracle.solve(referent, ideal, warm_start=self.warm_start)

            if strict_pareto_dominates(vec, referent):  # Check that new point is valid.
                self.update_box_queue(box, vec)
                self.pf = np.vstack((self.pf, vec))
            else:
                self.discarded_hv += box.volume
                self.completed = np.vstack((self.completed, referent))
                self.robust_points = np.vstack((self.robust_points, vec))

            self.estimate_error()
            self.coverage = (self.dominated_hv + self.discarded_hv) / self.total_hv

            iteration += 1

            self.log_iteration(iteration, self.dominated_hv, self.discarded_hv, self.coverage, self.error)
            if callback is not None:
                callback(iteration, self.dominated_hv, self.discarded_hv, self.coverage, self.error)
            print(f'Ref {referent} - Found {vec} - Time {time.time() - begin_loop:.2f}s')
            print('---------------------')

        self.pf = extreme_prune(np.vstack((self.pf, self.robust_points)))
        self.dominated_hv = self.compute_hypervolume(-self.pf, -self.nadir)
        self.log_iteration(iteration + 1, self.dominated_hv, self.discarded_hv, self.coverage, self.error)
        self.close_wandb()
        print(f'Algorithm finished in {time.time() - start:.2f} seconds.')

        return self.pf
