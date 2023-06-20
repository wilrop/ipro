import time

import numpy as np
from sortedcontainers import SortedKeyList

from outer_loops.box import Box
from utils.pareto import extreme_prune, strict_pareto_dominates


class Priol2D:
    """An inner-outer loop method for solving 2D multi-objective problems."""

    def __init__(self,
                 problem,
                 oracle,
                 linear_solver,
                 writer,
                 warm_start=False,
                 tolerance=1e-6,
                 seed=None):
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
        self.error_estimates = []

        self.writer = writer

    def estimate_error(self):
        """Estimate the error of the algorithm."""
        if len(self.box_queue) == 0:
            self.error_estimates.append(0)
        else:
            largest_box = self.box_queue[-1]
            self.error_estimates.append(np.max(largest_box.ideal - largest_box.nadir))

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

    def get_next_box(self):
        """Get the next box to search."""
        if self.box_queue:
            return self.box_queue.pop(-1)
        else:
            raise Exception('No more boxes to split.')

    def is_done(self):
        """Check if the algorithm is done."""
        return not self.box_queue or self.error_estimates[-1] <= self.tolerance

    def log_step(self, step):
        self.writer.add_scalar(f'outer/dominated_hv', self.dominated_hv, step)
        self.writer.add_scalar(f'outer/discarded_hv', self.discarded_hv, step)
        self.writer.add_scalar(f'outer/coverage', self.coverage, step)
        self.writer.add_scalar(f'outer/error', self.error_estimates[-1], step)

    def solve(self):
        """Solve the problem.

        Returns:
            set: The Pareto front.
        """
        start = time.time()
        self.init_phase()
        step = 0
        self.log_step(step)

        while not self.is_done():
            begin_loop = time.time()
            print(f'Step {step} - Covered {self.coverage:.5f}% - Error {self.error_estimates[-1]:.5f}')

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

            step += 1

            self.log_step(step)
            print(f'Ref {referent} - Found {vec} - Time {time.time() - begin_loop:.2f}s')
            print('---------------------')

        pf = {tuple(vec) for vec in extreme_prune(np.vstack((self.pf, self.robust_points)))}

        print(f'Algorithm finished in {time.time() - start:.2f} seconds.')
        return pf
