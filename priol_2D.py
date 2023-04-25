import numpy as np
from sortedcontainers import SortedKeyList

from box import Box
from pareto import p_prune, pareto_dominates


class OuterLoop2D:
    """An inner-outer loop method for solving 2D multi-objective problems."""

    def __init__(self, problem, inner_loop, linear_solver, tolerance=1e-6, log_dir=None):
        self.problem = problem
        self.inner_loop = inner_loop
        self.linear_solver = linear_solver
        self.tolerance = tolerance
        self.log_dir = log_dir

        self.bounding_box = None
        self.removed_boxes = []
        self.box_queue = SortedKeyList([], key=lambda x: x.volume)
        self.pf = set()

        self.covered_volume = 0

    def reset(self):
        """Reset the algorithm to its initial state."""
        self.bounding_box = None
        self.removed_boxes = []
        self.box_queue = SortedKeyList([], key=lambda x: x.volume)
        self.pf = set()
        self.covered_volume = 0

    def percentage_covered(self):
        """Get the percentage of the bounding box that is covered.

        Returns:
            float: The percentage of the bounding box that is covered.
        """
        return self.covered_volume / self.bounding_box.volume * 100

    def remove_box(self, box):
        """Remove a box from the algorithm.

        Args:
            box (Box): The box to remove.
        """
        self.removed_boxes.append(box)
        self.covered_volume += box.volume

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

        self.remove_box(Box(box.nadir, point))
        self.remove_box(Box(point, box.ideal))
        return new_box1, new_box2

    def update(self, box, point):
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

        self.pf.add(tuple(point))

    def get_outer_points(self):
        """Get the outer points of the problem.

        Returns:
            np.array: The outer points.
        """
        outer_points = []
        for d in range(2):
            weight_vec = np.zeros(2)
            weight_vec[d] = 1
            outer_points.append(self.linear_solver(self.problem, weight_vec))
        return np.array(outer_points)

    def init_phase(self):
        """The initial phase in solving the problem."""
        outer_points = self.get_outer_points()
        self.bounding_box = Box(np.min(outer_points, axis=0), np.max(outer_points, axis=0))
        self.pf.update([tuple(vec) for vec in outer_points])
        self.box_queue.add(self.bounding_box)

    def get_next_box(self):
        """Get the next box to search."""
        if self.box_queue:
            return self.box_queue.pop(-1)
        else:
            raise Exception('No more boxes to split.')

    def is_done(self):
        """Check if the algorithm is done."""
        return not self.box_queue

    def accept_point(self, point, utility, fast=True):
        """Check if a point is valid.

        Note:
            Preferable, this should use the utility but it does not work yet.

        Args:
            point (np.array): The point to check.
            utility (float): The utility of the point.

        Returns:
            bool: True if the point is valid, False otherwise.
        """
        if fast:
            return utility > 0

        for alternative in self.pf:
            if pareto_dominates(alternative, point) or np.all(alternative == point):
                return False
        return True

    def solve(self, log_freq=10):
        """Solve the problem."""
        self.init_phase()
        step = 0

        while not self.is_done():
            if step % log_freq == 0:
                print(f'Step {step} - Covered volume: {self.percentage_covered():.5f}%')

            box = self.get_next_box()
            target = np.copy(box.ideal)
            local_nadir = np.copy(box.nadir)
            found_vec, utility = self.inner_loop.solve(target, local_nadir)

            if self.accept_point(found_vec, utility):  # Check that new point is valid.
                self.update(box, found_vec)
            else:
                self.remove_box(box)

            step += 1

        pf = p_prune(self.pf.copy())
        return pf
