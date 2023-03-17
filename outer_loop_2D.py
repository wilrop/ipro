import numpy as np
from sortedcontainers import SortedKeyList

from box import Box
from pareto import p_prune, pareto_dominates


class OuterLoop2D:
    def __init__(self, problem, inner_loop, linear_solver, tolerance=1e-6, log_dir=None):
        self.problem = problem
        self.inner_loop = inner_loop
        self.linear_solver = linear_solver
        self.tolerance = tolerance
        self.log_dir = log_dir

        self.bounding_box = None
        self.box_queue = SortedKeyList([], key=lambda x: x.volume)
        self.pf = set()

    def reset(self):
        """Reset the algorithm to its initial state."""
        self.bounding_box = None
        self.box_queue = SortedKeyList([], key=lambda x: x.volume)

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

    def init_fase(self):
        """The initial fase in solving the problem."""
        outer_points = self.get_outer_points()
        self.bounding_box = Box(np.min(outer_points, axis=0), np.max(outer_points, axis=0))
        self.pf.update([tuple(vec) for vec in outer_points])
        self.box_queue.add(self.bounding_box)

    def get_next_box(self):
        """Get the next box to search."""
        if self.box_queue:
            return self.box_queue.pop(0)
        else:
            raise Exception('No more boxes to split.')

    def is_done(self):
        """Check if the algorithm is done."""
        return not self.box_queue

    def accept_point(self, point):
        """Check if a point is valid.

        Args:
            point (np.array): The point to check.

        Returns:
            bool: True if the point is valid, False otherwise.
        """
        for alternative in self.pf:
            if pareto_dominates(alternative, point) or np.all(alternative == point):
                return False
        return True

    def solve(self):
        """Solve the problem."""
        self.init_fase()
        step = 0

        while not self.is_done():
            box = self.get_next_box()
            target = box.ideal
            local_nadir = box.nadir
            found_vec = self.inner_loop.solve(target, local_nadir)

            if self.accept_point(found_vec):  # Check that new point is valid.
                self.update(box, found_vec)

            step += 1

        pf = p_prune({tuple(vec) for vec in self.pf})
        return pf
