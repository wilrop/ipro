from itertools import product

import numpy as np

from box import Box
from pareto import p_prune, pareto_dominates


class OuterLoop:
    """An inner-outer loop method for solving arbitrary multi-objective problems."""

    def __init__(self,
                 problem,
                 dimensions,
                 inner_loop,
                 linear_solver,
                 structured_grid=False,
                 sample_size=10000,
                 tolerance=1e-1,
                 max_steps=500,
                 rng=None,
                 seed=None,
                 save_figs=False,
                 log_dir=None):
        self.problem = problem
        self.dimensions = dimensions
        self.inner_loop = inner_loop
        self.linear_solver = linear_solver
        self.tolerance = tolerance
        self.max_steps = max_steps
        self.seed = seed
        self.save_figs = save_figs
        self.log_dir = log_dir

        self.bounding_box = None
        self.removed_boxes = []
        self.pf = set()
        self.points_to_check = None
        self.options = np.array(list(product([1, -1], repeat=dimensions))) * self.tolerance

        self.structured_grid = structured_grid
        self.grid_points = None
        self.total_grid_points = None
        self.sample_size = sample_size
        self.covered_volume = 0

        self.rng = rng if rng is not None else np.random.default_rng(seed)

    def get_config(self):
        """Return a dictionary with the configuration of the algorithm."""
        return {
            'problem': self.problem,
            'dimensions': self.dimensions,
            'inner_loop': self.inner_loop,
            'linear_solver': self.linear_solver,
            'tolerance': self.tolerance,
            'max_steps': self.max_steps,
            'seed': self.seed,
            'save_figs': self.save_figs,
            'log_dir': self.log_dir,
            'structured_grid': self.structured_grid,
            'sample_size': self.sample_size,
        }

    def reset(self):
        """Reset the state of the algorithm."""
        self.bounding_box = None
        self.removed_boxes = []
        self.pf = set()
        self.points_to_check = None

        self.grid_points = None
        self.total_grid_points = None
        self.covered_volume = 0

    def update_volume(self, new_box):
        """Update the covered volume of the search space."""
        if self.structured_grid:
            valid_samples = [p for p in self.grid_points if not new_box.contains(p)]
            sample_size = len(self.total_grid_points)
        else:
            samples = self.rng.uniform(self.bounding_box.nadir, self.bounding_box.ideal,
                                       size=(self.sample_size, self.dimensions))
            valid_samples = [p for p in samples if not (new_box.contains(p) or self.in_removed_boxes(p))]
            sample_size = self.sample_size

        self.covered_volume = 1 - len(valid_samples) / sample_size

    def remove_box(self, box):
        """Remove a box from the bounding box."""
        self.update_volume(box)
        self.removed_boxes.append(box)
        new_points = [v + self.options for v in box.vertices()]
        self.points_to_check = np.vstack((self.points_to_check, *new_points))

    def update(self, point):
        """Update the algorithm after accepting a new point.

        Args:
            point (np.ndarray): A new point to add to the Pareto front and to split the box at.
        """
        self.remove_box(Box(np.copy(self.bounding_box.nadir), np.copy(point)))
        self.remove_box(Box(np.copy(point), np.copy(self.bounding_box.ideal)))
        self.points_to_check = np.vstack((self.points_to_check, point + self.options))
        self.pf.add(tuple(point))

    def accept_point(self, point):
        """Check whether to accept a new point."""
        for alternative in self.pf:
            if pareto_dominates(alternative, point) or np.all(alternative == point):
                return False
        return True

    def stretch_box_dim(self, box, dim):
        """Stretch a box in a given dimension.

        Args:
            box (Box): The box to stretch.
            dim (int): The dimension to stretch the box in.

        Returns:
            Box: The stretched box.
        """
        intersecting_boxes = [rem_box for rem_box in self.removed_boxes if box.projection_is_intersecting(rem_box, dim)]

        ideals = []
        nadirs = []

        for rem_box in intersecting_boxes:
            ideals.append(rem_box.ideal[dim])
            nadirs.append(rem_box.nadir[dim])

        ideals = np.array(ideals)
        nadirs = np.array(nadirs)
        ideals = ideals[ideals <= box.nadir[dim]]
        nadirs = nadirs[nadirs >= box.ideal[dim]]

        stretched_nadir = np.copy(box.nadir)
        stretched_ideal = np.copy(box.ideal)
        stretched_nadir[dim] = np.max(ideals, initial=self.bounding_box.nadir[dim])
        stretched_ideal[dim] = np.min(nadirs, initial=self.bounding_box.ideal[dim])
        stretched_box = Box(stretched_nadir, stretched_ideal)
        return stretched_box

    def greedy_stretch_box(self, box):
        """Greedily stretch a box in all dimensions.

        The method sequentially finds the dimension that gives the largest stretch and stretches the box in that
        dimension. It keeps doing this until no more stretching is possible.

        Args:
            box (Box): The box to stretch.

        Returns:
            Box: The stretched box.
        """
        dimensions = list(range(self.dimensions))

        while dimensions:
            stretched_boxes = []
            volume_gains = []

            for dim in dimensions:
                dim_box = self.stretch_box_dim(box, dim)
                stretched_boxes.append(dim_box)
                volume_gains.append(dim_box.volume - box.volume)

            best_dim_idxs = np.argwhere(volume_gains == np.max(volume_gains)).flatten()
            best_dim_idx = self.rng.choice(best_dim_idxs)

            if volume_gains[best_dim_idx] == 0:
                return box
            else:
                box = stretched_boxes[best_dim_idx]
                dimensions.pop(best_dim_idx)
        return box

    def expand_point_to_box(self, point):
        """Expand a point to its surrounding box that does not overlap with any removed boxes.

        Args:
            point (np.ndarray): The point to expand to a box.

        Returns:
            Box: The box that contains the point and does not overlap with any removed boxes.
        """
        points = np.array(list(self.pf))
        dim_points_splits = np.split(points, self.dimensions, axis=1)
        nadir = []
        ideal = []

        for dim, (coord, dim_points) in enumerate(zip(point, dim_points_splits)):
            nadir_coord = dim_points[dim_points <= coord].max(initial=self.bounding_box.nadir[dim])
            ideal_coord = dim_points[dim_points >= coord].min(initial=self.bounding_box.ideal[dim])
            nadir.append(nadir_coord)
            ideal.append(ideal_coord)

        expanded_box = Box(np.array(nadir), np.array(ideal))
        return expanded_box

    def greedy_max_box_from_point(self, point):
        """Greedily find the largest box containing a point that does not overlap with previously excluded boxes.

        Args:
            point (np.ndarray): The point to find the largest box for.

        Returns:
            Box: The largest box containing the point that does not overlap with previously excluded boxes.
        """
        return self.greedy_stretch_box(self.expand_point_to_box(point))

    def in_removed_boxes(self, point):
        """Check whether a point is in any of the removed boxes.

        Args:
            point (np.ndarray): The point to check.

        Returns:
            bool: Whether the point is in any of the removed boxes.
        """
        for box in self.removed_boxes:
            if box.contains(point):
                return True
        return False

    def get_start_point(self):
        """Get a starting point for the algorithm.

        Returns:
            np.ndarray: A starting point for the algorithm.
        """
        point_indices = self.rng.permutation(len(self.points_to_check))
        points_to_remove = []
        start_point = None

        for pid in point_indices:
            point = self.points_to_check[pid]
            if self.bounding_box.contains(point) and not self.in_removed_boxes(point):
                start_point = point
                break
            else:
                points_to_remove.append(pid)

        self.points_to_check = np.delete(self.points_to_check, points_to_remove, axis=0)
        return start_point

    def get_next_box(self):
        """Get the next box to consider.

        Returns:
            Box: The next box to consider.
        """
        x0 = self.get_start_point()
        if x0 is None:
            return None
        greedy_box = self.greedy_max_box_from_point(x0)
        return greedy_box

    def is_done(self):
        """Check whether the algorithm is done."""
        return self.covered_volume >= 1.

    def get_outer_points(self):
        """Get the outer points of the Pareto front.

        Note:
            This also includes the worst points, as they are necessary to construct the nadir of the bounding box.

        Returns:
            np.ndarray: The minimum and maximum points of the objective space.
        """
        max_points = []
        min_points = []
        weight_vecs = np.eye(self.dimensions)
        for weight_vec in weight_vecs:
            max_points.append(self.linear_solver(self.problem, weight_vec))
            min_points.append(self.linear_solver(self.problem, -weight_vec))
        return np.array(min_points), np.array(max_points)

    def init_phase(self):
        """The initial phase in solving the problem."""
        min_points, max_points = self.get_outer_points()
        nadir = np.min(min_points, axis=0)
        ideal = np.max(max_points, axis=0)
        expanded_nadir = nadir - 1
        expanded_ideal = ideal + 1
        self.bounding_box = Box(expanded_nadir, expanded_ideal)

        if self.structured_grid:
            num_steps = (self.bounding_box.ideal - self.bounding_box.nadir) / self.tolerance
            coord_spaces = []
            for start, stop, num in zip(self.bounding_box.nadir, self.bounding_box.ideal, num_steps):
                coord_spaces.append(np.linspace(start, stop, num=int(num), endpoint=True))
            self.grid_points = list(product(*coord_spaces))
            self.total_grid_points = len(self.grid_points)

        self.points_to_check = np.vstack([point + self.options for point in self.bounding_box.vertices()])

        for point in max_points:
            self.update(point)

    def solve(self, log_freq=10):
        """Solve the problem."""
        self.init_phase()
        step = 0

        while not self.is_done() and step < self.max_steps:
            if step % log_freq == 0:
                print(f'Step {step} - Covered volume: {self.covered_volume * 100:.5f}%')

            box = self.get_next_box()
            if box is None:
                print(f'No more boxes to check. Stopping at step {step}. This should not happen.')
                break

            target = np.copy(box.ideal)
            local_nadir = np.copy(box.nadir)
            found_vec, utility = self.inner_loop.solve(target, local_nadir)

            if self.accept_point(found_vec):  # Check that new point is valid.
                self.update(found_vec)
            else:
                self.remove_box(box)

            step += 1

        pf = p_prune(self.pf.copy())
        return pf
