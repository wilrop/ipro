import time

import numpy as np

from sortedcontainers import SortedKeyList
from copy import deepcopy
from outer_loops.outer import OuterLoop
from outer_loops.box import Box
from utils.pareto import extreme_prune, strict_pareto_dominates, batched_strict_pareto_dominates, pareto_dominates


class Priol2D(OuterLoop):
    """An inner-outer loop method for solving 2D multi-objective problems."""

    def __init__(self,
                 problem,
                 oracle,
                 linear_solver,
                 ref_point=None,
                 offset=1,
                 tolerance=1e-6,
                 max_steps=None,
                 warm_start=False,
                 track=False,
                 exp_name=None,
                 wandb_project_name=None,
                 wandb_entity=None,
                 seed=None):
        super().__init__(problem,
                         2,
                         oracle,
                         linear_solver,
                         method="priol_2D",
                         ref_point=ref_point,
                         offset=offset,
                         tolerance=tolerance,
                         max_steps=max_steps,
                         warm_start=warm_start,
                         track=track,
                         exp_name=exp_name,
                         wandb_project_name=wandb_project_name,
                         wandb_entity=wandb_entity,
                         seed=seed)

        self.box_queue = SortedKeyList([], key=lambda x: x.volume)

    def reset(self):
        """Reset the algorithm."""
        self.box_queue = SortedKeyList([], key=lambda x: x.volume)
        super().reset()

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
        return [new_box1, new_box2]

    def update_box_queue(self, box, point):
        """Update the algorithm with a new point.

        Args:
            box (Box): The box that was searched.
            point (np.array): The new point.
        """
        for box in self.split_box(box, point):
            if box.volume > self.tolerance and pareto_dominates(box.ideal, box.nadir):
                self.box_queue.add(box)

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
        self.nadir = np.min(outer_points, axis=0) - self.offset
        self.ideal = np.max(outer_points, axis=0) + self.offset
        self.ref_point = np.copy(self.nadir) if self.ref_point is None else np.array(self.ref_point)
        self.bounding_box = Box(self.nadir, self.ideal)
        self.pf = np.vstack((self.pf, outer_points))
        self.box_queue.add(self.bounding_box)
        self.estimate_error()
        self.total_hv = self.bounding_box.volume
        self.hv = self.compute_hypervolume(-self.pf, -self.ref_point)

    def get_next_box(self):
        """Get the next box to search."""
        if self.box_queue:
            return self.box_queue.pop(-1)
        else:
            raise Exception('No more boxes to split.')

    def is_done(self, step):
        """Check if the algorithm is done."""
        return not self.box_queue or 1 - self.coverage <= self.tolerance or step > self.max_steps

    def replay(self, vec, box_point_pairs):
        replay_triggered = self.replay_triggered
        self.reset()
        self.replay_triggered = replay_triggered + 1
        self.init_phase()
        idx = 0
        new_box_point_pairs = []

        for box, point in box_point_pairs:  # Replay the points that were added correctly
            self.box_queue.pop(-1)  # Remove the box.
            idx += 1
            if strict_pareto_dominates(vec, point):
                self.update_found(box, vec)
                new_box_point_pairs.append((box, vec))
                break
            elif strict_pareto_dominates(point, box.nadir):
                self.update_found(box, point)
            else:
                self.update_not_found(box, point)
            new_box_point_pairs.append((box, point))

        for box, point in box_point_pairs[idx:]:  # Process the remaining points to see if we can still add them.
            box_queue = deepcopy(self.box_queue)  # Avoid messing with the box_queue during processing.
            if strict_pareto_dominates(point, box.nadir):
                for box_id, open_box in reversed(list(enumerate(box_queue))):
                    if strict_pareto_dominates(point, open_box.nadir):
                        self.box_queue.pop(box_id)  # This is okay because we are working backwards.
                        self.update_found(open_box, point)
                        new_box_point_pairs.append((open_box, point))
                        break
            else:
                for box_id, open_box in reversed(list(enumerate(box_queue))):
                    if pareto_dominates(open_box.nadir, box.nadir):
                        self.box_queue.pop(box_id)
                        self.update_not_found(open_box, point)
                        new_box_point_pairs.append((open_box, point))

        return new_box_point_pairs

    def update_found(self, box, vec):
        """The update to perform when the Pareto oracle found a new Pareto dominant vector."""
        self.update_box_queue(box, vec)
        self.pf = np.vstack((self.pf, vec))

    def update_not_found(self, box, vec):
        """The update to perform when the Pareto oracle did not find a new Pareto dominant vector."""
        self.discarded_hv += box.volume
        self.completed = np.vstack((self.completed, np.copy(box.nadir)))
        if strict_pareto_dominates(vec, self.nadir):
            self.robust_points = np.vstack((self.robust_points, vec))

    def solve(self, callback=None):
        """Solve the problem.

        Returns:
            ndarray: The Pareto front.
        """
        self.setup_wandb()
        start = time.time()
        self.init_phase()
        iteration = 0
        self.log_iteration(iteration)
        box_point_pairs = []

        while not self.is_done(iteration):
            begin_loop = time.time()
            print(f'Step {iteration} - Covered {self.coverage:.5f}% - Error {self.error:.5f}')

            box = self.get_next_box()
            ideal = np.copy(box.ideal)
            referent = np.copy(box.nadir)
            vec = self.oracle.solve(referent, ideal, warm_start=self.warm_start)

            if strict_pareto_dominates(vec, referent):  # Check that new point is valid.
                if np.any(batched_strict_pareto_dominates(vec, np.vstack((self.pf, self.completed)))):
                    box_point_pairs = self.replay(vec, box_point_pairs)
                else:
                    self.update_found(box, vec)
                    box_point_pairs.append((box, vec))
            else:
                self.update_not_found(box, vec)
                box_point_pairs.append((box, vec))

            self.estimate_error()
            self.coverage = (self.dominated_hv + self.discarded_hv) / self.total_hv
            self.hv = self.compute_hypervolume(-self.pf, -self.ref_point)

            iteration += 1

            self.log_iteration(iteration)
            if callback is not None:
                callback(iteration, self.hv, self.dominated_hv, self.discarded_hv, self.coverage, self.error)
            print(f'Ref {referent} - Found {vec} - Time {time.time() - begin_loop:.2f}s')
            print('---------------------')

        self.pf = extreme_prune(np.vstack((self.pf, self.robust_points)))
        self.dominated_hv = self.compute_hypervolume(-self.pf, -self.nadir)
        self.hv = self.compute_hypervolume(-self.pf, -self.ref_point)
        self.log_iteration(iteration + 1)

        print(f'Algorithm finished in {time.time() - start:.2f} seconds.')

        self.close_wandb()
        return self.pf
