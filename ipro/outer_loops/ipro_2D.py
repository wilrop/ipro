import numpy as np

from sortedcontainers import SortedKeyList
from copy import deepcopy

from ipro.outer_loops.typing import Subproblem
from ipro.outer_loops.outer import OuterLoop
from ipro.outer_loops.box import Box
from ipro.utils.pareto import strict_pareto_dominates, extreme_prune, pareto_dominates


class IPRO2D(OuterLoop):
    """IPRO algorithm for solving bi-objective multi-objective problems."""

    def __init__(self,
                 problem_id,
                 oracle,
                 linear_solver,
                 direction='maximize',
                 ref_point=None,
                 offset=1,
                 tolerance=1e-6,
                 max_iterations=None,
                 known_pf=None,
                 track=False,
                 exp_name=None,
                 wandb_project_name=None,
                 wandb_entity=None,
                 seed=None,
                 extra_config=None,
                 ):
        super().__init__(
            problem_id,
            2,
            oracle,
            linear_solver,
            method="IPRO-2D",
            direction=direction,
            ref_point=ref_point,
            offset=offset,
            tolerance=tolerance,
            max_iterations=max_iterations,
            known_pf=known_pf,
            track=track,
            exp_name=exp_name,
            wandb_project_name=wandb_project_name,
            wandb_entity=wandb_entity,
            seed=seed,
            extra_config=extra_config
        )

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
            outer_point = self.sign * self.linear_solver.solve(weights)
            outer_points.append(outer_point)
        return np.array(outer_points)

    def init_phase(self):
        """The initial phase in solving the problem."""
        outer_points = self.get_outer_points()
        self.nadir = np.min(outer_points, axis=0) - self.offset
        self.ideal = np.max(outer_points, axis=0) + self.offset
        self.ref_point = np.copy(self.nadir) if self.ref_point is None else np.array(self.ref_point)
        self.bounding_box = Box(self.nadir, self.ideal)
        self.pf = extreme_prune(np.array(outer_points))
        self.box_queue.add(self.bounding_box)
        self.estimate_error()
        self.total_hv = self.bounding_box.volume
        self.hv = self.compute_hypervolume(-self.sign * self.pf, -self.sign * self.ref_point)
        self.oracle.init_oracle(nadir=self.sign * self.nadir, ideal=self.sign * self.ideal)  # Initialise the oracle.
        return len(self.pf) == 1

    def is_done(self, step):
        """Check if the algorithm is done."""
        return not self.box_queue or super().is_done(step)

    def get_iterable_for_replay(self):
        box_queue = deepcopy(self.box_queue)
        return reversed(list(enumerate(box_queue)))

    def maybe_add_solution(
            self,
            subproblem: Subproblem,
            point: np.ndarray,
            item: tuple[Box, int],
    ) -> Subproblem | bool:
        open_box, open_box_idx = item
        if strict_pareto_dominates(point, open_box.nadir):
            new_subproblem = Subproblem(referent=open_box.nadir, nadir=open_box.nadir, ideal=open_box.ideal)
            self.update_found(new_subproblem, point, box_idx=open_box_idx)
            return new_subproblem
        else:
            return False

    def maybe_add_completed(
            self,
            subproblem: Subproblem,
            point: np.ndarray,
            item: tuple[Box, int],
    ) -> Subproblem | bool:
        open_box, open_box_idx = item
        if pareto_dominates(open_box.nadir, subproblem.referent):
            new_subproblem = Subproblem(referent=open_box.nadir, nadir=open_box.nadir, ideal=open_box.ideal)
            self.update_not_found(new_subproblem, point, box_idx=open_box_idx)
            return new_subproblem
        else:
            return False

    def update_found(self, subproblem, vec, box_idx=-1):
        """The update to perform when the Pareto oracle found a new Pareto dominant vector."""
        self.update_box_queue(self.box_queue.pop(box_idx), vec)
        self.pf = np.vstack((self.pf, vec))

    def update_not_found(self, subproblem, vec, box_idx=-1):
        """The update to perform when the Pareto oracle did not find a new Pareto dominant vector."""
        box = self.box_queue.pop(box_idx)
        self.discarded_hv += box.volume
        self.completed = np.vstack((self.completed, np.copy(subproblem.referent)))
        if strict_pareto_dominates(vec, self.nadir):
            self.robust_points = np.vstack((self.robust_points, vec))

    def decompose_problem(self, iteration, method='first'):
        box = self.box_queue[-1]
        subproblem = Subproblem(referent=box.nadir, nadir=box.nadir, ideal=box.ideal)
        return subproblem

    def update_excluded_volume(self):
        """This is already handled when splitting the boxes."""
        pass
