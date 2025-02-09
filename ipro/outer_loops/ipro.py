import time
import numpy as np
from ipro.outer_loops.outer import OuterLoop
from ipro.outer_loops.box import Box
from ipro.utils.pareto import strict_pareto_dominates, batched_strict_pareto_dominates, extreme_prune, pareto_dominates
from ipro.utils.hypervolume import compute_hypervolume


class IPRO(OuterLoop):
    """IPRO algorithm for solving multi-objective problems."""

    def __init__(
            self,
            problem_id,
            dimensions,
            oracle,
            linear_solver,
            ref_point=None,
            offset=1,
            tolerance=1e-1,
            max_iterations=None,
            known_pf=None,
            track=False,
            exp_name=None,
            wandb_project_name=None,
            wandb_entity=None,
            rng=None,
            seed=None,
            extra_config=None,
    ):
        super().__init__(problem_id,
                         dimensions,
                         oracle,
                         linear_solver,
                         method="IPRO",
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
                         extra_config=extra_config)
        self.lower_points = []
        self.upper_points = []

        self.rng = rng if rng is not None else np.random.default_rng(seed)

    def reset(self):
        """Reset the algorithm."""
        self.lower_points = []
        self.upper_points = []
        super().reset()

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
        nadir = nadir - self.offset  # Necessary to ensure every Pareto optimal point strictly dominates the nadir.
        ideal = ideal + self.offset
        self.nadir = np.copy(nadir)
        self.ideal = np.copy(ideal)
        self.ref_point = np.copy(nadir) if self.ref_point is None else np.array(self.ref_point)
        self.hv = compute_hypervolume(-self.pf, -self.ref_point)

        if len(self.pf) == 1:  # If the Pareto front is the ideal.
            return True

        self.bounding_box = Box(nadir, ideal)
        self.total_hv = self.bounding_box.volume
        self.lower_points = np.array([nadir])

        for point in self.pf:  # Initialise the lower points.
            self.update_lower_points(np.array(point))

        self.upper_points = np.array([ideal])  # Initialise the upper points.
        self.error = max(ideal - nadir)
        self.compute_hvis()
        self.oracle.init_oracle(nadir=self.nadir, ideal=self.ideal)  # Initialise the oracle.
        return False

    def compute_hvis(self, num=50):
        """Compute the hypervolume improvements of the lower points.

        Note:
            An optional num parameter can be given as computing the hypervolume for a large number of potential points
            is expensive.

        Args:
            num (int, optional): The number of lower points to compute hypervolume improvements for. Defaults to 50.
        """
        point_set = np.vstack((self.pf, self.completed))
        hvis = np.zeros(len(self.lower_points))

        for lower_id in self.rng.choice(len(self.lower_points), min(num, len(self.lower_points)), replace=False):
            hv = compute_hypervolume(np.vstack((point_set, self.lower_points[lower_id])), self.ideal)
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
        self.error = error

    def update_upper_points(self, vec):
        """Update the upper set.

        Args:
            vec (np.array): The point that is added to the boundary of the dominating space.
        """
        strict_dominates = batched_strict_pareto_dominates(self.upper_points, vec)
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
        strict_dominates = batched_strict_pareto_dominates(vec, self.lower_points)
        to_keep = self.lower_points[strict_dominates == 0]
        shifted = np.stack([self.lower_points[strict_dominates == 1]] * self.dim)
        shifted[range(self.dim), :, range(self.dim)] = np.expand_dims(vec, -1)
        shifted = shifted.reshape(-1, self.dim)
        shifted = shifted[np.all(self.ideal > shifted, axis=-1)]

        new_lower_points = np.vstack((to_keep, shifted))
        self.lower_points = -extreme_prune(-new_lower_points)

    def update_discarded_hv(self):
        """Update the hypervolume of the dominating space."""
        self.discarded_hv = compute_hypervolume(np.vstack((self.pf, self.completed)), self.ideal)

    def update_dominated_hv(self):
        """Update the hypervolume of the dominated space."""
        self.dominated_hv = compute_hypervolume(-self.pf, -self.nadir)

    def select_referent(self, method='random'):
        """The method to select a new referent."""
        if method == 'random':
            return self.lower_points[self.rng.integers(0, len(self.lower_points))]
        if method == 'first':
            return self.lower_points[0]
        else:
            raise ValueError(f'Unknown method {method}')

    def is_done(self, step):
        """Check if the algorithm is done."""
        return 1 - self.coverage <= self.tolerance or step >= self.max_iterations

    def replay(self, vec, ref_point_pairs):
        """Replay the algorithm while accounting for the non-optimal Pareto oracle.

        Note:
            This reexecutes the initialisation phase which may trigger expensive compute again. However, we always use
            a given box in the experiments, so this makes no difference **in this specific case**.

        Args:
            vec (ndarray): The vector that causes the conflict.
            ref_point_pairs (List): A list of (referent, point) tuples.

        Returns:
            An updated list of referent point tuples.
        """
        replay_triggered = self.replay_triggered
        self.reset()
        self.replay_triggered = replay_triggered + 1
        self.init_phase()
        idx = 0
        new_ref_point_pairs = []

        for ref, point in ref_point_pairs:  # Replay the points that were added correctly
            idx += 1
            if strict_pareto_dominates(point, ref):
                if strict_pareto_dominates(vec, point):
                    self.update_found(vec)
                    new_ref_point_pairs.append((ref, vec))
                    break
                else:
                    self.update_found(point)
                    new_ref_point_pairs.append((ref, point))
            else:
                if strict_pareto_dominates(vec, ref):
                    self.update_found(vec)
                    new_ref_point_pairs.append((ref, vec))
                    break
                else:
                    self.update_not_found(ref, point)
                    new_ref_point_pairs.append((ref, point))

        for ref, point in ref_point_pairs[idx:]:  # Process the remaining points to see if we can still add them.
            lower_points = np.copy(self.lower_points)  # Avoids messing with lower points while iterating over them.
            if strict_pareto_dominates(point, ref):
                for lower in lower_points:
                    if strict_pareto_dominates(point, lower):
                        self.update_found(point)
                        new_ref_point_pairs.append((lower, point))
                        break
            else:
                for lower in lower_points:
                    if pareto_dominates(lower, ref):
                        self.update_not_found(lower, point)
                        new_ref_point_pairs.append((lower, point))
        return new_ref_point_pairs

    def update_found(self, vec):
        """The update to perform when the Pareto oracle found a new Pareto dominant vector."""
        self.pf = np.vstack((self.pf, vec))
        self.update_lower_points(vec)
        self.update_upper_points(vec)

    def update_not_found(self, referent, vec):
        """The update to perform when the Pareto oracle did not find a new Pareto dominant vector."""
        self.completed = np.vstack((self.completed, referent))
        self.lower_points = self.lower_points[np.any(self.lower_points != referent, axis=1)]
        self.update_upper_points(referent)
        if strict_pareto_dominates(vec, self.nadir):
            self.robust_points = np.vstack((self.robust_points, vec))

    def solve(self, update_freq=1, callback=None):
        """Solve the problem.

        Args:
            update_freq (int, optional): The frequency of updates. Defaults to 50.

        Returns:
            set: The Pareto front.
        """
        start = self.setup()
        done = self.init_phase()
        iteration = 0

        if done:
            print('The problem is solved in the initial phase.')
            print(self.pf)
            return {tuple(vec) for vec in self.pf}

        self.log_iteration(iteration)
        ref_point_pairs = []

        while not self.is_done(iteration):
            begin_loop = time.time()
            print(f'Iter {iteration} - Covered {self.coverage:.5f}% - Error {self.error:.5f}')

            referent = np.copy(self.select_referent(method='first'))
            vec = self.oracle.solve(referent, nadir=referent)

            if strict_pareto_dominates(vec, referent):
                if np.any(batched_strict_pareto_dominates(vec, np.vstack((self.pf, self.completed)))):
                    ref_point_pairs = self.replay(vec, ref_point_pairs)
                else:
                    self.update_found(vec)
                    ref_point_pairs.append((referent, vec))
            else:
                self.update_not_found(referent, vec)
                ref_point_pairs.append((referent, vec))

            if iteration % update_freq == 0:
                self.compute_hvis()

            self.update_dominated_hv()
            self.update_discarded_hv()
            self.estimate_error()
            self.coverage = (self.dominated_hv + self.discarded_hv) / self.total_hv
            self.hv = compute_hypervolume(-self.pf, -self.ref_point)

            iteration += 1
            self.log_iteration(iteration, referent=referent, ideal=self.ideal, pareto_point=vec)
            if callback is not None:
                callback(iteration, self.hv, self.dominated_hv, self.discarded_hv, self.coverage, self.error)
            print(f'Ref {referent} - Found {vec} - Time {time.time() - begin_loop:.2f}s')
            print('---------------------')

        self.finish(start, iteration)
        return self.pf.copy()
