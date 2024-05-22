import numpy as np
from ipro.utils.pareto import extreme_prune
from ipro.oracles.vector_u import create_batched_aasf


class FiniteOracle:
    """A simple oracle for the known problem setting."""

    def __init__(self, problem, aug=0.01):
        self.problem = problem
        self.aug = aug

    def get_nadir(self):
        """Get the true nadir point.

        Note:
            This is purely used for testing purposes.

        Returns:
            np.ndarray: The true nadir point.
        """
        correct_pf = extreme_prune(np.copy(self.problem))
        return np.min(correct_pf, axis=0)

    def solve(self, referent, ideal):
        """The inner loop solver for the basic setting.

        Args:
            referent (np.ndarray): The reference vector.
            ideal (np.ndarray): The ideal vector.

        Returns:
            np.ndarray: The Pareto optimal vector.
        """
        u_f = create_batched_aasf(referent, referent, ideal, aug=self.aug)
        utilities = u_f(self.problem)
        best_point = np.argmax(utilities)
        return self.problem[best_point]
