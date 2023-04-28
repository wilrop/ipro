import numpy as np


class Finite:
    def __init__(self, problem):
        self.problem = problem

    def solve(self, weights):
        """Find the utility maximising point.

        Args:
            weights (np.ndarray): The weights to use.

        Returns:
            np.ndarray: The utility maximising point.
        """
        max_utility_index = np.argmax(np.dot(self.problem, weights))
        return self.problem[max_utility_index]
