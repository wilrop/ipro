import numpy as np


class KnownBox:
    """A linear solver for settings with known ideal and nadir points"""

    def __init__(self, nadirs, ideals):
        self.nadirs = nadirs
        self.ideals = ideals

    def solve(self, weight):
        """Find the utility maximising point.

        Args:
            weight (np.ndarray): The weights to use.

        Returns:
            np.ndarray: The utility maximising point.
        """
        if sum(weight) < 0:
            vecs = self.nadirs
            weight *= -1
        else:
            vecs = self.ideals

        return np.sum(vecs * np.reshape(weight, (-1, 1)), axis=0)
