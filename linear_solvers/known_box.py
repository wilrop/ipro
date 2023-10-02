import numpy as np


class KnownBox:
    """A linear solver for settings with known minimal and maximal points"""

    def __init__(self, minimals, maximals):
        self.minimals = minimals
        self.maximals = maximals

    def solve(self, weight):
        """Find the utility maximising point.

        Args:
            weight (np.ndarray): The weights to use.

        Returns:
            np.ndarray: The utility maximising point.
        """
        if sum(weight) < 0:
            vecs = self.minimals
            weight *= -1
        else:
            vecs = self.maximals

        return np.sum(vecs * np.reshape(weight, (-1, 1)), axis=0)
