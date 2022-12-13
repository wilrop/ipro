import numpy as np


class Patch2d:
    """
    A two-dimensional patch of unchecked area.
    """

    def __init__(self, point1, point2):
        self.point1 = point1
        self.point2 = point2
        self.midpoint = (point1 + point2) / 2
        self.max_x = max(point1[0], point2[0])
        self.area = np.prod(np.abs(np.array(self.point1) - np.array(self.point2)))

    def split(self, point):
        """Split a patch in two.

        Args:
            point (ndarray): The point which splits the current patch.

        Returns:
            Patch2d, Patch2d: Two new patches.
        """
        patch1 = Patch2d(self.point1, point)
        patch2 = Patch2d(self.point2, point)
        return patch1, patch2
