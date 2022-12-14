import numpy as np


class Patch2d:
    """
    A two-dimensional patch of unchecked area.
    """

    def __init__(self, point1, point2):
        self.top_left, self.top_right, self.bot_left, self.bot_right = None, None, None, None
        self.init_rectangle(point1, point2)
        self.midpoint = (point1 + point2) / 2
        self.area = np.prod(np.abs(np.array(point1) - np.array(point2)))

    def init_rectangle(self, point1, point2):
        if point1[0] > point2[0]:
            self.bot_right = point1
            self.top_left = point2
        else:
            self.bot_right = point2
            self.top_left = point1

        self.top_right = np.array([self.bot_right[0], self.top_left[1]])
        self.bot_left = np.array([self.top_left[0], self.bot_right[1]])

    def split(self, point):
        """Split a patch in two.

        Args:
            point (ndarray): The point which splits the current patch.

        Returns:
            Patch2d, Patch2d: Two new patches.
        """
        patch1 = Patch2d(self.top_left, point)
        patch2 = Patch2d(self.bot_right, point)
        return patch1, patch2

    def get_intersection_point(self, start=(0, 0)):
        """Get the furthest intersection point of a line through the midpoint with the patch.

        Args:
            start (array_like, optional): The start point for the line. By default, this is the origin.

        Returns:
            ndarray: The intersection point.
        """
        start = np.array(start)
        diffs = self.midpoint - start
        slope = diffs[1] / diffs[0]
        b = start[1] - slope * start[0]

        y_at_bot_left = slope * self.bot_left[0] + b
        if y_at_bot_left >= self.bot_left[0]:
            intersect_y = slope * self.bot_right[0] + b
            target = np.array([self.bot_right[0], intersect_y])
        else:
            intersect_x = (self.top_right[1] - b) / slope
            target = np.array([self.top_right[1], intersect_x])
        return target
