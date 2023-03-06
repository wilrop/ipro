import numpy as np


class Patch2d:
    """
    A two-dimensional patch of unchecked area.
    """

    def __init__(self, point1, point2):
        self.top_left, self.top_right, self.bot_left, self.bot_right = None, None, None, None
        self.width = None
        self.height = None
        self.init_rectangle(point1, point2)
        self.midpoint = (point1 + point2) / 2
        self.area = np.prod(np.abs(np.array(point1) - np.array(point2)))

    def init_rectangle(self, point1, point2):
        """Initialise the rectangle.

        Args:
            point1 (ndarray): A point on the rectangle. Either the top left or bottom right.
            point2 (ndarray): A point on the rectangle. Either the top left or bottom right.
        """
        if point1[0] > point2[0]:
            self.bot_right = point1
            self.top_left = point2
        else:
            self.bot_right = point2
            self.top_left = point1

        self.top_right = np.array([self.bot_right[0], self.top_left[1]])
        self.bot_left = np.array([self.top_left[0], self.bot_right[1]])

        self.width = self.top_right[0] - self.top_left[0]
        self.height = self.top_right[1] - self.bot_right[1]

    def on_rectangle(self, point):
        """Check if a point is on the rectangle.

        Args:
            point (ndarray): A point.

        Returns:
            bool: Whether the point is on the rectangle.
        """
        borders = np.array([self.top_left, self.top_right, self.bot_right, self.bot_left])
        return any([np.all(point == border) for border in borders])

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
        if y_at_bot_left >= self.bot_left[1]:
            intersect_y = slope * self.bot_right[0] + b
            target = np.array([self.bot_right[0], intersect_y])
        else:
            intersect_x = (self.top_right[1] - b) / slope
            target = np.array([intersect_x, self.top_right[1]])
        return target

    def get_target(self, method='ideal', start=(0, 0)):
        """Get the target point of the patch.

        Args:
            method (str, optional): The method to use. Either 'ideal' or 'nadir'. Defaults to 'ideal'.
            start (array_like, optional): The start point for the line. By default, this is the origin.

        Returns:
            ndarray: The target point.
        """
        if method == 'ideal':
            return self.get_ideal()
        elif method == 'intersection':
            return self.get_intersection_point(start=start)
        else:
            raise ValueError(f'Invalid method: {method}')

    def get_nadir(self):
        """Get the nadir point of the patch.

        Returns:
            ndarray: The nadir point.
        """
        return self.bot_left

    def get_ideal(self):
        """Get the ideal point of the patch.

        Returns:
            ndarray: The ideal point.
        """
        return self.top_right

    def __repr__(self):
        return f'(BL: {self.bot_left}, BR: {self.bot_right}, TL: {self.top_left}, TR: {self.top_right})'
