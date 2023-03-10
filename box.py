import numpy as np


class Box:
    """
    A d-dimensional box.
    """

    def __init__(self, point1, point2):
        self.dimensions = len(point1)
        self.bounds = np.array([point1, point2])
        self.nadir = np.min(self.bounds, axis=0)
        self.ideal = np.max(self.bounds, axis=0)
        self.volume = self.compute_volume()

    def compute_volume(self):
        """Compute the volume of the box.

        Returns:
            float: The volume of the box.
        """
        return abs(np.prod([max_dim - min_dim for min_dim, max_dim in zip(self.nadir, self.ideal)]))

    def get_intersecting_box(self, box):
        """If the box intersect, construct the intersecting box.

        Args:
            box (Box): The box.

        Returns:
            Box: The intersecting box.
        """
        if not self.is_intersecting(box):
            return None
        return Box(np.max([self.nadir, box.nadir], axis=0), np.min([self.ideal, box.ideal], axis=0))

    def is_intersecting(self, box):
        """If the box intersect, construct the intersecting box.

        Note:
            To check if two boxes intersect, we can compare the ranges of each dimension for the two boxes. If there is
            any dimension where the ranges of the two boxes do not overlap, then the boxes do not intersect. If the
            ranges of all dimensions overlap, then the boxes intersect.

            For example, suppose we have two boxes in three dimensions:
            Box A: [x1, x2] x [y1, y2] x [z1, z2]
            Box B: [u1, u2] x [v1, v2] x [w1, w2]

            The two boxes intersect if and only if:
            x1 <= u2 and u1 <= x2 (overlap in the x dimension)
            y1 <= v2 and v1 <= y2 (overlap in the y dimension)
            z1 <= w2 and w1 <= z2 (overlap in the z dimension)

        Args:
            box (Box): The box.

        Returns:
            Box: The intersecting box.
        """
        return np.all(np.logical_and(self.nadir <= box.ideal, box.nadir <= self.ideal))

    def __repr__(self):
        return f"Box({self.nadir}, {self.ideal})"
