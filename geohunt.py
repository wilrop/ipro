import numpy as np
from sortedcontainers import SortedKeyList

from patch2d import Patch2d


def inner_loop(momdp, target):
    vec = target
    return vec


def get_edge_points():
    return np.array([5, 1]), np.array([1, 5])


def geohunt2d(momdp, tolerance=0.1):
    max_x, max_y = get_edge_points()
    point1 = np.array([max_y[0], max_x[1]])  # Bottom left intersection of two dominated areas.
    point2 = np.array([max_x[0], max_y[1]])  # Upper right corner.
    start_patch = Patch2d(point1, point2)
    patches_queue = SortedKeyList([start_patch], key=lambda x: x.area)

    while patches_queue:
        patch = patches_queue.pop()

        target = patch.get_intersection_point()
        pareto_optimal_vec = inner_loop(momdp, target)
        if pareto_optimal_vec not in [patch.point1, patch.point2]:  # Check that a new Pareto optimal point was found.
            new_patch1, new_patch2 = patch.split(pareto_optimal_vec)

            if new_patch1.area > tolerance:
                patches_queue.add(new_patch1)

            if new_patch2.area > tolerance:
                patches_queue.add(new_patch2)


if __name__ == '__main__':
    momdp = 0
    tolerance = 0.1
    geohunt2d(momdp, tolerance=tolerance)
