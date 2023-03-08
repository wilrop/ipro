from math import floor, log2

import numpy as np


def dumitrescu_jiang(points, nadir, ideal):
    """The Dimutrescu-Jiang algorithm for computing a large empty box amidst n points in [0, 1]^d.

    Notes:
        The volume of the largest empty box is guaranteed to be larger than or equal to log(d)/(4 * (n + log(d))).
        The algorithm is guaranteed to produce an empty box of volume at least this value and has a runtime complexity
        of O(n + d log(d)). As such, for constant d, the algorithm runs in linear time.

        To ensure that the start box is [0, 1]^d the points are normalised, which is always possible to do from the
        nadir and ideal vectors.

        This is executed on the Pareto front.

    args:
        points (ndarray): The points in [0, 1]^d.
        d (int): The dimension of the space.

    Returns:
        ndarray: The lower and upper bounds of the empty box.
    """
    d = len(nadir)
    n = len(points)
    scaled_points = (points - nadir) / (ideal - nadir)

    # Compute the setup variables.
    l = floor(log2(d))
    k = floor(n / (l + 1))
    num_boxes = k + 1

    # Divide in equal volume boxes. The volume at this point is 1/(k+1).
    hyperplanes, stepsize = np.linspace(0, 1, num=num_boxes + 1, retstep=True)

    # Find the box with the lowest number of points.
    containments = [[] for _ in range(num_boxes)]
    for point in scaled_points:
        box_idx = floor(point[0] / stepsize)
        containments[box_idx].append(point)

    min_box_idx = np.argmin([len(box) for box in containments])
    min_box_points = np.array(containments[min_box_idx])
    a, b = np.zeros(d), np.ones(d)
    a[0], b[0] = hyperplanes[min_box_idx], hyperplanes[min_box_idx + 1]

    # Construct the binary vectors.
    comp_vals = np.reshape((a + b) / 2, (d, -1))
    binary_vectors = min_box_points.T > comp_vals

    # Construct the final large boxes.
    vector_sums = np.sum(binary_vectors, axis=1)
    min_sum_idx = np.argmin(vector_sums)
    max_sum_idx = np.argmax(vector_sums)
    min_sum = vector_sums[min_sum_idx]
    max_sum = vector_sums[max_sum_idx]

    large_box_a, large_box_b = None, None
    if min_sum == 0:
        large_box_a = np.concatenate((a[:min_sum_idx], [(a[min_sum_idx] + b[min_sum_idx]) / 2], a[min_sum_idx + 1:]))
        large_box_b = b
    elif max_sum == d:
        large_box_a = a
        large_box_b = np.concatenate((b[:min_sum_idx], (a[min_sum_idx] + b[min_sum_idx]) / 2, b[min_sum_idx + 1:]))
    else:  # There is a pair of equal rows. We can construct an empty quarter.
        integer_values = np.packbits(binary_vectors, axis=1).squeeze()
        sorted_indices = np.argsort(integer_values)
        for idx in sorted_indices:
            i, j = idx, idx + 1
            if integer_values[i] == integer_values[j]:
                large_box_a = np.concatenate((a[:j], (a[j] + b[j]) / 2, a[j + 1:]))
                large_box_b = np.concatenate((a[:i], (a[i] + b[i]) / 2, a[i + 1:]))
                break

    if large_box_a is None or large_box_b is None:
        raise ValueError('Could not construct the empty box.')

    large_box_a = large_box_a * (ideal - nadir) + nadir
    large_box_b = large_box_b * (ideal - nadir) + nadir
    return large_box_a, large_box_b
