from pymoo.indicators.hv import Hypervolume
from ipro.utils.pareto import batched_pareto_dominates


def compute_hypervolume(points, ref):
    """Compute the hypervolume of a set of points.

    Note:
        This computes the hypervolume assuming all objectives are to be minimized.

    Args:
        points (array_like): List of points.
        ref (np.array): Reference point.

    Returns:
        float: The computed hypervolume.
    """
    points = points[batched_pareto_dominates(ref, points)]
    if points.size == 0:
        return 0
    ind = Hypervolume(ref_point=ref)
    return ind(points)
