import numpy as np


def pareto_dominates(a, b):
    """Check if the vector in a Pareto dominates vector b.

    Args:
        a (ndarray): A numpy array.
        b (ndarray): A numpy array.

    Returns:
        bool: Whether vector a dominates vector b.
    """
    a = np.array(a)
    b = np.array(b)
    return np.all(a >= b) and np.any(a > b)


def p_prune(candidates):
    """Create a Pareto coverage set from a set of candidate points.

    References:
        .. [1] Roijers, D. M., & Whiteson, S. (2017). Multi-objective decision making. 34, 129–129.
            https://doi.org/10.2200/S00765ED1V01Y201704AIM034

    Args:
        candidates (Set[Tuple]): A set of vectors.

    Returns:
        Set[Tuple]: A Pareto coverage set.
    """
    pcs = set()
    while candidates:
        vector = candidates.pop()

        for alternative in candidates:
            if pareto_dominates(alternative, vector):
                vector = alternative

        to_remove = set(vector)
        for alternative in candidates:
            if pareto_dominates(vector, alternative):
                to_remove.add(alternative)

        candidates -= to_remove
        pcs.add(vector)
    return pcs


def is_dominated(vec, pcs):
    """Check if a vector is dominated by a Pareto coverage set.

    Args:
        vec (ndarray): A vector.
        pcs (Set[Tuple]): A Pareto coverage set.

    Returns:
        bool: Whether the vector is dominated by the Pareto coverage set.
    """
    for alternative in pcs:
        if pareto_dominates(alternative, vec):
            return True
    return False
