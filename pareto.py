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


def verify_pcs(point_set, pareto_set):
    """Verify that the Pareto coverage set is correct.

    Args:
        point_set (Set[Tuple]): A set of vectors.
        pareto_set (Set[Tuple]): A set of vectors.

    Returns:
        bool: Whether the Pareto coverage set is correct.
    """
    if type(pareto_set) != set:
        pareto_set = {tuple(vec) for vec in pareto_set}
    if type(point_set) != set:
        point_set = {tuple(vec) for vec in point_set}

    correct_pf = p_prune(point_set)
    return correct_pf, correct_pf == pareto_set
