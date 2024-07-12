from ipro.linear_solvers.finite import Finite


def init_linear_solver(alg, *args, **kwargs):
    """Initialise a linear solver.

    Args:
        alg (str): The algorithm to use.
        *args: The arguments to pass to the algorithm.
        **kwargs: The keyword arguments to pass to the algorithm.

    Returns:
        LinearSolver: The linear solver.
    """
    if alg == 'finite':
        return Finite(*args, **kwargs)
    else:
        raise ValueError(f'Unknown linear solver algorithm: {alg}')
