from ipro.linear_solvers.linear_solver import LinearSolver
from ipro.linear_solvers.finite import Finite


def init_linear_solver(alg: str, *args: tuple, **kwargs: dict) -> LinearSolver:
    """Initialise a linear solver."""
    if alg == 'finite':
        return Finite(*args, **kwargs)
    else:
        raise ValueError(f'Unknown linear solver algorithm: {alg}')
