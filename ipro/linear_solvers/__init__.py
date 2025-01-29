from ipro.linear_solvers.linear_solver import LinearSolver
from ipro.linear_solvers.finite import Finite
from typing import Any


def init_linear_solver(alg: str, *args: Any, **kwargs: Any) -> LinearSolver:
    """Initialise a linear solver."""
    if alg == 'finite':
        return Finite(*args, **kwargs)
    else:
        raise ValueError(f'Unknown linear solver algorithm: {alg}')
