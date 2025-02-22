from ipro.outer_loops.ipro import IPRO
from ipro.outer_loops.ipro_2D import IPRO2D


def init_outer_loop(method, problem_id, objectives, oracle, linear_solver, **kwargs):
    """Initialise an outer loop.

    Args:
        method (str): The method to use.
        problem_id (any): The problem id.
        objectives (int): The number of objectives.
        oracle (Oracle): The oracle.
        linear_solver (LinearSolver): The linear solver.

    Returns:
        OuterLoop: The outer loop.
    """
    if method == 'IPRO-2D':
        if objectives != 2:
            raise ValueError('The 2D outer loop can only be used for 2D problems.')
        return IPRO2D(problem_id, oracle, linear_solver, **kwargs)
    elif method == 'IPRO':
        return IPRO(problem_id, objectives, oracle, linear_solver, **kwargs)
    else:
        raise ValueError(f'Unknown outer loop algorithm: {method}')
