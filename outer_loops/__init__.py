from outer_loops.ipro import IPRO
from outer_loops.ipro_2D import IPRO2D


def init_outer_loop(alg, problem, objectives, oracle, linear_solver, **kwargs):
    """Initialise an outer loop.

    Args:
        alg (str): The algorithm to use.
        problem (any): The problem.
        objectives (int): The number of objectives.
        oracle (Oracle): The oracle.
        linear_solver (LinearSolver): The linear solver.

    Returns:
        OuterLoop: The outer loop.
    """
    if alg == 'IPRO-2D':
        if objectives != 2:
            raise ValueError('The 2D outer loop can only be used for 2D problems.')
        return IPRO2D(problem, oracle, linear_solver, **kwargs)
    elif alg == 'IPRO':
        return IPRO(problem, objectives, oracle, linear_solver, **kwargs)
    else:
        raise ValueError(f'Unknown outer loop algorithm: {alg}')
