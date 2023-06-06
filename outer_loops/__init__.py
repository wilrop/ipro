from outer_loops.priol import Priol
from outer_loops.priol_2D import Priol2D


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
    if alg == '2D':
        if objectives != 2:
            raise ValueError('The 2D outer loop can only be used for 2D problems.')
        return Priol2D(problem, oracle, linear_solver, **kwargs)
    elif alg == 'PRIOL':
        return Priol(problem, objectives, oracle, linear_solver, **kwargs)
    else:
        raise ValueError(f'Unknown outer loop algorithm: {alg}')
