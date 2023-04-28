from outer_loops.priol import Priol
from outer_loops.priol_2D import Priol2D


def init_outer_loop(alg, problem, objectives, oracle, linear_solver, seed=None):
    """Initialise an outer loop.

    Args:
        alg (str): The algorithm to use.
        problem (any): The problem.
        objectives (int): The number of objectives.
        oracle (Oracle): The oracle.
        linear_solver (LinearSolver): The linear solver.
        seed (int, optional): The seed for the random number generator. Defaults to None.

    Returns:
        OuterLoop: The outer loop.
    """
    if alg == '2D':
        if objectives != 2:
            raise ValueError('The 2D outer loop can only be used for 2D problems.')
        return Priol2D(problem, oracle, linear_solver)
    elif alg == 'PRIOL':
        return Priol(problem, objectives, oracle, linear_solver, seed=seed)
    else:
        raise ValueError(f'Unknown outer loop algorithm: {alg}')
