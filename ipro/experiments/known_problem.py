import argparse

import numpy as np
from ipro.linear_solvers import init_linear_solver
from ipro.oracles import init_oracle
from ipro.outer_loops import init_outer_loop
from ipro.utils.pareto import extreme_prune


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--iterations', type=int, default=10, help='The number of iterations.')
    parser.add_argument('--objectives', type=int, default=2, help='The number of objectives.')
    parser.add_argument('--vecs', type=int, default=10, help='The number of vectors.')
    parser.add_argument('--low', type=int, default=0, help='The lower bound for the random integers.')
    parser.add_argument('--high', type=int, default=10, help='The upper bound for the random integers.')
    parser.add_argument('--aug', type=float, default=1e-3, help='The augmentation parameter.')
    parser.add_argument('--outer_loop', type=str, default='IPRO-2D', help='The outer loop to use.')
    parser.add_argument('--direction', type=str, default='maximize', help='The direction to optimize in.')
    parser.add_argument('--seed', type=int, default=1, help='The seed for the random number generator.')
    return parser.parse_args()


def generate_problem(objectives=2, vecs=10, low=0, high=10, rng=None):
    """Generate a random problem.

    Args:
        objectives (int, optional): The number of objectives. Defaults to 2.
        vecs (int, optional): The number of vectors. Defaults to 10.
        low (int, optional): The lower bound for the random integers. Defaults to 0.
        high (int, optional): The upper bound for the random integers. Defaults to 10.
        rng (np.random.Generator, optional): The random number generator. Defaults to None.

    Returns:
        np.ndarray: A random problem.
    """
    if rng is None:
        rng = np.random.default_rng()
    return rng.integers(low=low, high=high, size=(vecs, objectives))


if __name__ == '__main__':
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    sign = 1 if args.direction == 'maximize' else -1

    for i in range(args.iterations):
        print(f'Iteration {i}')

        problem = generate_problem(objectives=args.objectives, vecs=args.vecs, low=args.low, high=args.high, rng=rng)
        linear_solver = init_linear_solver('finite', problem, direction=args.direction)
        oracle = init_oracle('finite', problem, aug=args.aug, direction=args.direction)
        outer_loop = init_outer_loop(
            method=args.outer_loop,
            direction=args.direction,
            problem_id='known_problem',
            objectives=args.objectives,
            oracle=oracle,
            linear_solver=linear_solver,
            seed=args.seed,
            tolerance=0,
        )

        ps = outer_loop.solve()
        pf = outer_loop.pf
        correct_set = sign * extreme_prune(sign * problem)
        if {tuple(vec) for vec in pf} != {tuple(vec) for vec in correct_set}:
            print(f'Problem: {problem}')
            print(f'Correct set: {correct_set}')
            print(f'Obtained set: {ps}')
            print(f'Bounding box: {outer_loop.bounding_box}')
            break
        else:
            print(f'Verified!')
        print('-' * 40)
