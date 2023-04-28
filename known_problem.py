import argparse

import numpy as np
from linear_solvers import init_linear_solver
from oracles import init_oracle
from outer_loops import init_outer_loop
from utils.pareto import verify_pcs


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--iterations', type=int, default=100, help='The number of iterations.')
    parser.add_argument('--objectives', type=int, default=5, help='The number of objectives.')
    parser.add_argument('--vecs', type=int, default=100, help='The number of vectors.')
    parser.add_argument('--low', type=int, default=0, help='The lower bound for the random integers.')
    parser.add_argument('--high', type=int, default=10, help='The upper bound for the random integers.')
    parser.add_argument('--aug', type=float, default=0.01, help='The augmentation parameter.')
    parser.add_argument('--outer_loop', type=str, default='PRIOL', help='The outer loop to use.')
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

    for i in range(args.iterations):
        print(f'Iteration {i}')

        problem = generate_problem(objectives=args.objectives, vecs=args.vecs, low=args.low, high=args.high, rng=rng)
        linear_solver = init_linear_solver('finite', problem)
        oracle = init_oracle('finite', problem, aug=args.aug)
        outer_loop = init_outer_loop(alg=args.outer_loop, problem=problem, objectives=args.objectives, oracle=oracle,
                                     linear_solver=linear_solver, seed=args.seed)

        pf = outer_loop.solve()
        correct_set, is_correct = verify_pcs(problem, pf)
        missing_set = correct_set - pf

        if not is_correct:
            print(f'Problem: {problem}')
            print(f'Is correct: {is_correct}')
            print(f'Correct set: {correct_set}')
            print(f'Obtained set: {pf}')
            print(f'Number of missing elements: {len(missing_set)} - Difference : {missing_set}')
            print(f'Bounding box: {outer_loop.bounding_box}')
            print(f'Lower set: {outer_loop.lower_points}')

            for point in missing_set:
                strict_ok = np.any(np.all(np.array(point) > outer_loop.lower_points, axis=1))
                print(f'Point {point} strictly dominates some lower point: {strict_ok}')
