import argparse

import numpy as np

from outer_loop import OuterLoop
from outer_loop_2D import OuterLoop2D
from pareto import verify_pcs
from vector_u import create_batched_fast_translated_rectangle_u


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--objectives', type=int, default=3, help='The number of objectives.')
    parser.add_argument('--vecs', type=int, default=10, help='The number of vectors.')
    parser.add_argument('--low', type=int, default=0, help='The lower bound for the random integers.')
    parser.add_argument('--high', type=int, default=10, help='The upper bound for the random integers.')
    parser.add_argument('--outer_loop', type=str, default='ND', help='The outer loop to use.')
    parser.add_argument('--seed', type=int, default=6, help='The seed for the random number generator.')
    return parser.parse_args()


def linear_solver(problem, weights):
    """Find the utility maximising point.

    Args:
        problem (np.ndarray): The problem to solve.
        weights (np.ndarray): The weights to use.

    Returns:
        np.ndarray: The utility maximising point.
    """
    max_utility_index = np.argmax(np.dot(problem, weights))
    return problem[max_utility_index]


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


class InnerLoop:
    """A simple inner-loop method for the known problem setting."""

    def __init__(self, problem):
        self.problem = problem

    def solve(self, target, nadir):
        """The inner loop solver for the basic setting.

        Args:
            target (np.ndarray): The target vector.
            nadir (np.ndarray): The nadir vector.

        Returns:
            np.ndarray: The Pareto optimal vector.
        """
        u_f = create_batched_fast_translated_rectangle_u(target, nadir)
        utilities = u_f(self.problem)
        best_point = np.argmax(utilities)
        return self.problem[best_point], utilities[best_point]


if __name__ == '__main__':
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    problem = generate_problem(objectives=args.objectives, vecs=args.vecs, low=args.low, high=args.high, rng=rng)
    inner_loop = InnerLoop(problem)

    if args.outer_loop == '2D':
        if args.objectives != 2:
            raise ValueError('The 2D outer loop can only be used for 2D problems.')
        outer_loop = OuterLoop2D(problem, inner_loop, linear_solver)
    else:
        outer_loop = OuterLoop(problem, args.objectives, inner_loop, linear_solver, seed=args.seed)

    pf = outer_loop.solve()
    correct_set, is_correct = verify_pcs(problem, pf)
    missing_set = correct_set - pf
    print(f'Is correct: {is_correct}')
    print(f'Correct set: {correct_set}')
    print(f'Obtained set: {pf}')
    print(f'Number of missing elements: {len(missing_set)} - Difference : {missing_set}')
    print(f'Bounding box: {outer_loop.bounding_box}')

    for point in missing_set:
        for box in outer_loop.removed_boxes:
            if box.contains(point):
                print(f'Point {point} was removed by box {box}. Inner point? {box.contains_inner(point)}')
