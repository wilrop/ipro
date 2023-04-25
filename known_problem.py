import argparse

import numpy as np

from box_splitting_nd import BoxSplittingND
from priol_2D import Priol2D
from priol import Priol
from pareto import verify_pcs, extreme_prune
from vector_u import create_batched_aasf


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--iterations', type=int, default=100, help='The number of iterations.')
    parser.add_argument('--objectives', type=int, default=8, help='The number of objectives.')
    parser.add_argument('--vecs', type=int, default=100, help='The number of vectors.')
    parser.add_argument('--low', type=int, default=0, help='The lower bound for the random integers.')
    parser.add_argument('--high', type=int, default=10, help='The upper bound for the random integers.')
    parser.add_argument('--outer_loop', type=str, default='PRIOL', help='The outer loop to use.')
    parser.add_argument('--seed', type=int, default=1, help='The seed for the random number generator.')
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

    def __init__(self, problem, aug=0.01):
        self.problem = problem
        self.aug = aug

    def get_nadir(self):
        """Get the true nadir point.

        Note:
            This is purely used for testing purposes.

        Returns:
            np.ndarray: The true nadir point.
        """
        correct_pf = extreme_prune(np.copy(self.problem))
        return np.min(correct_pf, axis=0)

    def solve(self, referent, ideal):
        """The inner loop solver for the basic setting.

        Args:
            referent (np.ndarray): The reference vector.
            ideal (np.ndarray): The ideal vector.

        Returns:
            np.ndarray: The Pareto optimal vector.
        """
        u_f = create_batched_aasf(referent, referent, ideal, aug=self.aug)
        utilities = u_f(self.problem)
        best_point = np.argmax(utilities)
        return self.problem[best_point]


if __name__ == '__main__':
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    for i in range(args.iterations):
        print(f'Iteration {i}')

        problem = generate_problem(objectives=args.objectives, vecs=args.vecs, low=args.low, high=args.high, rng=rng)
        inner_loop = InnerLoop(problem)

        if args.outer_loop == '2D':
            if args.objectives != 2:
                raise ValueError('The 2D outer loop can only be used for 2D problems.')
            outer_loop = Priol2D(problem, inner_loop, linear_solver)
        elif args.outer_loop == 'PRIOL':
            outer_loop = Priol(problem, args.objectives, inner_loop, linear_solver, seed=args.seed)
        else:
            outer_loop = BoxSplittingND(problem, args.objectives, inner_loop, linear_solver, seed=args.seed)

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
