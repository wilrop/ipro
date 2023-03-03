import argparse

import numpy as np

from outer_loop import outer_loop
from pareto import p_prune
from vector_u import rectangle_u


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--objectives', type=int, default=2, help='The number of objectives.')
    parser.add_argument('--vecs', type=int, default=10, help='The number of vectors.')
    parser.add_argument('--low', type=int, default=0, help='The lower bound for the random integers.')
    parser.add_argument('--high', type=int, default=10, help='The upper bound for the random integers.')
    parser.add_argument('--tolerance', type=float, default=1e-5, help='The tolerance for the outer loop.')
    parser.add_argument('--seed', type=int, default=3, help='The seed for the random number generator.')
    parser.add_argument('--log_dir', type=str, default='runs', help='The directory to save the logs.')
    parser.add_argument('--save_figs', default=True, action='store_true', help='Whether to save the figures.')
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
    """Generate a random problem."""
    if rng is None:
        rng = np.random.default_rng()
    return rng.integers(low=low, high=high, size=(vecs, objectives))


class InnerLoop:
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
        best_vec = np.zeros(self.problem.shape[1])
        best_u = -np.inf
        for vec in self.problem:
            u = rectangle_u(vec, target, nadir)
            if u > best_u:
                best_u = u
                best_vec = vec
        return best_vec


def verify(pf, problem):
    pf = {tuple(vec) for vec in pf}
    candidates = {tuple(vec) for vec in problem}
    print(f'Candidates: {candidates}')
    print(f'Pareto front: {pf}')
    correct_pf = p_prune(candidates)
    print(f'Correct Pareto front: {correct_pf}')
    print(f'Correct: {pf == correct_pf}')


if __name__ == '__main__':
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    problem = generate_problem(objectives=args.objectives, vecs=args.vecs, low=args.low, high=args.high, rng=rng)
    inner_loop = InnerLoop(problem)
    pf = outer_loop(problem, inner_loop, linear_solver, tolerance=args.tolerance, save_figs=args.save_figs,
                    log_dir=args.log_dir)
    verify(pf, problem)
