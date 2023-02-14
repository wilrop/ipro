import argparse

import numpy as np

from geohunt import outer_loop
from pareto import p_prune
from vector_u import rectangle_u


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--objectives', type=int, default=2)
    parser.add_argument('--vecs', type=int, default=10)
    parser.add_argument('--low', type=int, default=0)
    parser.add_argument('--high', type=int, default=10)
    parser.add_argument('--seed', type=int, default=3)
    parser.add_argument('--log_dir', type=str, default='runs')
    parser.add_argument('--save_figs', default=True, action='store_true')
    return parser.parse_args()


def linear_solver(problem, weights):
    """Find the utility maximising point.

    Args:
        problem (np.ndarray): The problem to solve.
        weights (np.ndarray): The weights to use.

    Returns:
        np.ndarray: The utility maximising point.
    """
    utilities = problem @ weights
    max_utility = np.max(utilities)
    best_vecs = problem[utilities == max_utility]
    best_vec = np.array(p_prune({tuple(vec) for vec in best_vecs}).pop())
    return best_vec


def generate_problem(objectives=2, vecs=10, low=0, high=10, rng=None):
    """Generate a random problem."""
    if rng is None:
        rng = np.random.default_rng()
    return rng.integers(low=low, high=high, size=(vecs, objectives))


def inner_loop(problem, target):
    """The inner loop solver for the basic setting.

    Args:
        problem (np.ndarray): The problem to solve.
        target (np.ndarray): The target vector.

    Returns:
        np.ndarray: The Pareto optimal vector.
    """
    best_vec = np.zeros(problem.shape[1])
    best_u = -np.inf
    for vec in problem:
        u = rectangle_u(vec, target)
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
    pf = outer_loop(problem, inner_loop, linear_solver, num_objectives=args.objectives, save_figs=args.save_figs,
                    log_dir=args.log_dir)
    verify(pf, problem)
