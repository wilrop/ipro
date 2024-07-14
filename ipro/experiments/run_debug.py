import argparse

import numpy as np
from ipro.linear_solvers import init_linear_solver
from ipro.oracles import init_oracle
from ipro.outer_loops import init_outer_loop
from ipro.utils.pareto import extreme_prune


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--outer_loop', type=str, default='IPRO-2D', help='The outer loop to use.')
    parser.add_argument('--aug', type=float, default=1e-3, help='The augmentation parameter.')
    parser.add_argument('--scale', type=float, default=1, help='The scale parameter.')
    parser.add_argument('--direction', type=str, default='maximize', help='The direction to optimize in.')
    parser.add_argument('--seed', type=int, default=1, help='The seed for the random number generator.')
    return parser.parse_args()


DEBUG_RUN = {
    'pareto_point_1': [0, 2.999],
    'pareto_point_2': [1, 1],
    'pareto_point_3': [2, 2],
    'pareto_point_4': [2, 2],
    'pareto_point_5': [2.5, 0.5],
}
DEBUG_OUTERS = np.array([
    [0, 3],
    [3, 0]
])

DEBUG_PROBLEM = np.concatenate([DEBUG_OUTERS, list(DEBUG_RUN.values())], axis=0)

if __name__ == '__main__':
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    sign = 1 if args.direction == 'maximize' else -1
    num_objectives = DEBUG_PROBLEM.shape[1]

    linear_solver = init_linear_solver('finite', DEBUG_OUTERS, direction=args.direction)
    oracle = init_oracle('debug', DEBUG_RUN, aug=args.aug, scale=args.scale, direction=args.direction)
    outer_loop = init_outer_loop(
        method=args.outer_loop,
        direction=args.direction,
        problem_id='debug',
        objectives=num_objectives,
        oracle=oracle,
        linear_solver=linear_solver,
        seed=args.seed,
        tolerance=0,
    )

    pf = outer_loop.solve()
    correct_set = sign * extreme_prune(sign * DEBUG_PROBLEM)
    if {tuple(vec) for vec in pf} != {tuple(vec) for vec in correct_set}:
        print(f'Problem: {DEBUG_PROBLEM}')
        print(f'Correct set: {correct_set}')
        print(f'Obtained set: {pf}')
        print(f'Bounding box: {outer_loop.bounding_box}')
