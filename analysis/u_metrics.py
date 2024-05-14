import os
import numpy as np
from utility_function.utility_eval import generalised_expected_utility, generalised_maximum_utility_loss


def compute_and_save_all():
    environments = []
    algorithms = []
    seeds = []
    utility_fns = []
    for env in environments:
        for algorithm in algorithms:
            for seed in seeds:
                geu_data, mul_data = compute_exp_geu_data(utility_fns, env, algorithm, seed)


def compute_exp_geu_data(utility_fns, env, algorithm, seed):
    eval_dir = os.path.join('artifacts', env, algorithm)
    seed_dir = os.path.join(eval_dir, str(seed))
    joint_pf = np.load(os.path.join(eval_dir, 'joint_pf.npy'))
    evals = [int(f) for f in os.listdir(seed_dir) if os.path.isdir(f)]
    eu_data = []
    mul_data = []
    for eval in evals:
        eval_dir = os.path.join(seed_dir, str(eval))
        eval_front = np.load(os.path.join(eval_dir, 'front.npy'))
        eu = generalised_expected_utility(eval_front, utility_fns)
        mul = generalised_maximum_utility_loss(eval_front, joint_pf, utility_fns)
        eu_data.append(eu)
        mul_data.append(mul)
    return eu_data, mul_data


if __name__ == "__main__":
    compute_all()
