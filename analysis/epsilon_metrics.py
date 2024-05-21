import os
import numpy as np
from utils.pareto import extreme_prune

DST_PF = np.array([
    [1.0, -1.0],
    [2.0, -3.0],
    [3.0, -5.0],
    [5.0, -7.0],
    [8.0, -8.0],
    [16.0, -9.0],
    [24.0, -13.0],
    [50.0, -14.0],
    [74.0, -17.0],
    [124.0, -19.0]
])

ADD_TABLE = """
\caption{{The minimum $\\varepsilon$ shift necessary to obtain any undiscovered Pareto optimal solution. The best mean is in bold.}}
\label{{tab:approx-quality}}
\\centering
\\begin{{small}}
\\begin{{sc}}
\\begin{{tabular}}{{llc}}
\\toprule
Env & Algorithm & $\\varepsilon$ \\\\
\midrule
& IPRO (PPO)    & ${}$ \\\\
& IPRO (A2C)    & ${}$ \\\\
DST & IPRO (DQN)    & ${}$ \\\\
& PCN           & ${}$ \\\\
& GPI-LS        & ${}$ \\\\
& Envelope      & ${}$ \\\\
\midrule
& IPRO (PPO)    & ${}$ \\\\
& IPRO (A2C)    & ${}$ \\\\
Minecart & IPRO (DQN)    & ${}$ \\\\
& PCN           & ${}$ \\\\
& GPI-LS        & ${}$ \\\\
& Envelope      & ${}$ \\\\
\midrule
& IPRO (PPO)    & ${}$ \\\\
& IPRO (A2C)    & ${}$ \\\\
MO-Reacher & IPRO (DQN)    & ${}$ \\\\
& PCN           & ${}$ \\\\
& GPI-LS        & ${}$ \\\\
& Envelope      & ${}$ \\\\
\\bottomrule
\end{{tabular}}
\end{{sc}}
\end{{small}}
"""


def load_data(set_path):
    front = np.load(set_path)
    data = np.around(front, decimals=5)
    data = np.copy(extreme_prune(data))
    return data


def load_exp_data(env_id, alg, num_seeds=5):
    """Read the data for each individual algorithm."""
    datasets = []
    for seed in range(num_seeds):
        front_path = os.path.join('fronts', env_id, alg, str(seed), 'final_front.npy')
        data = load_data(front_path)
        datasets.append(data)
    return datasets


def load_joint_data(env_id):
    if env_id == 'deep-sea-treasure-concave-v0':
        complete_set = DST_PF
    else:
        complete_path = os.path.join('fronts', env_id, 'joint_front.npy')
        complete_set = load_data(complete_path)
    print(f'Complete set size: {len(complete_set)}')
    return complete_set


def additive_epsilon(complete_set, eval_set):
    complete_set = np.copy(complete_set)
    eval_set = np.copy(eval_set)
    num_objectives = eval_set.shape[-1]
    epsilon = 0
    for vec in complete_set:
        diffs = vec - eval_set
        maxes = np.max(diffs, axis=1)
        epsilon = max(epsilon, np.min(maxes))

    for vec in complete_set:
        adjusted = eval_set + epsilon + 1e-6
        checks = vec <= adjusted
        check_sums = np.sum(checks, axis=1)
        idxes = np.argwhere(check_sums == num_objectives)
        assert len(idxes) > 0
        assert np.min(adjusted[idxes] - vec) >= 0
    return epsilon


def multiplicative_epsilon(complete_set, eval_set):
    complete_set = np.copy(complete_set)
    eval_set = np.copy(eval_set)

    epsilon = 0
    num_objectives = eval_set.shape[-1]
    eval_set[np.isclose(eval_set, 0)] = 1e-6
    complete_set[np.isclose(complete_set, 0)] = 1e-6
    non_neg_complete_set = np.copy(complete_set)
    neg_indices = non_neg_complete_set < 0
    inverted = -1 / non_neg_complete_set
    non_neg_complete_set = inverted * neg_indices + non_neg_complete_set * (1 - neg_indices)
    non_neg_eval_set = np.copy(eval_set)
    neg_indices = non_neg_eval_set < 0
    inverted = -1 / non_neg_eval_set
    non_neg_eval_set = inverted * neg_indices + non_neg_eval_set * (1 - neg_indices)

    for vec in non_neg_complete_set:
        diffs = vec - non_neg_eval_set
        fracs = diffs / non_neg_eval_set
        maxes = np.max(fracs, axis=1)
        epsilon = max(epsilon, np.min(maxes))

    for vec in non_neg_complete_set:
        adjusted = non_neg_eval_set * (1 + epsilon) + 1e-6
        checks = vec <= adjusted
        check_sums = np.sum(checks, axis=1)
        idxes = np.argwhere(check_sums == num_objectives)
        assert len(idxes) > 0
        assert np.min(adjusted[idxes] - vec) >= 0
    return epsilon


def compose_add_table(env_ids, algs):
    add_str_results = []  # Will contain strings to complete the ADD_TABLE.

    for env_id in env_ids:  # Loop over all environments.
        complete_set = load_joint_data(env_id)
        env_means = []
        env_stds = []

        for alg in algs:
            alg_datasets = load_exp_data(env_id, alg)  # Loads the data for all seeds in this experiment.
            alg_epsilons = []

            for dataset in alg_datasets:
                add_eps = additive_epsilon(complete_set, dataset)
                alg_epsilons.append(add_eps)

            mean_eps = np.mean(alg_epsilons)
            std_eps = np.std(alg_epsilons)
            env_means.append(mean_eps)
            env_stds.append(std_eps)
            print_str = f'{env_id} | {alg}'
            print_str += f' | Additive {mean_eps} ({std_eps})'
            print(print_str)

        env_means = np.round(env_means, 2)
        env_stds = np.round(env_stds, 2)
        best_eps = np.min(env_means)

        for mean, std in zip(env_means, env_stds):
            mean_str = f'{mean}'
            if mean == best_eps:
                mean_str = f'\\bm{{{mean_str}}}'
            add_res_str = f'{mean_str} \\pm {std}'
            add_str_results.append(add_res_str)

    table = ADD_TABLE.format(*add_str_results)
    return table


if __name__ == '__main__':
    algs = ['SN-MO-PPO', 'SN-MO-A2C', 'SN-MO-DQN', 'PCN', 'GPI-LS', 'Envelope']
    env_ids = ['deep-sea-treasure-concave-v0', 'minecart-v0', 'mo-reacher-v4']
    add_table = compose_add_table(env_ids, algs)
    print(add_table)
