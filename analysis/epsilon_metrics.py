import pandas as pd
import numpy as np
from utils.pareto import extreme_prune

dst_pf = np.array([[1.0, -1.0],
                   [2.0, -3.0],
                   [3.0, -5.0],
                   [5.0, -7.0],
                   [8.0, -8.0],
                   [16.0, -9.0],
                   [24.0, -13.0],
                   [50.0, -14.0],
                   [74.0, -17.0],
                   [124.0, -19.0]])


def read_data(env_id, alg, num_seeds=5, round=5):
    """Read the data for each individual algorithm."""
    datasets = []
    for seed in range(num_seeds):
        data = np.around(pd.read_csv(f'fronts/{env_id}/{alg}_{seed}.csv').to_numpy(dtype=np.float32), decimals=round)
        data = np.copy(extreme_prune(data))
        datasets.append(data)
    return datasets


def compose_complete_set(env_id, datasets, round=5):
    if env_id == 'deep-sea-treasure-concave-v0':
        complete_set = dst_pf
    else:
        all_fronts = []
        for dataset in datasets:
            all_fronts.extend(dataset)
        complete_set = np.around(np.copy(np.concatenate(all_fronts, axis=0)), decimals=round)
        complete_set = np.copy(extreme_prune(complete_set))
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


if __name__ == '__main__':
    algs = ['SN-MO-PPO', 'SN-MO-A2C', 'SN-MO-DQN', 'PCN', 'GPI-LS', 'Envelope']
    env_ids = ['deep-sea-treasure-concave-v0', 'minecart-v0', 'mo-reacher-v4']

    add_mul_table = """
\\begin{{table}}[t]
\caption{{The additive and multiplicative $\\varepsilon$ indicator. The highest means are highlighted in bold.}}
\label{{tab:epsilon-indicator}}
\\vskip 0.15in
\\begin{{center}}
\\begin{{small}}
\\begin{{sc}}
\\begin{{tabular}}{{lccc}}
\\toprule
Algorithm & $\\varepsilon^+$ & $\\varepsilon^\\times$ \\\\
\midrule
PRO (PPO)    & ${}$ & ${}$ \\\\
IPRO (A2C)    & ${}$ & ${}$ \\\\
IPRO (DQN)    & ${}$ & ${}$ \\\\
PCN           & ${}$ & ${}$ \\\\
GPI-LS        & ${}$ & ${}$ \\\\
%Envelope      & ${}$ & ${}$ \\\\
\midrule
IPRO (PPO)    & ${}$ & ${}$ \\\\
IPRO (A2C)    & ${}$ & ${}$ \\\\
IPRO (DQN)    & ${}$ & ${}$ \\\\
PCN           & ${}$ & ${}$ \\\\
GPI-LS        & ${}$ & ${}$ \\\\
%Envelope      & ${}$ & ${}$ \\\\
\midrule
IPRO (PPO)    & ${}$ & ${}$ \\\\
IPRO (A2C)    & ${}$ & ${}$ \\\\
IPRO (DQN)    & ${}$ & ${}$ \\\\
PCN           & ${}$ & ${}$ \\\\
GPI-LS        & ${}$ & ${}$ \\\\
%Envelope      & ${}$ & ${}$ \\\\
\\bottomrule
\end{{tabular}}
\end{{sc}}
\end{{small}}
\end{{center}}
\\vskip -0.1in
\end{{table}}
    """

    add_table = """
\\begin{{table}}[t]
\caption{{The additive and multiplicative $\\varepsilon$ indicator. The highest means are highlighted in bold.}}
\label{{tab:epsilon-indicator}}
\\vskip 0.15in
\\begin{{center}}
\\begin{{small}}
\\begin{{sc}}
\\begin{{tabular}}{{llc}}
\\toprule
Env & Algorithm & $\\varepsilon^+$ \\\\
\midrule
DST & PRO (PPO)    & ${}$ \\\\
DST & IPRO (A2C)    & ${}$ \\\\
DST & IPRO (DQN)    & ${}$ \\\\
DST & PCN           & ${}$ \\\\
DST & GPI-LS        & ${}$ \\\\
%Envelope      & ${}$ \\\\
\midrule
Minecart & IPRO (PPO)    & ${}$ \\\\
Minecart & IPRO (A2C)    & ${}$ \\\\
Minecart & IPRO (DQN)    & ${}$ \\\\
Minecart & PCN           & ${}$ \\\\
Minecart & GPI-LS        & ${}$ \\\\
%Envelope      & ${}$ \\\\
\midrule
MO-Reacher & IPRO (PPO)    & ${}$ \\\\
MO-Reacher & IPRO (A2C)    & ${}$ \\\\
MO-Reacher & IPRO (DQN)    & ${}$ \\\\
MO-Reacher & PCN           & ${}$ \\\\
MO-Reacher & GPI-LS        & ${}$ \\\\
%Envelope      & ${}$ \\\\
\\bottomrule
\end{{tabular}}
\end{{sc}}
\end{{small}}
\end{{center}}
\\vskip -0.1in
\end{{table}}
"""

    all_datasets = {}
    for env_id in env_ids:
        all_datasets[env_id] = {}
        for alg in algs:
            all_datasets[env_id][alg] = read_data(env_id, alg)

    dataset_results = {env_id: {alg: {'add': [], 'mul': []} for alg in algs} for env_id in env_ids}
    add_results = {env_id: {alg: {'mean': 0, 'std': 0} for alg in algs} for env_id in env_ids}
    mul_results = {env_id: {alg: {'mean': 0, 'std': 0} for alg in algs} for env_id in env_ids}
    add_mul_str_results = []
    add_str_results = []

    for env_id in env_ids:
        env_datasets = all_datasets[env_id]
        complete_set = compose_complete_set(env_id, env_datasets.values())
        for alg in algs:
            alg_datasets = env_datasets[alg]
            for dataset in alg_datasets:
                dataset_results[env_id][alg]['add'].append(additive_epsilon(complete_set, dataset))
                dataset_results[env_id][alg]['mul'].append(multiplicative_epsilon(complete_set, dataset))

            additive_res = dataset_results[env_id][alg]['add']
            multiplicative_res = dataset_results[env_id][alg]['mul']
            mean_add_eps = np.mean(additive_res)
            std_add_eps = np.std(additive_res)
            mean_mul_eps = np.mean(multiplicative_res)
            std_mul_eps = np.std(multiplicative_res)
            print_str = f'{env_id} | {alg}'
            print_str += f' | Additive {mean_add_eps} ({std_add_eps})'
            print_str += f' | Multiplicative {mean_mul_eps} ({std_mul_eps})'
            print(print_str)
            add_results[env_id][alg]['mean'] = mean_add_eps
            add_results[env_id][alg]['std'] = std_add_eps
            mul_results[env_id][alg]['mean'] = mean_mul_eps
            mul_results[env_id][alg]['std'] = std_mul_eps

        additive_means = np.array([add_results[env_id][alg]['mean'] for alg in algs])
        additive_stds = np.array([add_results[env_id][alg]['std'] for alg in algs])
        multiplicative_means = np.array([mul_results[env_id][alg]['mean'] for alg in algs])
        multiplicative_stds = np.array([mul_results[env_id][alg]['std'] for alg in algs])

        best_add = np.min(additive_means)
        best_add_idx = np.argwhere(additive_means == best_add)
        adjusted_add_means = additive_means - additive_stds
        best_add_indices = np.argwhere(adjusted_add_means <= best_add)

        best_mul = np.min(multiplicative_means)
        best_mul_idx = np.argwhere(multiplicative_means == best_mul)
        adjusted_mul_means = multiplicative_means - multiplicative_stds
        best_mul_indices = np.argwhere(adjusted_mul_means <= best_mul)

        for idx, alg in enumerate(algs):
            add_res_str = f'{add_results[env_id][alg]["mean"]:.2f} \\pm {add_results[env_id][alg]["std"]:.2f}'
            mul_res_str = f'{mul_results[env_id][alg]["mean"]:.2f} \\pm {mul_results[env_id][alg]["std"]:.2f}'
            if idx in best_add_idx:
                add_res_str = f'\\bm{{{add_res_str}}}'
            if idx in best_mul_idx:
                mul_res_str = f'\\bm{{{mul_res_str}}}'

            add_mul_str_results.extend([add_res_str, mul_res_str])
            add_str_results.append(add_res_str)

    add_mul_table = add_mul_table.format(*add_mul_str_results)
    add_table = add_table.format(*add_str_results)
    print(add_mul_table)
    print('-------')
    print(add_table)
