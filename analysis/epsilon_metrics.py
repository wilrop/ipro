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
        data = np.around(pd.read_csv(f'fronts/{env_id}/{alg}_{seed}.csv').to_numpy(), decimals=round)
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
    num_objectives = eval_set.shape[-1]
    epsilon = -1
    for vec in complete_set:
        diffs = vec - eval_set
        maxes = np.max(diffs, axis=1)
        epsilon = max(epsilon, np.min(maxes))

    for vec in complete_set:
        checks = vec <= eval_set + epsilon + 1e-6
        check_sums = np.sum(checks, axis=1)
        assert np.any(check_sums == num_objectives)
    return epsilon


def multiplicative_epsilon(complete_set, eval_set):
    epsilon = 0
    num_objectives = eval_set.shape[-1]
    zeros = eval_set == 0
    eval_set[zeros] = 1e-6
    zeros = complete_set == 0
    complete_set[zeros] = 1e-6
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
        checks = vec <= non_neg_eval_set * (1 + epsilon) + 1e-6
        check_sums = np.sum(checks, axis=1)
        assert np.any(check_sums == num_objectives)
    return epsilon


if __name__ == '__main__':
    algs = ['SN-MO-PPO', 'SN-MO-A2C', 'SN-MO-DQN', 'PCN', 'GPI-LS', 'Envelope']
    env_ids = ['deep-sea-treasure-concave-v0', 'minecart-v0', 'mo-reacher-v4']

    table = """
\\begin{{table}}[t]
\caption{{The additive and multiplicative $\\varepsilon$ indicator. The highest means are highlighted in bold.}}
\label{{tab:epsilon-indicator}}
\\vskip 0.15in
\\begin{{center}}
\\begin{{small}}
\\begin{{sc}}
\\begin{{tabular}}{{lcccr}}
\\toprule
Algorithm & Additive & Multiplicative \\\\
\midrule
IPRO (PPO)    & ${} \pm {}$ & ${} \pm {}$ \\\\
IPRO (A2C)    & ${} \pm {}$ & ${} \pm {}$ \\\\
IPRO (DQN)    & ${} \pm {}$ & ${} \pm {}$ \\\\
PCN           & ${} \pm {}$ & ${} \pm {}$ \\\\
GPI-LS        & ${} \pm {}$ & ${} \pm {}$ \\\\
%Envelope      & ${} \pm {}$ & ${} \pm {}$ \\\\
\midrule
IPRO (PPO)    & ${} \pm {}$ & ${} \pm {}$ \\\\
IPRO (A2C)    & ${} \pm {}$ & ${} \pm {}$ \\\\
IPRO (DQN)    & ${} \pm {}$ & ${} \pm {}$ \\\\
PCN           & ${} \pm {}$ & ${} \pm {}$ \\\\
GPI-LS        & ${} \pm {}$ & ${} \pm {}$ \\\\
%Envelope      & ${} \pm {}$ & ${} \pm {}$ \\\\
\midrule
IPRO (PPO)    & ${} \pm {}$ & ${} \pm {}$ \\\\
IPRO (A2C)    & ${} \pm {}$ & ${} \pm {}$ \\\\
IPRO (DQN)    & ${} \pm {}$ & ${} \pm {}$ \\\\
PCN           & ${} \pm {}$ & ${} \pm {}$ \\\\
GPI-LS        & ${} \pm {}$ & ${} \pm {}$ \\\\
%Envelope      & ${} \pm {}$ & ${} \pm {}$ \\\\
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

    additive_results = {env_id: {alg: [] for alg in algs} for env_id in env_ids}
    multiplicative_results = {env_id: {alg: [] for alg in algs} for env_id in env_ids}
    str_results = []

    for env_id in env_ids:
        env_datasets = all_datasets[env_id]
        complete_set = compose_complete_set(env_id, env_datasets.values())
        for alg in algs:
            alg_datasets = env_datasets[alg]
            for dataset in alg_datasets:
                additive_results[env_id][alg].append(additive_epsilon(complete_set, dataset))
                multiplicative_results[env_id][alg].append(multiplicative_epsilon(complete_set, dataset))

            additive_res = additive_results[env_id][alg]
            multiplicative_res = multiplicative_results[env_id][alg]
            mean_add_eps = np.mean(additive_res)
            std_add_eps = np.std(additive_res)
            mean_mul_eps = np.mean(multiplicative_res)
            std_mul_eps = np.std(multiplicative_res)
            print_str = f'{env_id} | {alg}'
            print_str += f' | Additive {mean_add_eps} ({std_add_eps})'
            print_str += f' | Multiplicative {mean_mul_eps} ({std_mul_eps})'
            print(print_str)
            additive_results[env_id][alg] = (mean_add_eps, std_add_eps)
            multiplicative_results[env_id][alg] = (mean_mul_eps, std_mul_eps)

        best_additive_idx = np.argmin([res[0] for res in additive_results[env_id].values()])
        best_additive_alg = algs[best_additive_idx]
        best_multiplicative_idx = np.argmin([res[0] for res in multiplicative_results[env_id].values()])
        best_multiplicative_alg = algs[best_multiplicative_idx]

        for idx, alg in enumerate(algs):
            for metric, metric_dict, best_idx in zip(['additive', 'multiplicative'],
                                                     [additive_results[env_id], multiplicative_results[env_id]],
                                                     [best_additive_idx, best_multiplicative_idx]):
                if idx == best_idx:
                    str_results.append(f'\\bm{{{metric_dict[alg][0]:.2f}}}')
                else:
                    str_results.append(f'{metric_dict[alg][0]:.2f}')
                str_results.append(f'{metric_dict[alg][1]:.2f}')

    table = table.format(*str_results)
    print(table)
