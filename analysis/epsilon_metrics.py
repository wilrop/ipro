import pandas as pd
import numpy as np
from collections import defaultdict
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


def read_data(env_id, alg, num_seeds=5):
    """Read the data for each individual algorithm."""
    datasets = []
    for seed in range(num_seeds):
        data = pd.read_csv(f'fronts/{env_id}/{alg}_{seed}.csv').to_numpy()
        datasets.append(data)
    return datasets


def compose_complete_set(env_id, datasets):
    if env_id == 'deep-sea-treasure-concave-v0':
        complete_set = dst_pf
    else:
        all_fronts = []
        for dataset in datasets:
            all_fronts.extend(dataset)
        complete_set = np.copy(np.concatenate(all_fronts, axis=0))
        complete_set = np.copy(extreme_prune(complete_set))
    print(f'Complete set size: {len(complete_set)}')
    return complete_set


def additive_epsilon(complete_set, eval_set):
    epsilon = -1
    for vec in complete_set:
        diffs = vec - eval_set
        maxes = np.max(diffs, axis=1)
        epsilon = max(epsilon, np.min(maxes))
    return epsilon


def multiplicative_epsilon(complete_set, eval_set):
    epsilon = 0
    zeros = eval_set == 0
    eval_set[zeros] = 1e-6
    for vec in complete_set:
        negatives = eval_set < 0
        diffs = vec - eval_set
        fracs = diffs / eval_set
        fracs = np.nan_to_num(fracs, nan=np.inf)
        fracs[negatives] *= -1
        maxes = np.max(fracs, axis=1)
        epsilon = max(epsilon, np.min(maxes))
    return epsilon


if __name__ == '__main__':
    algs = ['SN-MO-PPO', 'SN-MO-A2C', 'SN-MO-DQN', 'PCN', 'GPI-LS', 'Envelope']
    env_ids = ['deep-sea-treasure-concave-v0', 'minecart-v0', 'mo-reacher-v4']

    table = """
\\begin{{table}}[t]
\caption{{Additive and multiplicative indicator.}}
\label{{tab:epsilon-indicator}}
\\vskip 0.15in
\\begin{{center}}
\\begin{{small}}
\\begin{{sc}}
\\begin{{tabular}}{{lcccr}}
\\toprule
Algorithm & Additive & Multiplicative \\\\
\midrule
IPRO (PPO)    & ${:.2f} \pm {:.2f}$ & ${:.2f} \pm {:.2f}$ \\\\
IPRO (A2C)    & ${:.2f} \pm {:.2f}$ & ${:.2f} \pm {:.2f}$ \\\\
IPRO (DQN)    & ${:.2f} \pm {:.2f}$ & ${:.2f} \pm {:.2f}$ \\\\
PCN           & ${:.2f} \pm {:.2f}$ & ${:.2f} \pm {:.2f}$ \\\\
GPI-LS        & ${:.2f} \pm {:.2f}$ & ${:.2f} \pm {:.2f}$ \\\\
Envelope      & ${:.2f} \pm {:.2f}$ & ${:.2f} \pm {:.2f}$ \\\\
\midrule
IPRO (PPO)    & ${:.2f} \pm {:.2f}$ & ${:.2f} \pm {:.2f}$ \\\\
IPRO (A2C)    & ${:.2f} \pm {:.2f}$ & ${:.2f} \pm {:.2f}$ \\\\
IPRO (DQN)    & ${:.2f} \pm {:.2f}$ & ${:.2f} \pm {:.2f}$ \\\\
PCN           & ${:.2f} \pm {:.2f}$ & ${:.2f} \pm {:.2f}$ \\\\
GPI-LS        & ${:.2f} \pm {:.2f}$ & ${:.2f} \pm {:.2f}$ \\\\
Envelope      & ${:.2f} \pm {:.2f}$ & ${:.2f} \pm {:.2f}$ \\\\
\midrule
IPRO (PPO)    & ${:.2f} \pm {:.2f}$ & ${:.2f} \pm {:.2f}$ \\\\
IPRO (A2C)    & ${:.2f} \pm {:.2f}$ & ${:.2f} \pm {:.2f}$ \\\\
IPRO (DQN)    & ${:.2f} \pm {:.2f}$ & ${:.2f} \pm {:.2f}$ \\\\
PCN           & ${:.2f} \pm {:.2f}$ & ${:.2f} \pm {:.2f}$ \\\\
GPI-LS        & ${:.2f} \pm {:.2f}$ & ${:.2f} \pm {:.2f}$ \\\\
Envelope      & ${:.2f} \pm {:.2f}$ & ${:.2f} \pm {:.2f}$ \\\\
\\bottomrule
\end{{tabular}}
\end{{sc}}
\end{{small}}
\end{{center}}
\\vskip -0.1in
\end{{table}}
    """

    datasets = {}
    results = []
    for env_id in env_ids:
        datasets[env_id] = {}
        for alg in algs:
            datasets[env_id][alg] = read_data(env_id, alg)

    additive_results = {env_id: {alg: defaultdict(list) for alg in algs} for env_id in env_ids}
    multiplicative_results = {env_id: {alg: defaultdict(list) for alg in algs} for env_id in env_ids}

    for env_id, alg_datasets in datasets.items():
        complete_set = compose_complete_set(env_id, alg_datasets.values())
        for alg, datasets in alg_datasets.items():
            for dataset in datasets:
                additive_results[env_id][alg]['add_eps'].append(additive_epsilon(complete_set, dataset))
                multiplicative_results[env_id][alg]['mul_eps'].append(multiplicative_epsilon(complete_set, dataset))

            additive_res = additive_results[env_id][alg]
            multiplicative_res = multiplicative_results[env_id][alg]
            mean_add_eps = np.mean(additive_res['add_eps'])
            std_add_eps = np.std(additive_res['add_eps'])
            mean_mul_eps = np.mean(multiplicative_res['mul_eps'])
            std_mul_eps = np.std(multiplicative_res['mul_eps'])
            print_str = f'{env_id} | {alg}'
            print_str += f' | Additive {mean_add_eps} ({std_add_eps})'
            print_str += f' | Multiplicative {mean_mul_eps} ({std_mul_eps})'
            print(print_str)
            results.extend([mean_add_eps, std_add_eps, mean_mul_eps, std_mul_eps])

    table = table.format(*results)
    print(table)

