import os
import numpy as np
import pandas as pd
from ipro.utility_function.generate_utility_fns import load_utility_fns
from ipro.utility_function.utility_eval import (
    generalised_expected_utility,
    generalised_maximum_utility_loss,
    generalised_expected_utility_loss
)


DST_FRONT = np.array([
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


def save_u_metrics(metric, df, env_id, algorithm):
    # Save the pandas dataframes.
    metrics_dir = os.path.join('metrics', env_id)
    os.makedirs(metrics_dir, exist_ok=True)
    if algorithm in ['PCN', 'SN-MO-DQN', 'SN-MO-A2C', 'SN-MO-PPO']:
        name = f'{algorithm}_unprocessed'
    else:
        name = algorithm
    df.to_csv(os.path.join(metrics_dir, f'{name}_{metric}.csv'), index=False)


def compute_and_save_all(metrics, environments, algorithms, seeds):
    number_of_metrics = len(metrics) * len(environments) * len(algorithms)
    i = 1
    for metric in metrics:
        for env_id in environments:
            u_path = os.path.join('metrics', 'utility_fns', 'increasing_cumsum', env_id)
            utility_fns = load_utility_fns(u_path)
            for algorithm in algorithms:
                print(f'PROCESSING {i}/{number_of_metrics}: {metric} - {env_id} - {algorithm}')
                df = compute_u_metrics(metric, utility_fns, env_id, algorithm, seeds)
                save_u_metrics(metric, df, env_id, algorithm)
                print('--------')
                i += 1


def scale(data, min_val, max_val):
    return (data - min_val) / (max_val - min_val)


def compute_u_metrics(metric, utility_fns, env_id, algorithm, seeds):
    env_dir = os.path.join('fronts', env_id)

    if env_id == 'deep-sea-treasure-concave-v0':
        joint_pf = DST_FRONT
    else:
        joint_pf = np.load(os.path.join(env_dir, 'joint_front.npy'))

    begin_front = np.full(joint_pf.shape[1:], -1e14)

    if metric == 'EU':
        eval_func = lambda x: generalised_expected_utility(x, utility_fns)
        metric_min = 0
        metric_max = eval_func(joint_pf)
    elif metric == 'MUL':
        eval_func = lambda x: generalised_maximum_utility_loss(x, joint_pf, utility_fns)
        metric_min = 0
        metric_max = eval_func(begin_front)
    elif metric == 'EUL':
        eval_func = lambda x: generalised_expected_utility_loss(x, joint_pf, utility_fns)
        metric_min = 0
        metric_max = eval_func(begin_front)
    else:
        raise ValueError(f'Unknown metric: {metric}')

    data = []

    for seed in seeds:
        fronts_dir = os.path.join(env_dir, algorithm, str(seed))
        evals = []
        for f in os.listdir(fronts_dir):
            if 'front_' in f:
                step = int(f.split('_')[-1].split('.')[0])
                evals.append((step, f))
        evals = sorted(evals, key=lambda x: x[0])

        start_metric = eval_func(begin_front)
        scaled_start_metric = scale(start_metric, metric_min, metric_max)
        data.append((0, seed, scaled_start_metric))

        for step, eval_front_name in evals:
            eval_front_path = os.path.join(fronts_dir, eval_front_name)
            eval_front = np.load(eval_front_path)
            res = eval_func(eval_front)
            scaled_res = scale(res, metric_min, metric_max)
            data.append((step, seed, scaled_res))

    df = pd.DataFrame(data, columns=['Step', 'Seed', metric])
    return df


if __name__ == "__main__":
    metrics = ['EU', 'MUL', 'EUL']
    environments = ['deep-sea-treasure-concave-v0', 'minecart-v0', 'mo-reacher-v4', 'mo-reacher-concave-v4']
    algorithms = ['Envelope', 'GPI-LS', 'PCN', 'SN-MO-DQN', 'SN-MO-A2C', 'SN-MO-PPO']
    seeds = list(range(5))
    compute_and_save_all(metrics, environments, algorithms, seeds)
