import os
import numpy as np
import pandas as pd
from utility_function.generate_utility_fns import load_utility_fns
from utility_function.utility_eval import generalised_expected_utility, generalised_maximum_utility_loss


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


def save_u_metrics(eu_df, mul_df, env_id, algorithm):
    # Save the pandas dataframes.
    metrics_dir = os.path.join('metrics', env_id)
    os.makedirs(metrics_dir, exist_ok=True)
    if algorithm in ['PCN', 'SN-MO-DQN', 'SN-MO-A2C', 'SN-MO-PPO']:
        name = f'{algorithm}_unprocessed'
    else:
        name = algorithm
    eu_df.to_csv(os.path.join(metrics_dir, f'{name}_eu.csv'), index=False)
    mul_df.to_csv(os.path.join(metrics_dir, f'{name}_mul.csv'), index=False)


def compute_and_save_all(environments, algorithms, seeds):
    for env_id in environments:
        u_path = os.path.join('metrics', 'utility_fns', 'increasing_cumsum', env_id)
        utility_fns = load_utility_fns(u_path)
        for algorithm in algorithms:
            print(f'Computing utility metrics for {env_id} - {algorithm} over {len(seeds)} seeds.')
            geu_data, mul_data = compute_u_metrics(utility_fns, env_id, algorithm, seeds)
            save_u_metrics(geu_data, mul_data, env_id, algorithm)
            print('--------')


def scale(data, min_val, max_val):
    return (data - min_val) / (max_val - min_val)


def compute_u_metrics(utility_fns, env_id, algorithm, seeds):
    env_dir = os.path.join('fronts', env_id)

    if env_id == 'deep-sea-treasure-concave-v0':
        joint_pf = DST_FRONT
    else:
        joint_pf = np.load(os.path.join(env_dir, 'joint_front.npy'))

    eu_data = []
    mul_data = []
    for seed in seeds:
        fronts_dir = os.path.join(env_dir, algorithm, str(seed))
        evals = []
        for f in os.listdir(fronts_dir):
            if 'front_' in f:
                step = int(f.split('_')[-1].split('.')[0])
                evals.append((step, f))
        evals = sorted(evals, key=lambda x: x[0])

        begin_front = np.full(joint_pf.shape[1:], -1e14)
        eu_min = generalised_expected_utility(begin_front, utility_fns)
        eu_max = generalised_expected_utility(joint_pf, utility_fns)
        mul_min = 0
        mul_max = generalised_maximum_utility_loss(begin_front, joint_pf, utility_fns)
        eu_data.append((0, seed, scale(eu_min, eu_min, eu_max)))
        mul_data.append((0, seed, scale(mul_max, mul_min, mul_max)))

        for step, eval_front_name in evals:
            eval_front_path = os.path.join(fronts_dir, eval_front_name)
            eval_front = np.load(eval_front_path)
            eu = generalised_expected_utility(eval_front, utility_fns)
            mul = generalised_maximum_utility_loss(eval_front, joint_pf, utility_fns)
            eu_data.append((step, seed, scale(eu, eu_min, eu_max)))
            mul_data.append((step, seed, scale(mul, mul_min, mul_max)))

    # make pandas dataframe.
    eu_data = pd.DataFrame(eu_data, columns=['Step', 'Seed', 'EU'])
    mul_data = pd.DataFrame(mul_data, columns=['Step', 'Seed', 'MUL'])
    return eu_data, mul_data


if __name__ == "__main__":
    environments = ['deep-sea-treasure-concave-v0', 'minecart-v0', 'mo-reacher-v4']
    algorithms = ['Envelope', 'GPI-LS', 'PCN', 'SN-MO-DQN', 'SN-MO-A2C', 'SN-MO-PPO']
    seeds = list(range(5))
    compute_and_save_all(environments, algorithms, seeds)
