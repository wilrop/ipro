import os
import numpy as np
from utility_function.generate_utility_fns import load_utility_fns
from utility_function.utility_eval import generalised_expected_utility, generalised_maximum_utility_loss


def save_u_metrics(eu_data, mul_data, env_id, algorithm, seed):
    metrics_dir = os.path.join('metrics', env_id, algorithm, str(seed))
    os.makedirs(metrics_dir, exist_ok=True)
    np.save(os.path.join(metrics_dir, 'eu.npy'), eu_data)
    np.save(os.path.join(metrics_dir, 'mul.npy'), mul_data)


def compute_and_save_all(utility_fns, environments, algorithms, seeds):
    for env_id in environments:
        for algorithm in algorithms:
            for seed in seeds:
                print(f'Computing utility metrics for {env_id} - {algorithm} - {seed}')
                geu_data, mul_data = compute_u_metrics(utility_fns, env_id, algorithm, seed)
                save_u_metrics(geu_data, mul_data, env_id, algorithm, seed)
                print('--------')


def compute_u_metrics(utility_fns, env_id, algorithm, seed):
    env_dir = os.path.join('fronts', env_id)
    fronts_dir = os.path.join(env_dir, algorithm, str(seed))
    joint_pf = np.load(os.path.join(env_dir, 'joint_front.npy'))
    evals = sorted([f for f in os.listdir(fronts_dir) if 'front_' in f])
    eu_data = []
    mul_data = []
    for eval_front_name in evals:
        eval_front_path = os.path.join(fronts_dir, eval_front_name)
        eval_front = np.load(eval_front_path)
        eu = generalised_expected_utility(eval_front, utility_fns)
        mul = generalised_maximum_utility_loss(eval_front, joint_pf, utility_fns)
        eu_data.append(eu)
        mul_data.append(mul)
    return eu_data, mul_data


if __name__ == "__main__":
    utility_fns = load_utility_fns('../utility_function/utility_fns/increasing_cumsum/deep-sea-treasure-concave-v0')
    environments = ['deep-sea-treasure-concave-v0']
    algorithms = ['Envelope', 'GPI-LS', 'PCN', 'SN-MO-DQN']
    seeds = list(range(5))
    compute_and_save_all(utility_fns, environments, algorithms, seeds)
