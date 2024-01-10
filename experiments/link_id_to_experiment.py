import wandb
import json
from itertools import zip_longest


def link_id_to_experiment(algs, envs, num_seeds, max_runs_per_config=100):
    """Generate a JSON of experiments to reproduce.

    Args:
        algs (list): The list of algorithms to reproduce.
        envs (list): The list of environments to reproduce.
        num_seeds (int): The number of seeds to reproduce.
        max_runs_per_config (int): The maximum number of runs to reproduce per configuration.
    """
    to_reproduce = {(alg_name, env_id): [] for env_id in envs for alg_name in algs}

    # Collect the possible runs for this task.
    api = wandb.Api(timeout=120)
    runs = api.runs("wilrop/IPRO_opt", order='-summary_metrics.outer/hypervolume')

    print(f'Collecting runs')
    for run in runs:
        if all([len(runs) >= max_runs_per_config for runs in to_reproduce.values()]):
            break
        if 'reproduced' not in run.summary.keys() or not run.summary['reproduced']:
            alg_name = run.config['alg_name']
            env_id = run.config['env_id']
            if alg_name in algs and env_id in envs and len(to_reproduce[(alg_name, env_id)]) < max_runs_per_config:
                to_reproduce[(alg_name, env_id)].append(run)

    for idx, ((alg_name, env_id), runs) in enumerate(to_reproduce.items()):
        print(f'Processing {idx}/{len(to_reproduce)} - {alg_name} - {env_id}')
        sorted_runs = sorted(runs, key=lambda x: x.summary['outer/hypervolume'], reverse=True)
        sorted_runs = sorted_runs[:max_runs_per_config]
        sorted_run_paths = ['/'.join(run.path) for run in sorted_runs]
        reproduce_lst = []
        for run_path in sorted_run_paths:
            reproduce_lst.extend([(alg_name, env_id, seed, run_path) for seed in range(num_seeds)])
        to_reproduce[(alg_name, env_id)] = reproduce_lst

    print(f'Linking IDs')
    idx = 0
    id_exp = {}
    for runs in zip_longest(*to_reproduce.values()):
        for run in runs:
            if run is not None:
                idx += 1
                id_exp[idx] = run

    json.dump(id_exp, open('evaluation/experiments.json', 'w'))
    print(f'Number of experiments: {idx}')


if __name__ == '__main__':
    algs = ['SN-MO-DQN', 'SN-MO-A2C', 'SN-MO-PPO']
    envs = ['deep-sea-treasure-concave-v0', 'minecart-v0', 'mo-reacher-v4']
    num_seeds = 5
    link_id_to_experiment(algs, envs, num_seeds)
