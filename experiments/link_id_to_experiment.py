import wandb
import json
from datetime import datetime
from itertools import zip_longest


def link_id_to_experiment(combos, num_seeds, cutoff_date=None, max_runs_per_config=1000):
    """Generate a JSON of experiments to reproduce.

    Args:
        combos (list): The list of environment and algorithm combinations to reproduce.
        num_seeds (int): The number of seeds to reproduce.
        necessary_params (dict): The necessary parameters to reproduce.
        max_runs_per_config (int): The maximum number of runs to reproduce per configuration.
    """
    to_reproduce = {combo: [] for combo in combos}

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
            if (env_id, alg_name) in to_reproduce and len(to_reproduce[(env_id, alg_name)]) < max_runs_per_config:
                if cutoff_date is not None:
                    created_time = run.created_at
                    created_time_obj = datetime.strptime(created_time, '%Y-%m-%dT%H:%M:%S')
                    cutoff_date_obj = datetime.strptime(cutoff_date, '%d/%m/%Y')
                    if created_time_obj >= cutoff_date_obj:
                        to_reproduce[(env_id, alg_name)].append(run)
                else:
                    to_reproduce[(env_id, alg_name)].append(run)

    for idx, ((env_id, alg_name), runs) in enumerate(to_reproduce.items()):
        print(f'Processing {idx}/{len(to_reproduce)} - {alg_name} - {env_id}')
        sorted_runs = sorted(runs, key=lambda x: x.summary['outer/hypervolume'], reverse=True)
        sorted_runs = sorted_runs[:max_runs_per_config]
        sorted_run_paths = ['/'.join(run.path) for run in sorted_runs]
        reproduce_lst = []
        for run_path in sorted_run_paths:
            reproduce_lst.extend([(alg_name, env_id, seed, run_path) for seed in range(num_seeds)])
        to_reproduce[(env_id, alg_name)] = reproduce_lst

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
    combos = [
        ('deep-sea-treasure-concave-v0', 'SN-MO-DQN'),
        ('deep-sea-treasure-concave-v0', 'SN-MO-A2C'),
        ('deep-sea-treasure-concave-v0', 'SN-MO-PPO'),
    ]
    cutoff_date = "28/01/2024"
    num_seeds = 5
    link_id_to_experiment(combos, num_seeds, cutoff_date=cutoff_date)
