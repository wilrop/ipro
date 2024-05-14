import wandb
import json
from datetime import datetime
from itertools import zip_longest


def link_id_to_experiment(to_reproduce):
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


def link_best_runs(combos, num_seeds, cutoff_date=None, max_runs_per_config=1000):
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

    link_id_to_experiment(to_reproduce)


def link_given_runs(run_paths, num_seeds):
    to_reproduce = {'all': []}
    for run_path in run_paths:
        api = wandb.Api(timeout=120)
        run = api.run(run_path)
        alg_name = run.config['alg_name']
        env_id = run.config['env_id']
        to_reproduce['all'].extend([(alg_name, env_id, seed, run_path) for seed in range(num_seeds)])
    link_id_to_experiment(to_reproduce)


if __name__ == '__main__':
    combos = [
        ('deep-sea-treasure-concave-v0', 'SN-MO-DQN'),
        ('deep-sea-treasure-concave-v0', 'SN-MO-A2C'),
        ('deep-sea-treasure-concave-v0', 'SN-MO-PPO'),
    ]
    cutoff_date = "28/01/2024"
    num_seeds = 5
    #link_best_runs(combos, num_seeds, cutoff_date=cutoff_date)
    run_paths = [
        'wilrop/IPRO_dst_neurips/26d3gy5z',
        'wilrop/IPRO_dst_neurips/uic1iqm2',
        'wilrop/IPRO_dst_neurips/oiehwksm',
        'wilrop/IPRO_dst_neurips/34id3l3g',
        'wilrop/IPRO_dst_neurips/3l26mm8w',
        'wilrop/IPRO_dst_neurips/l2udoicf',
        'wilrop/IPRO_dst_neurips/47n4ughs',
        'wilrop/IPRO_dst_neurips/23okxmc6',
        'wilrop/IPRO_dst_neurips/3fq1k123',
        'wilrop/IPRO_dst_neurips/17dk1fpk',
        'wilrop/IPRO_dst_neurips/nvul5vam',
        'wilrop/IPRO_dst_neurips/2z7rtg42',
        'wilrop/IPRO_dst_neurips/2jdq6c2e',
        'wilrop/IPRO_dst_neurips/2da3ijti',
        'wilrop/IPRO_dst_neurips/1jooorls'
    ]
    link_given_runs(run_paths, num_seeds)
