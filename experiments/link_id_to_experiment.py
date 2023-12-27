import os
import json


def link_experiments(num_seeds=5):
    """Creates a json file that links an experiment id to a hyperparameter file and seed"""
    checked = {
        'dqn': {
            'deep-sea-treasure-concave-v0': [],
            'minecart-v0': [],
            'mo-reacher-v4': []
        },
        'a2c': {
            'deep-sea-treasure-concave-v0': [],
            'minecart-v0': [],
            'mo-reacher-v4': []
        },
        'ppo': {
            'deep-sea-treasure-concave-v0': [],
            'minecart-v0': [],
            'mo-reacher-v4': []
        }
    }

    to_evaluate = {
        'dqn': {
            'deep-sea-treasure-concave-v0': ['wilrop/IPRO_opt/2ihwzr76',
                                             'wilrop/IPRO_opt/2ii52wzk',
                                             'wilrop/IPRO_opt/3msxdhww',
                                             'wilrop/IPRO_opt/2cysay30',
                                             'wilrop/IPRO_opt/1moes42e'],
            'minecart-v0': [],
            'mo-reacher-v4': ['wilrop/IPRO_opt/vbxkaso4']
        },
        'a2c': {
            'deep-sea-treasure-concave-v0': ['wilrop/IPRO_opt/3jgusppf',
                                             'wilrop/IPRO_opt/3kcnlngp',
                                             'wilrop/IPRO_opt/3ljj79w4',
                                             'wilrop/IPRO_opt/2ilk1u5q',
                                             'wilrop/IPRO_opt/3mrstqet'],
            'minecart-v0': ['wilrop/IPRO_opt/2fgq90hl'],
            'mo-reacher-v4': ['wilrop/IPRO_opt/2ga1y2rc',
                              'wilrop/IPRO_opt/y5gcmdof',
                              'wilrop/IPRO_opt/18kxy4la']
        },
        'ppo': {
            'deep-sea-treasure-concave-v0': [],
            'minecart-v0': ['wilrop/IPRO_opt/14kyvvys',
                            'wilrop/IPRO_opt/1ucq5ti1'],
            'mo-reacher-v4': ['wilrop/IPRO_opt/3jbowwrd',
                              'wilrop/IPRO_opt/3pzv9rk4',
                              'wilrop/IPRO_opt/14jtu9k9',
                              'wilrop/IPRO_opt/1xox36av']
        }
    }

    id_exp = {}
    idx = 0

    for alg, envs_dict in to_evaluate.items():
        for env_id, run_ids in envs_dict.items():
            for run_id in run_ids:
                if run_id not in checked[alg][env_id]:
                    for seed in range(num_seeds):
                        idx += 1
                        id_exp[idx] = (alg, env_id, seed, run_id)

    json.dump(id_exp, open('evaluation/experiments.json', 'w'))
    print(f'Number of experiments: {len(id_exp)}')


if __name__ == '__main__':
    link_experiments()
