import json
import argparse


def link_experiments(num_seeds):
    """Creates a json file that links an experiment id to a hyperparameter file and seed"""
    checked = {
        'dqn': {
            'deep-sea-treasure-concave-v0': [],
            'minecart-v0': [],
            'mo-reacher-v4': ['wilrop/IPRO_opt/vbxkaso4']
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
            'deep-sea-treasure-concave-v0': ['wilrop/IPRO_opt/2tn5owa1',
                                             'wilrop/IPRO_opt/zetw2qex'],
            'minecart-v0': [],
            'mo-reacher-v4': ['wilrop/IPRO_opt/vbxkaso4']
        },
        'a2c': {
            'deep-sea-treasure-concave-v0': ['wilrop/IPRO_opt/g347p7nz',
                                             'wilrop/IPRO_opt/10sl3sct',
                                             'wilrop/IPRO_opt/239dy8eu',
                                             'wilrop/IPRO_opt/37hzrb0t',
                                             'wilrop/IPRO_opt/2ilk1u5q',
                                             'wilrop/IPRO_opt/2s5q3nff'],
            'minecart-v0': [],
            'mo-reacher-v4': ['wilrop/IPRO_opt/2ga1y2rc',
                              'wilrop/IPRO_opt/2i8jgu31',
                              'wilrop/IPRO_opt/34omm1q6',
                              'wilrop/IPRO_opt/y5gcmdof',
                              'wilrop/IPRO_opt/18kxy4la',
                              'wilrop/IPRO_opt/2niygvug',
                              'wilrop/IPRO_opt/1hf09bbk',
                              'wilrop/IPRO_opt/3abwo2bu',
                              'wilrop/IPRO_opt/2cfpvgbm']
        },
        'ppo': {
            'deep-sea-treasure-concave-v0': ['wilrop/IPRO_opt/3gyzdum4',
                                             'wilrop/IPRO_opt/g1dxgkl4'],
            'minecart-v0': ['wilrop/IPRO_opt/14kyvvys',
                            'wilrop/IPRO_opt/1ucq5ti1',
                            'wilrop/IPRO_opt/31wg6d5l'],
            'mo-reacher-v4': ['wilrop/IPRO_opt/3jbowwrd',
                              'wilrop/IPRO_opt/3pzv9rk4',
                              'wilrop/IPRO_opt/14jtu9k9',
                              'wilrop/IPRO_opt/1xox36av',
                              'wilrop/IPRO_opt/2je0ir5m',
                              'wilrop/IPRO_opt/19jh6tg4']
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
    parser = argparse.ArgumentParser(description='Link a job ID to a specific experiment.')
    parser.add_argument('--num_seeds', type=int, default=5, help='Number of seeds to run.')
    args = parser.parse_args()

    link_experiments(args.num_seeds)
