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
            'deep-sea-treasure-concave-v0': ['wilrop/IPRO/x9xfnbv8', 'wilrop/IPRO/ttzb2vs0', 'wilrop/IPRO/ix8r4fsh'],
            'minecart-v0': ['wilrop/IPRO/x2pv2xwv', 'wilrop/IPRO/fpoxkpym', 'wilrop/IPRO/bpb21s4t'],
            'mo-reacher-v4': ['wilrop/IPRO/1e0xr6jn', 'wilrop/IPRO/307g4ksp', 'wilrop/IPRO/12zmmwr8']
        },
        'a2c': {
            'deep-sea-treasure-concave-v0': ['wilrop/IPRO/nxiv5fqn', 'wilrop/IPRO/32eh8222', 'wilrop/IPRO/2pc0xpyf'],
            'minecart-v0': ['wilrop/IPRO/84vg9vtt', 'wilrop/IPRO/1boe8jil', 'wilrop/IPRO/g96509f9'],
            'mo-reacher-v4': ['wilrop/IPRO/1vfi17wo', 'wilrop/IPRO/2xqvo1cc', 'wilrop/IPRO/3sroox9j']
        },
        'ppo': {
            'deep-sea-treasure-concave-v0': ['wilrop/IPRO/2o7zx3ic', 'wilrop/IPRO/1y9o463h', 'wilrop/IPRO/1fz5i5sy'],
            'minecart-v0': ['wilrop/IPRO/3re02q13', 'wilrop/IPRO/2lo9cwwv', 'wilrop/IPRO/y4lbmsh8'],
            'mo-reacher-v4': ['wilrop/IPRO/1to7naft', 'wilrop/IPRO/27vlgn7k', 'wilrop/IPRO/3706ujn8']
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
