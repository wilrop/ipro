import os
import json


def link_experiments(num_seeds=5):
    """Creates a json file that links an experiment id to a hyperparameter file and seed"""
    checked = {
        'dqn': {
            'deep-sea-treasure-concave-v0': ['wilrop/IPRO/x9xfnbv8',
                                             'wilrop/IPRO/ttzb2vs0',
                                             'wilrop/IPRO/ix8r4fsh',
                                             'wilrop/IPRO/x9xfnbv8',
                                             'wilrop/IPRO/ttzb2vs0',
                                             'wilrop/IPRO/ix8r4fsh',
                                             'wilrop/IPRO/f581k7r3',
                                             'wilrop/IPRO/b8kd5c59',
                                             'wilrop/IPRO/3ba9x1uo',
                                             'wilrop/IPRO/306dyvmk',
                                             'wilrop/IPRO/2vuh88h6',
                                             'wilrop/IPRO/2vn1iqdl',
                                             'wilrop/IPRO/2ts0ezja',
                                             'wilrop/IPRO/289gpteh',
                                             'wilrop/IPRO/286x9a6t',
                                             'wilrop/IPRO/2055czjj',
                                             'wilrop/IPRO/1w0m1y6i',
                                             'wilrop/IPRO/1odcxii3',
                                             'wilrop/IPRO/1m9l90u5',
                                             'wilrop/IPRO/1caf8qmt',
                                             'wilrop/IPRO/vhctag70',
                                             'wilrop/IPRO/nkfq8tgc',
                                             'wilrop/IPRO/m8f1ib2h',
                                             'wilrop/IPRO/8l7hf6gd',
                                             'wilrop/IPRO/3vh5e87c',
                                             'wilrop/IPRO/3sqwmyow',
                                             'wilrop/IPRO/3p3u7gln',
                                             'wilrop/IPRO/3nf9ebo6',
                                             'wilrop/IPRO/3lugayto',
                                             'wilrop/IPRO/3iaoq66k',
                                             'wilrop/IPRO/3i7w0kfb',
                                             'wilrop/IPRO/3f963c97',
                                             'wilrop/IPRO/3dejfk3b',
                                             'wilrop/IPRO/35kx3zix',
                                             'wilrop/IPRO/2wrelpkv',
                                             'wilrop/IPRO/2wmmzkkl',
                                             'wilrop/IPRO/2vimvoar',
                                             'wilrop/IPRO/2v2akoo8',
                                             'wilrop/IPRO/2n8ek51m',
                                             'wilrop/IPRO/2i2wryg6',
                                             'wilrop/IPRO/2i0a1d5z',
                                             'wilrop/IPRO/2cjrfzw3',
                                             'wilrop/IPRO/2bbqo7x8'
                                             ],
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

    to_evaluate = {
        'dqn': {
            'deep-sea-treasure-concave-v0': ['wilrop/IPRO/29uv0zr4',
                                             'wilrop/IPRO/28ntua1d',
                                             'wilrop/IPRO/26p2niv0',
                                             'wilrop/IPRO/2490522u',
                                             'wilrop/IPRO/23faolb4',
                                             'wilrop/IPRO/21jpgplg',
                                             'wilrop/IPRO/1zcvl6hf',
                                             'wilrop/IPRO/1yd9vbn3',
                                             'wilrop/IPRO/1x5fyoz8',
                                             'wilrop/IPRO/1vjgwvle',
                                             'wilrop/IPRO/1k0qbjb2',
                                             'wilrop/IPRO/1bgt59h2',
                                             'wilrop/IPRO/1bge7ow4',
                                             'wilrop/IPRO/1a61d8l7',
                                             'wilrop/IPRO/17h5ubly',
                                             'wilrop/IPRO/15aasfpq',
                                             'wilrop/IPRO/12h59sxr',
                                             'wilrop/IPRO/2iv0yb8a',
                                             'wilrop/IPRO/2e2m028e',
                                             'wilrop/IPRO/18nohhqv',
                                             'wilrop/IPRO/1p0x0mfe',
                                             'wilrop/IPRO/vj7q7wp9',
                                             'wilrop/IPRO/ye3t1erm',
                                             'wilrop/IPRO/g1lklsy2',
                                             'wilrop/IPRO/evjn9ziv',
                                             'wilrop/IPRO/e9tmu2y4',
                                             'wilrop/IPRO/3rd96jfj',
                                             'wilrop/IPRO/2usn173h',
                                             'wilrop/IPRO/2rwz7gv5',
                                             'wilrop/IPRO/2kvlrctx',
                                             'wilrop/IPRO/2fsarabr',
                                             'wilrop/IPRO/2aq7fj7a',
                                             'wilrop/IPRO/27iotgif',
                                             'wilrop/IPRO/26rctowe',
                                             'wilrop/IPRO/1newuhpd',
                                             'wilrop/IPRO/192ms1y8',
                                             'wilrop/IPRO/vbo0id04',
                                             'wilrop/IPRO/gglykawu',
                                             'wilrop/IPRO/8tuphyez',
                                             'wilrop/IPRO/3b1nwm1f'],
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
