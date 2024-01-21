import argparse
import json


def link_baseline(num_seeds):
    """Link a job ID to a baseline experiment."""
    id_exp = {}
    idx = 0
    combos = [('GPI-LS', 'deep-sea-treasure-concave-v0'),
              ('PCN', 'deep-sea-treasure-concave-v0'),
              ('PCN', 'mo-reacher-v4'),
              ('Envelope', 'deep-sea-treasure-concave-v0')]
    for baseline, env_id in combos:
        for seed in range(num_seeds):
            idx += 1
            id_exp[idx] = (baseline, env_id, seed)

    json.dump(id_exp, open('evaluation/baselines.json', 'w'))
    print(f'Number of baseline experiments: {len(id_exp)}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Link a job ID to a baseline experiment.')
    parser.add_argument('--num_seeds', type=int, default=5, help='Number of seeds to run.')
    args = parser.parse_args()

    link_baseline(args.num_seeds)
