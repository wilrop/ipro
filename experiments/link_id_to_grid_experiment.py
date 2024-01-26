import argparse
import json


def link_grid_experiment(parents, num_seeds):
    """Link a job ID to a grid experiment."""
    id_exp = {}
    idx = 0
    for alg, parent in parents:
        for seed in range(num_seeds):
            idx += 1
            id_exp[idx] = (alg, 'deep-sea-treasure-concave-v0', seed, parent)

    json.dump(id_exp, open('evaluation/grid_experiments.json', 'w'))
    print(f'Number of grid experiments: {len(id_exp)}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Link a job ID to a baseline experiment.')
    parser.add_argument('--num_seeds', type=int, default=5, help='Number of seeds to run.')
    args = parser.parse_args()
    parents = [
        ('SN-MO-DQN', 'wilrop/IPRO_opt/5dunve73'),
        ('SN-MO-DQN', 'wilrop/IPRO_opt/2byvb8h5'),
        ('SN-MO-DQN', 'wilrop/IPRO_opt/3jwx3yaa'),
        ('SN-MO-A2C', 'wilrop/IPRO_opt/3a7qvc13'),
        ('SN-MO-A2C', 'wilrop/IPRO_opt/29gaymtx')
    ]
    link_grid_experiment(parents, args.num_seeds)
