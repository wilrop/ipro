import argparse
import yaml
import numpy as np


def num_experiments(config):
    # Extract and sort the grid which is useful for logging.
    seeds = config.pop('seeds')
    grid = list(config.pop('hyperparameters').items())
    grid = [(key, v['choices']) for key, v in grid]
    grid = sorted(grid, key=lambda x: x[0])
    grid_shape = tuple([len(v) for k, v in grid] + [len(seeds)])
    print(f'Grid needs {np.prod(grid_shape)} experiments.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run a hyperparameter search study')
    parser.add_argument('--config', type=str, default='../optimisation/grid_a2c_dst.yaml',
                        help='path of a yaml file containing the configuration of this grid search')
    args = parser.parse_args()

    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)

    num_experiments(config)
