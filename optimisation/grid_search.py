import yaml
import json
import argparse

import numpy as np
from experiments.run_experiment import run_experiment


def grid_search(config, u_dir, exp_id, offset):
    """Run a grid search.

    Args:
        config (dict): The parameters for the grid search.
        u_dir (str): The directory containing the utility functions.
        exp_id (int): The experiment id. This starts at 1 and so needs to be decremented by 1.
        offset (int): The offset for the experiment id.
    """
    outer_params = config.pop('outer_loop')
    oracle_params = config.pop('oracle')
    method = outer_params.pop('method')
    algorithm = oracle_params.pop('algorithm')
    try:
        seeds = config.pop('seeds')
    except KeyError:
        seeds = [config.pop('seed')]

    # Extract and sort the grid which is useful for logging.
    grid = list(config.pop('hyperparameters').items())
    grid = [(key, v['choices']) for key, v in grid]
    grid = sorted(grid, key=lambda x: x[0])
    grid_shape = tuple([len(v) for k, v in grid] + [len(seeds)])
    print(f'Grid needs {np.prod(grid_shape)} experiments.')

    # Create key values for the experiment.
    idx = np.unravel_index(exp_id + offset - 1, grid_shape)  # Determine the parameter combination from the flat id.
    oracle_hyperparams = {key: values[i] for (key, values), i in zip(grid, idx)}

    # Create the filled oracle hyperparameters and set the seed.
    filled_oracle_params = {**oracle_params, **oracle_hyperparams}
    config['seed'] = seeds[idx[-1]]

    # Set the group string and run the experiment.
    all_params = {**outer_params, **filled_oracle_params}
    extra_config = {'group': json.dumps(all_params, sort_keys=True)}

    return run_experiment(method, algorithm, config, outer_params, filled_oracle_params, u_dir,
                          extra_config=extra_config)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run a hyperparameter search study')
    parser.add_argument(
        '--config',
        type=str,
        default='grid_a2c_dst.yaml',
        help='path of a yaml file containing the configuration of this grid search'
    )
    parser.add_argument(
        '--u_dir',
        type=str,
        default='./utility_function/utility_fns',
        help='Path to directory containing utility functions.'
    )
    parser.add_argument(
        '--exp_id',
        type=int,
        default=1
    )
    parser.add_argument(
        '--offset',
        type=int,
        default=0
    )
    args = parser.parse_args()

    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)

    grid_search(config, args.u_dir, args.exp_id, args.offset)
