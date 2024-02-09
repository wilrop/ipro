import yaml
import argparse

import numpy as np
from experiments.run_experiment import run_experiment


def grid_search(parameters, exp_id, offset):
    """Run a grid search.

    Args:
        parameters (dict): The parameters for the grid search.
        exp_id (int): The experiment id. This starts at 1 and so needs to be decremented by 1.
        offset (int): The offset for the experiment id.
    """
    outer_params = parameters.pop('outer_loop')
    oracle_params = parameters.pop('oracle')
    method = outer_params.pop('method')
    algorithm = oracle_params.pop('algorithm')
    seeds = parameters.pop('seeds')

    # Determine the parameter combination from the flat id.
    grid = [(key, v['choices']) for key, v in list(parameters.pop('hyperparameters').items())]
    grid_shape = tuple([len(v) for k, v in grid] + [len(seeds)])
    idx = np.unravel_index(exp_id + offset - 1, grid_shape)  # Determine the parameter combination from the flat id.
    print(f'Grid needs {np.prod(grid_shape)} experiments.')

    # Create ordered parameter group hash for the experiment.
    sorted_grid = sorted(grid, key=lambda x: x[0])
    group_str = '_'.join([f'{key}={values[idx[i]]}' for i, (key, values) in enumerate(sorted_grid)])
    extra_config = {'group': group_str}

    # Create the oracle hyperparameters and set the seed.
    oracle_hyperparams = {key: values[idx[i]] for i, (key, values) in enumerate(grid)}
    filled_oracle_params = {**oracle_params, **oracle_hyperparams}
    parameters['seed'] = seeds[idx[-1]]
    return run_experiment(method, algorithm, parameters, outer_params, filled_oracle_params, extra_config=extra_config)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run a hyperparameter search study')
    parser.add_argument('--params', type=str, default='grid_ppo_dst.yaml',
                        help='path of a yaml file containing the parameters of this study')
    parser.add_argument('--exp_id', type=str, default=1)
    parser.add_argument('--offset', type=int, default=0)
    args = parser.parse_args()

    with open(args.params, 'r') as file:
        params = yaml.safe_load(file)

    exp_id = int(args.exp_id)
    offset = int(args.offset)
    grid_search(params, args.exp_id, args.offset)
