import wandb
import json
import argparse
from experiments.run_experiment import run_experiment


def get_env_info(env_id):
    """Get environment information."""
    if env_id == 'deep-sea-treasure-concave-v0':
        gamma = 1.0
        max_episode_steps = 50
        one_hot_wrapper = True
        tolerance = 0.0
    elif env_id == 'mo-reacher-v4':
        gamma = 0.99
        max_episode_steps = 50
        one_hot_wrapper = False
        tolerance = 1.e-15
    elif env_id == 'minecart-v0':
        gamma = 0.98
        max_episode_steps = 1000
        one_hot_wrapper = False
        tolerance = 1.e-15
    else:
        raise NotImplementedError
    return gamma, max_episode_steps, one_hot_wrapper, tolerance


def remove_unused_params(parameters):
    """Remove unused parameters."""
    del parameters['alg_name']  # Use the provided algorithm.
    del parameters['seed']  # Use the provided seed.
    del parameters['tolerance']  # Use a fixed tolerance.
    del parameters['dimensions']  # Recomputed later.
    del parameters['vary_ideal']  # Defaults to false.
    del parameters['vary_nadir']  # Defaults to false.
    del parameters['warm_start']  # Defaults to false.
    del parameters['max_iterations']  # Defaults to None.
    del parameters['deterministic_eval']  # Defaults to true.
    return parameters


def setup_config(parameters, seed, max_episode_steps, one_hot_wrapper):
    """Setup the configuration dictionary."""
    config = {
        'env_id': parameters.pop('env_id'),
        'max_episode_steps': max_episode_steps,
        'one_hot_wrapper': one_hot_wrapper,
        'gamma': parameters.pop('gamma'),
        'seed': seed,
        'wandb_project_name': 'IPRO_runs',
        'wandb_entity': None,
    }
    return config


def setup_outer_params(tolerance):
    """Setup the parameters for the outer loop."""
    outer_params = {
        'tolerance': tolerance,
        'max_iterations': None,
        'track': True
    }
    return outer_params


def setup_oracle_params(parameters, grid=False):
    """Setup the parameters for the oracle."""
    oracle_params = {
        **parameters,
        'track': False
    }
    if grid:
        oracle_params['grid_sample'] = True
        oracle_params['pretrain_iters'] = 25
    return oracle_params


def reproduce_experiment(oracle, env_id, seed, run_id, grid=False):
    """Reproduce an experiment."""
    api = wandb.Api(timeout=120)
    run = api.run(f'{run_id}')
    parameters = run.config
    run.summary['reproduced'] = True  # Mark as reproduced.
    run.update()
    _, max_episode_steps, one_hot_wrapper, tolerance = get_env_info(env_id)
    parameters = remove_unused_params(parameters)
    extra_config = {'parent_run_id': run_id}

    # Setup experiment parameters.
    method = parameters.pop('method')
    config = setup_config(parameters, seed, max_episode_steps, one_hot_wrapper)
    outer_params = setup_outer_params(tolerance)
    oracle_params = setup_oracle_params(parameters, grid=grid)

    # Run experiment and mark as reproduced.
    run_experiment(method, oracle, config, outer_params, oracle_params, extra_config=extra_config)


def reproduce_from_id(exp_id, exp_dir, leftovers=False, grid=False):
    """Reproduce an experiment given its ID."""
    if leftovers:
        id_exp_dict = json.load(open(f'{exp_dir}/leftovers.json', 'r'))
    elif grid:
        id_exp_dict = json.load(open(f'{exp_dir}/grid_experiments.json', 'r'))
    else:
        id_exp_dict = json.load(open(f'{exp_dir}/experiments.json', 'r'))
    alg, env_id, seed, run_id = id_exp_dict[str(exp_id)]
    return reproduce_experiment(alg, env_id, seed, run_id, grid=grid)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Reproduce experiments given in a JSON file.')
    parser.add_argument('--exp_id', type=str, default=1)
    parser.add_argument('--exp_dir', type=str, default='./evaluation')
    parser.add_argument('--leftovers', default=False, action='store_true')
    parser.add_argument('--grid', default=False, action='store_true')
    args = parser.parse_args()

    reproduce_from_id(args.exp_id, args.exp_dir, args.leftovers, args.grid)
