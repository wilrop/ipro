import wandb
import json
import argparse
from experiments.run_experiment import run_experiment


def remove_unused_params(parameters):
    """Remove unused parameters."""
    del parameters['alg_name']  # Use the provided algorithm.
    del parameters['seed']  # Use the provided seed.
    del parameters['tolerance']  # Use a fixed tolerance.
    del parameters['dimensions']  # Recomputed later.
    del parameters['vary_ideal']  # Defaults to false.
    del parameters['vary_nadir']  # Defaults to false.
    parameters.pop('warm_start', None)
    del parameters['max_iterations']  # Defaults to None.
    del parameters['deterministic_eval']  # Defaults to true.
    parameters.pop('group', None)
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


def setup_oracle_params(parameters):
    """Setup the parameters for the oracle."""
    oracle_params = {
        **parameters,
        'track': False
    }
    return oracle_params


def load_config_from_id(oracle, env_id, seed, run_id, u_dir):
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
    oracle_params = setup_oracle_params(parameters)

    # Run experiment and mark as reproduced.
    return method, oracle, config, outer_params, oracle_params, u_dir, extra_config


def reproduce_experiment(oracle, env_id, seed, run_id, u_dir):
    config = load_config_from_id(oracle, env_id, seed, run_id, u_dir)
    method, oracle, config, outer_params, oracle_params, u_dir, extra_config = config
    return run_experiment(method, oracle, config, outer_params, oracle_params, u_dir, extra_config=extra_config)


def reproduce_from_id(u_dir, exp_id, exp_dir, leftovers=False):
    """Reproduce an experiment given its ID."""
    if leftovers:
        id_exp_dict = json.load(open(f'{exp_dir}/leftovers.json', 'r'))
    else:
        id_exp_dict = json.load(open(f'{exp_dir}/experiments.json', 'r'))
    alg, env_id, seed, run_id = id_exp_dict[str(exp_id)]
    return reproduce_experiment(alg, env_id, seed, run_id, u_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Reproduce experiments given in a JSON file.')
    parser.add_argument(
        '--u_dir',
        type=str,
        default='./utility_function/utility_fns',
        help='Path to directory containing utility functions.'
    )
    parser.add_argument('--exp_id', type=str, default=1)
    parser.add_argument('--exp_dir', type=str, default='./evaluation')
    parser.add_argument('--leftovers', default=False, action='store_true')
    args = parser.parse_args()

    reproduce_from_id(args.u_dir, args.exp_id, args.exp_dir, args.leftovers)
