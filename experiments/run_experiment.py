import yaml
import torch
import random
import argparse

import numpy as np

from environments import setup_env, setup_vector_env
from environments.bounding_boxes import get_bounding_box
from linear_solvers import init_linear_solver
from oracles import init_oracle
from outer_loops import init_outer_loop


def run_experiment(method, algorithm, config, outer_params, oracle_params, callback=None):
    """Run an single experiment.

    Args:
        method (str): The name of the outer loop method.
        algorithm (str): The name of the oracle algorithm.
        config (dict): The configuration dictionary.
        outer_params (dict): The parameters for the outer loop.
        oracle_params (dict): The parameters for the oracle.
        callback (function | None): The callback function.

    Returns:
        float: The hypervolume of the final Pareto front.
    """
    env_id = config['env_id']
    max_episode_steps = config['max_episode_steps']
    one_hot = config['one_hot_wrapper']
    gamma = config['gamma']
    seed = config['seed']
    wandb_project_name = config['wandb_project_name']
    wandb_entity = config['wandb_entity']
    run_name = f'{method}__{algorithm}__{env_id}__{seed}'

    # Seeding
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Setup environment.
    minimals, maximals, ref_point = get_bounding_box(env_id)

    if algorithm in ['MO-PPO', 'SN-MO-PPO']:
        env, num_objectives = setup_vector_env(env_id,
                                               oracle_params['num_envs'],
                                               seed,
                                               max_episode_steps=max_episode_steps,
                                               one_hot=one_hot,
                                               capture_video=False,
                                               run_name=run_name)
    else:
        env, num_objectives = setup_env(env_id,
                                        max_episode_steps=max_episode_steps,
                                        one_hot=one_hot,
                                        capture_video=False,
                                        run_name=run_name)

    if 'hidden_size' in oracle_params:
        hl_actor = (oracle_params['hidden_size'],) * oracle_params['num_hidden_layers']
        hl_critic = (oracle_params['hidden_size'],) * oracle_params['num_hidden_layers']
        oracle_params.pop('hidden_size')
        oracle_params.pop('num_hidden_layers')
    else:
        hl_actor = (oracle_params['hidden_size_actor'],) * oracle_params['num_hidden_layers_actor']
        hl_critic = (oracle_params['hidden_size_critic'],) * oracle_params['num_hidden_layers_critic']
        oracle_params.pop('hidden_size_actor')
        oracle_params.pop('hidden_size_critic')
        oracle_params.pop('num_hidden_layers_actor')
        oracle_params.pop('num_hidden_layers_critic')

    if algorithm in ['MO-DQN', 'SN-MO-DQN']:
        oracle_params['hidden_layers'] = hl_critic
    else:
        oracle_params['actor_hidden'] = hl_actor
        oracle_params['critic_hidden'] = hl_critic

    linear_solver = init_linear_solver('known_box', minimals=minimals, maximals=maximals)
    oracle = init_oracle(algorithm,
                         env,
                         gamma,
                         seed=seed,
                         **oracle_params)
    ol = init_outer_loop(method,
                         env,
                         num_objectives,
                         oracle,
                         linear_solver,
                         ref_point=ref_point,
                         exp_name=run_name,
                         wandb_project_name=wandb_project_name,
                         wandb_entity=wandb_entity,
                         seed=seed,
                         **outer_params)
    ol.solve(callback=callback)
    return ol.hv


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run experiment.')
    parser.add_argument('--config', type=str, default='./configs/sn_a2c_dst.yaml', help='Path to config file.')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    outer_params = config.pop('outer_loop')
    oracle_params = config.pop('oracle')
    method = outer_params.pop('method')
    algorithm = oracle_params.pop('algorithm')
    hv = run_experiment(method, algorithm, config, outer_params, oracle_params, callback=None)
    print(f'Hypervolume: {hv}')
