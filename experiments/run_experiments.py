import json
import torch
import random
import argparse
import wandb

import numpy as np

from environments import setup_env, setup_vector_env
from environments.bounding_boxes import get_bounding_box
from linear_solvers import init_linear_solver
from oracles import init_oracle
from outer_loops import init_outer_loop


def get_env_info(env_id):
    if env_id == 'deep-sea-treasure-concave-v0':
        max_episode_steps = 50
        outer = "IPRO-2D"
    elif env_id == 'mo-reacher-v4':
        max_episode_steps = 50
        outer = "IPRO"
    elif env_id == 'minecart-v0':
        max_episode_steps = 1000
        outer = "IPRO"
    else:
        raise NotImplementedError
    tolerance = 0.00001
    return outer, max_episode_steps, tolerance


def load_parameters(run_id):
    api = wandb.Api(timeout=120)
    run = api.run(f'{run_id}')
    parameters = run.config

    # Remove unused parameters.
    parameters.pop('seed', None)
    parameters.pop('window_size', None)
    parameters.pop('track', None)
    parameters.pop('method', None)
    parameters.pop('max_steps', None)
    parameters.pop('max_iterations')
    parameters.pop('warm_start', None)
    parameters.pop('tolerance', None)
    parameters.pop('dimensions', None)
    return parameters


def run_experiment(exp_id, exp_dir):
    id_exp_dict = json.load(open(f'{exp_dir}/evaluation/experiments.json', 'r'))
    alg, env_id, seed, run_id = id_exp_dict[str(exp_id)]

    if alg == 'a2c':
        oracle_name = 'MO-A2C'
    elif alg == 'ppo':
        oracle_name = 'MO-PPO'
    else:
        oracle_name = 'MO-DQN'

    parameters = load_parameters(run_id)

    # Seeding
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    env_id = parameters.pop('env_id')
    run_name = f"{env_id}__{run_id}__{seed}"

    minimals, maximals, ref_point = get_bounding_box(env_id)
    outer_loop_name, max_episode_steps, tolerance = get_env_info(env_id)

    if oracle_name == 'MO-PPO':
        env, num_objectives = setup_vector_env(env_id, parameters['num_envs'], seed, run_name, False,
                                               max_episode_steps=max_episode_steps)
    else:
        env, num_objectives = setup_env(env_id, max_episode_steps, capture_video=False, run_name=run_name)

    linear_solver = init_linear_solver('known_box', minimals=minimals, maximals=maximals)
    oracle = init_oracle(oracle_name,
                         env,
                         parameters.pop('gamma'),
                         track=False,
                         warm_start=False,
                         log_freq=parameters.pop('log_freq'),
                         seed=seed,
                         **parameters)
    ol = init_outer_loop(outer_loop_name,
                         env,
                         num_objectives,
                         oracle,
                         linear_solver,
                         ref_point=ref_point,
                         tolerance=tolerance,
                         track=True,
                         exp_name=run_name,
                         wandb_project_name='IPRO_experiments',
                         wandb_entity=None,
                         seed=seed)
    ol.solve()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run experiments from JSON files.')
    parser.add_argument('--exp_id', type=str, default=1)
    parser.add_argument('--exp_dir', type=str, default='.')
    args = parser.parse_args()

    run_experiment(args.exp_id, args.exp_dir)
