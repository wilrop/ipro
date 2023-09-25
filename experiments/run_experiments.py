import json
import torch
import argparse

import numpy as np

from environments import setup_env, setup_vector_env
from linear_solvers import init_linear_solver
from oracles import init_oracle
from outer_loops import init_outer_loop


def get_env_info(env_id):
    if env_id == 'deep-sea-treasure-concave-v0':
        nadirs = np.array([[0, 0], [124, -19]])
        ideals = np.array([[124, -19], [0, 0]])
        ref_point = np.array([0, -50])
        max_episode_steps = 50
        outer = "2D"
    elif env_id == 'mo-reacher-v4':
        nadirs = np.array([[-50, -50, -50, -50], [-50, -50, -50, -50], [-50, -50, -50, -50], [-50, -50, -50, -50]])
        ideals = np.array([[40, -50, -50, -50], [-50, 40, -50, -50], [-50, -50, 40, -50], [-50, -50, -50, 40]])
        ref_point = np.array([-50, -50, -50, -50])
        max_episode_steps = 50
        outer = "PRIOL"
    elif env_id == 'minecart-v0':
        nadirs = np.array([[-1, -1, -200], [-1, -1, -200], [-1, -1, -200]])
        ideals = np.array([[1.5, -1, -200], [-1, 1.5, -200], [-1, -1, 0]])
        ref_point = np.array([-1, -1, -200])
        max_episode_steps = 1000
        outer = "PRIOL"
    else:
        raise NotImplementedError
    tolerance = 0.0
    return outer, nadirs, ideals, ref_point, max_episode_steps, tolerance


def load_parameters(exp_dir, params_file):
    with open(f'{exp_dir}/hyperparams/{params_file}', "r") as f:
        parameters = json.load(f)
    for key, value in parameters.items():
        parameters[key] = value['value']

    # Remove unused parameters.
    parameters.pop('_wandb')
    parameters.pop('seed')
    parameters.pop('track')
    parameters.pop('method')
    parameters.pop('max_steps')
    parameters.pop('warm_start')
    parameters.pop('tolerance')
    parameters.pop('dimensions')
    return parameters


def run_experiment(exp_id, exp_dir):
    params_file, seed = json.load(open(f'{exp_dir}/hyperparams/experiments.json', 'r'))[str(exp_id)]

    splitted_params_file = params_file.split('.')[0].split('_')
    exp_name = '_'.join(splitted_params_file[:2])
    if splitted_params_file[0] == 'a2c':
        oracle_name = 'MO-A2C'
    elif splitted_params_file[0] == 'ppo':
        oracle_name = 'MO-PPO'
    else:
        oracle_name = 'MO-DQN'
    arg_idx = splitted_params_file[2]
    parameters = load_parameters(exp_dir, params_file)
    torch.manual_seed(seed)

    env_id = parameters.pop('env_id')
    run_name = f"{exp_name}__{seed}__arg{arg_idx}"

    outer_loop_name, nadirs, ideals, ref_point, max_episode_steps, tolerance = get_env_info(env_id)

    if oracle_name == 'MO-PPO':
        env, num_objectives = setup_vector_env(env_id, parameters['num_envs'], seed, run_name, False,
                                               max_episode_steps=max_episode_steps)
    else:
        env, num_objectives = setup_env(env_id, max_episode_steps, capture_video=False, run_name=run_name)

    linear_solver = init_linear_solver('known_box', nadirs=nadirs, ideals=ideals)
    oracle = init_oracle(oracle_name,
                         env,
                         parameters.pop('gamma'),
                         track=True,
                         log_freq=parameters.pop('log_freq'),
                         seed=seed,
                         **parameters)
    ol = init_outer_loop(outer_loop_name,
                         env,
                         num_objectives,
                         oracle,
                         linear_solver,
                         ref_point=ref_point,
                         track=True,
                         exp_name=run_name,
                         wandb_project_name='IPRO_experiments',
                         wandb_entity=None,
                         warm_start=False,
                         tolerance=tolerance,
                         seed=seed)
    ol.solve()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run experiments from JSON files.')
    parser.add_argument('--exp_id', type=str, default=1)
    parser.add_argument('--exp_dir', type=str)
    args = parser.parse_args()

    run_experiment(args.exp_id, args.exp_dir)
