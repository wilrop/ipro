import time
import torch
import random

import numpy as np

from omegaconf import OmegaConf

from ipro.experiments.parser import get_experiment_runner_parser
from ipro.experiments.load_config import load_config
from ipro.environments import setup_env, setup_vector_env
from ipro.linear_solvers import init_linear_solver
from ipro.oracles import init_oracle
from ipro.outer_loops import init_outer_loop


def run_experiment(config, callback=None, extra_config=None):
    """Run an experiment."""
    method = config.outer_loop.pop('method')
    algorithm = config.oracle.pop('algorithm')
    env_id = config.environment.pop('env_id')
    seed = config.experiment.pop('seed')
    run_name = f'{method}__{algorithm}__{env_id}__{seed}__{int(time.time())}'

    # Seeding
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    if algorithm in ['MO-PPO', 'SN-MO-PPO']:
        env, num_objectives = setup_vector_env(
            env_id,
            config.oracle.num_envs,
            seed,
            gamma=config.environment.gamma,
            max_episode_steps=config.environment.max_episode_steps,
            one_hot=config.environment.one_hot_wrapper,
            capture_video=config.experiment.capture_video,
            run_name=run_name
        )
    else:
        env, num_objectives = setup_env(
            env_id,
            gamma=config.environment.gamma,
            max_episode_steps=config.environment.max_episode_steps,
            one_hot=config.environment.one_hot_wrapper,
            capture_video=config.experiment.capture_video,
            run_name=run_name
        )

    linear_solver = init_linear_solver(
        'known_box',
        minimals=config.environment.minimals,
        maximals=config.environment.maximals
    )
    oracle = init_oracle(
        algorithm,
        env,
        config.environment.gamma,
        seed=seed,
        track=config.experiment.track_oracle,
        log_freq=config.experiment.oracle_log_freq,
        **OmegaConf.to_container(config.oracle, resolve=True),
    )
    ol = init_outer_loop(
        method,
        env,
        num_objectives,
        oracle,
        linear_solver,
        ref_point=config.environment.ref_point,
        exp_name=run_name,
        wandb_project_name=config.experiment.wandb_project_name,
        wandb_entity=config.experiment.wandb_entity,
        seed=seed,
        extra_config=extra_config,
        track=config.experiment.track_outer,
        **OmegaConf.to_container(config.outer_loop, resolve=True),
    )
    ol.solve(callback=callback)
    return ol.hv


if __name__ == '__main__':
    parser = get_experiment_runner_parser()
    args = parser.parse_args()
    config = load_config(args)
    run_experiment(config)
