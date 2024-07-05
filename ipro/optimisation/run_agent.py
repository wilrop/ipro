import wandb
import json
import numpy as np

from typing import Any
from copy import deepcopy
from functools import partial
from omegaconf import OmegaConf, DictConfig

from ipro.experiments.parser import get_agent_runner_parser
from ipro.experiments.load_config import load_config
from ipro.experiments.run_experiment import run_experiment


def construct_hidden(algorithm, oracle_params):
    if 'hidden_size' in oracle_params:
        hidden_size = oracle_params.pop('hidden_size')
        num_hidden_layers = oracle_params.pop('num_hidden_layers')
        hl_actor = (hidden_size,) * num_hidden_layers
        hl_critic = (hidden_size,) * num_hidden_layers
    else:
        hidden_size_actor = oracle_params.pop('hidden_size_actor')
        hidden_size_critic = oracle_params.pop('hidden_size_critic')
        num_hidden_layers_actor = oracle_params.pop('num_hidden_layers_actor')
        num_hidden_layers_critic = oracle_params.pop('num_hidden_layers_critic')
        hl_actor = (hidden_size_actor,) * num_hidden_layers_actor
        hl_critic = (hidden_size_critic,) * num_hidden_layers_critic

    if algorithm in ['MO-DQN', 'SN-MO-DQN']:
        oracle_params.hidden_layers = hl_critic
    else:
        oracle_params.actor_hidden = hl_actor
        oracle_params.critic_hidden = hl_critic
    return oracle_params


def run_single_seed(config):
    config.experiment.seed = config.pop('seed')  # Relocate seed.
    return run_experiment(config)


def run_multi_seed(config, max_hv=4255, hv_buffer=5):
    results = []
    config.experiment.track_outer = False  # Necessary because we repeat the same config multiple times.
    config.experiment.track_oracle = False
    for seed in range(config.pop('num_seeds')):
        config.experiment.seed = seed
        hv = run_experiment(deepcopy(config))
        results.append(hv)
        wandb.log({
            'mean_hv': np.mean(results),
            'n_runs': seed + 1,
        })
        if hv < (max_hv - hv_buffer):  # Early stopping
            break
    return np.mean(results)


def run_hp_search(exp_config) -> Any:
    """Simple function to extract the config and run the experiment."""
    run = wandb.init()
    config = OmegaConf.create(dict(run.config))
    if 'hidden_size' in config.oracle or 'hidden_size_actor' in config.oracle:
        config.oracle = construct_hidden(config.oracle.algorithm, config.oracle)
    config.experiment = exp_config
    run.config['group'] = json.dumps(OmegaConf.to_container(config.oracle, resolve=True), sort_keys=True)
    if 'num_seeds' in config:
        return run_multi_seed(config)
    else:
        return run_single_seed(config)


def run_agents(exp_config: DictConfig, sweep_id: str):
    """Run an agent for a sweep."""
    agent_fun = partial(run_hp_search, exp_config)
    while True:
        wandb.agent(
            sweep_id,
            function=agent_fun,
            project=exp_config.wandb_project_name,
            entity=exp_config.wandb_entity,
            count=1,
        )


if __name__ == "__main__":
    parser = get_agent_runner_parser()
    args = parser.parse_args()
    exp_config = load_config(args).pop('experiment')
    run_agents(exp_config, args.sweep_id)
