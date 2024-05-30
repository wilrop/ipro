import wandb
import json

from typing import Any
from functools import partial
from omegaconf import OmegaConf, DictConfig

from ipro.experiments.parser import get_agent_runner_parser
from ipro.experiments.load_config import load_config
from ipro.experiments.run_experiment import run_experiment


def run_hp_search(exp_config) -> Any:
    """Simple function to extract the config and run the experiment."""
    run = wandb.init()
    config = OmegaConf.create(dict(run.config))
    config['experiment'] = exp_config
    config.experiment.seed = config.pop('seed')  # Relocate seed.
    run.config['group'] = json.dumps(OmegaConf.to_container(config.oracle, resolve=True), sort_keys=True)
    return run_experiment(config)


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
