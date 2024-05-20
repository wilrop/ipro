import wandb
import json

from typing import Any
from functools import partial
from experiments.parser import get_agent_runner_parser
from experiments.run_experiment import run_experiment


def run_hp_search(u_dir) -> Any:
    """Simple function to extract the config and run the experiment."""
    run = wandb.init()
    config = dict(run.config)
    outer_params = config.pop('outer_loop')
    method = outer_params.pop('method')
    oracle_params = config.pop('oracle')
    algorithm = oracle_params.pop('algorithm')
    extra_config = {'group': json.dumps(oracle_params, sort_keys=True)}
    return run_experiment(method, algorithm, config, outer_params, oracle_params, u_dir, extra_config=extra_config)


def run_agents(project: str, sweep_id: str, entity: str, u_dir: str):
    """Run an agent for a sweep."""
    agent_fun = partial(run_hp_search, u_dir)
    while True:
        wandb.agent(
            sweep_id,
            function=agent_fun,
            project=project,
            entity=entity,
            count=1,
        )


if __name__ == "__main__":
    parser = get_agent_runner_parser()
    args = parser.parse_args()
    run_agents(args.project, args.sweep_id, args.entity, args.u_dir)
