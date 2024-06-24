import wandb

from omegaconf import DictConfig, OmegaConf
from ipro.experiments.parser import get_sweep_parser
from ipro.experiments.load_config import load_config
from ipro.utils.search_utils import calc_num_experiments


def add_params_layers(config: dict) -> dict:
    assert type(config) == dict
    new_config = {}
    for key, value in config.items():
        if type(value) == dict:
            if len(value.keys()) == 1 and "values" in value:
                new_config[key] = value
            else:
                new_config[key] = add_params_layers(value)
        else:
            new_config[key] = {"value": value}
    return {"parameters": new_config}


def merge_configs(config: DictConfig, sweep_config: DictConfig) -> dict:
    """Merge the base config with the sweep config."""
    merged_config = OmegaConf.merge(config, sweep_config.parameters)
    merged_config = OmegaConf.to_container(merged_config, resolve=False)
    sweep_config.parameters = add_params_layers(merged_config)["parameters"]
    return OmegaConf.to_container(sweep_config, resolve=False)


def create_sweep(config: DictConfig, sweep_config: DictConfig, logger_config: DictConfig) -> str:
    """Create a sweep and return its ID."""
    sweep_wandb_config = merge_configs(config, sweep_config)
    sweep_id = wandb.sweep(
        sweep=sweep_wandb_config,
        project=logger_config.wandb_project_name,
        entity=logger_config.wandb_entity
    )
    return sweep_id


if __name__ == "__main__":
    parser = get_sweep_parser()
    args = parser.parse_args()
    config = load_config(args)
    num_experiments = calc_num_experiments(config.hyperparams.parameters)
    experiment_config = config.pop('experiment')
    sweep_config = config.pop('hyperparams')
    sweep_id = create_sweep(config, sweep_config, experiment_config)
    print(f"Sweep ID: {sweep_id} - Needs {num_experiments} runs.")
