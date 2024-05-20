import wandb

from omegaconf import DictConfig, OmegaConf
from experiments.parser import get_sweep_parser
from experiments.load_config import load_config


def calc_num_experiments(config: DictConfig) -> int:
    num_experiments = 1
    for top_key, sub_dict in config.parameters.items():
        if 'values' in sub_dict:
            num_experiments *= len(sub_dict['values'])
        else:
            for key, value in sub_dict.items():
                num_experiments *= len(value['values'])
    return num_experiments


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
    merged_config = OmegaConf.to_container(merged_config, resolve=True)
    sweep_config.parameters = add_params_layers(merged_config)["parameters"]
    return OmegaConf.to_container(sweep_config, resolve=True)


def create_sweep(config: DictConfig, sweep_config: DictConfig, project: str):
    """Create a sweep and return its ID."""
    num_experiments = calc_num_experiments(sweep_config)
    sweep_wandb_config = merge_configs(config, sweep_config)
    sweep_id = wandb.sweep(
        sweep=sweep_wandb_config,
        project=project
    )
    return sweep_id, num_experiments


if __name__ == "__main__":
    parser = get_sweep_parser()
    args = parser.parse_args()
    config = load_config(args)
    sweep_config = config.pop('hyperparams')
    sweep_id, num_experiments = create_sweep(config, sweep_config, args.project)
    print(f"Sweep ID: {sweep_id} - Needs {num_experiments} runs.")
