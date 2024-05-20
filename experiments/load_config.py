import argparse
from omegaconf import OmegaConf, DictConfig


def override_config(config: DictConfig, args: argparse.Namespace) -> DictConfig:
    """Override the configuration with specific arguments."""
    if args.override_config is not None:
        override_config = OmegaConf.from_cli(args.override_config)
        config = OmegaConf.merge(config, override_config)
    return config


def load_parts(config_parts: list[str], args: argparse.Namespace) -> DictConfig:
    """Load a specific list of parts from the configuration files."""
    config = OmegaConf.create()
    for part in config_parts:
        try:
            yaml_file = getattr(args, part)
        except AttributeError:
            continue
        path = f'{args.config_dir}/{part}/{yaml_file}.yaml'
        with open(path, 'r') as f:
            part_config = OmegaConf.load(f)
        config[part] = part_config
    return config


def load_config(args: argparse.Namespace) -> DictConfig:
    """Load the configuration file from the arguments."""
    config_parts = [
        'environment',
        'oracle',
        'outer_loop',
        'hyperparams',
    ]

    config = load_parts(config_parts, args)
    config = override_config(config, args)
    return config
