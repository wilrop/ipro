import argparse


def add_experiment_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        '--config_dir',
        type=str,
        help='The directory of the config files.'
    )
    parser.add_argument(
        '--experiment',
        type=str,
        help='The path to the experiment config file.'
    )
    return parser


def add_part_config_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add the config arguments for specific parts."""
    parser.add_argument(
        '--environment',
        type=str,
        help='The path to the environment config file.'
    )
    parser.add_argument(
        '--oracle',
        type=str,
        help='The path to the oracle config file.'
    )
    parser.add_argument(
        '--outer_loop',
        type=str,
        help='The path to the outer loop config file.'
    )
    return parser


def add_override_arg(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add the option to override the config."""
    parser.add_argument(
        '--override_config',
        nargs="+",
        type=str,
        default=[
        ],
        help='Override parameters from the config file.'
    )
    return parser


def get_experiment_runner_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Setup a hyperparameter sweep.")
    parser = add_experiment_args(parser)
    parser = add_part_config_args(parser)
    parser = add_override_arg(parser)
    return parser


def get_sweep_parser() -> argparse.ArgumentParser:
    """Get the parser for the hyperparameter sweep."""
    parser = argparse.ArgumentParser(description="Setup a hyperparameter sweep.")
    parser.add_argument(
        '--hyperparams',
        type=str,
        help='The directory of the hyperparams config file.'
    )
    parser = add_experiment_args(parser)
    parser = add_part_config_args(parser)
    parser = add_override_arg(parser)
    return parser


def get_agent_runner_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run an agent for a hyperparameter search.")
    parser.add_argument(
        "--sweep_id",
        type=str,
        help="The ID of the sweep.",
    )
    parser = add_experiment_args(parser)
    parser = add_override_arg(parser)
    return parser
