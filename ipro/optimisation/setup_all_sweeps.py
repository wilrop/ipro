from collections import namedtuple

from ipro.optimisation.setup_sweep import create_sweep
from ipro.experiments.load_config import load_config
from ipro.utils.search_utils import calc_num_experiments


def setup_all_sweeps(default_params: dict, agent_combos: list[dict]) -> list:
    sweeps = []

    for agent_combo in agent_combos:
        args = default_params.copy()
        args.update(agent_combo)
        args = namedtuple("Args", args.keys())(*args.values())
        config = load_config(args)
        num_experiments = calc_num_experiments(config.hyperparams.parameters)
        sweep_config = config.pop('hyperparams')
        experiment_config = config.pop('experiment')
        sweep_id = create_sweep(config, sweep_config, experiment_config)
        sweeps.append((sweep_id, num_experiments))

    for sweep_id, num_experiments in sweeps:
        print(f"Sweep ID: {sweep_id} - Needs {num_experiments} runs.")

    print(f'For easy copy and pasting...')

    for sweep_id, _ in sweeps:
        print(sweep_id)
    return sweeps


if __name__ == '__main__':
    default_params = {
        'config_dir': '../configs',
        'experiment': 'base',
        'environment': 'dst',
        'outer_loop': 'ipro_2d',
        'override_config': None,
    }
    agent_combos = [
        {'oracle': 'dqn', 'hyperparams': 'dqn'},
        {'oracle': 'a2c', 'hyperparams': 'a2c'},
        {'oracle': 'ppo', 'hyperparams': 'ppo'},
    ]

    sweep_ids = setup_all_sweeps(default_params, agent_combos)
    print(sweep_ids)
