import argparse
import os
import yaml
from experiments.run_experiment import run_experiment

def reproduce_icml(u_dir, exp_id, exp_dir):
    """Reproduce an experiment given its ID."""
    files = [
        'sn_a2c_dst.yaml',
        'sn_a2c_minecart.yaml',
        'sn_a2c_reacher.yaml',
        'sn_dqn_dst.yaml',
        'sn_dqn_minecart.yaml',
        'sn_dqn_reacher.yaml',
        'sn_ppo_dst.yaml',
        'sn_ppo_minecart.yaml',
        'sn_ppo_reacher.yaml',
    ]
    num_seeds = 5
    exp_id = int(exp_id) - 1  # 0-indexed.
    file = files[exp_id // num_seeds]
    seed = exp_id % num_seeds

    file_path = os.path.join(exp_dir, file)
    with open(file_path, 'r') as f:
        config = yaml.safe_load(f)

    config['seed'] = seed
    oracle_params = config.pop('oracle')
    oracle = oracle_params.pop('algorithm')
    outer_params = config.pop('outer_loop')
    method = outer_params.pop('method')

    run_experiment(method, oracle, config, outer_params, oracle_params, u_dir, extra_config=None)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Reproduce experiments from a YAML file.')
    parser.add_argument(
        '--u_dir',
        type=str,
        default='./utility_function/utility_fns',
        help='Path to directory containing utility functions.'
    )
    parser.add_argument('--exp_id', type=str, default=1)
    parser.add_argument('--exp_dir', type=str, default='./icml_configs')
    args = parser.parse_args()

    reproduce_icml(args.u_dir, args.exp_id, args.exp_dir)
