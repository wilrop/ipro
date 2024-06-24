import argparse
from omegaconf import OmegaConf


def calc_num_experiments(config):
    num_experiments = 1
    for key, subdict in config.items():
        if 'values' not in subdict:
            num_experiments *= calc_num_experiments(subdict)
        else:
            num_experiments *= len(subdict['values'])
    return num_experiments


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate the number of experiments in a sweep.')
    parser.add_argument('--sweep_config', type=str, default='../configs/hyperparams/a2c.yaml',
                        help='path of a yaml file containing the configuration of this grid search')
    args = parser.parse_args()

    config = OmegaConf.load(args.sweep_config)
    config = config.parameters
    num_experiments = calc_num_experiments(config)
    print(f'Number of experiments: {num_experiments}')
