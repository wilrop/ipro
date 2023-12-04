import logging
import os
import sys
import yaml
import argparse
import optuna

import numpy as np

from optuna._callbacks import RetryFailedTrialCallback
from experiments.run_experiment import run_experiment


def optimize_hyperparameters(study_name, env_name, optimize_trial, storage=None, n_trials=100, log_dir='.'):
    # Add stream handler of stdout to show the messages
    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))

    if storage is None:
        studies_dir = os.path.join(log_dir, 'studies')
        if not os.path.exists(studies_dir):
            os.makedirs(studies_dir)
        storage = f'sqlite:///{studies_dir}/{study_name}.db'

    sqlite_timeout = 300
    storage = optuna.storages.RDBStorage(
        storage,
        engine_kwargs={
            'connect_args': {'timeout': sqlite_timeout},
            'pool_size': 1,
        },
        heartbeat_interval=60,
        grace_period=120,
        failed_trial_callback=RetryFailedTrialCallback(),
    )
    study = optuna.create_study(
        study_name=study_name if env_name is not None else env_name,
        storage=storage,
        load_if_exists=True,
        direction='maximize')

    return study.optimize(optimize_trial, n_trials=n_trials)


def suggest_hyperparameters(trial, hyperparams_options):
    hyperparams = dict()

    def _suggest(_param, _config, _suggestion_type):

        def _str_categorical():
            cat_params = {str(x): x for x in _config['choices']}
            suggestion = trial.suggest_categorical(_param, cat_params.keys())
            return cat_params[suggestion]

        hyperparams[_param] = {
            'str_categorical': _str_categorical,
            'categorical': lambda: trial.suggest_categorical(_param, _config['choices']),
            'int': lambda: trial.suggest_int(
                _param, _config['min'], _config['max']),
            'constant': lambda: _config['value']
        }[_suggestion_type]()

    def _str_replace(str_, dict_):
        for word, replacement in dict_.items():
            str_ = str_.replace(word, str(replacement))
        return str_

    for param, config in hyperparams_options.items():
        if param != 'conditionals':
            assert 'type' in config, f'Please provide a suggestion type for parameter {param}'
            suggestion_type = config['type']
            _suggest(param, config, suggestion_type)

    conditionals = []
    conditional_params = hyperparams_options.get('conditionals', [])
    for conditional in conditional_params:
        class ConditionalStatement:
            _conditional_parameters = conditional['vars']
            _cond_statement = conditional['cond']
            _if_cond = conditional['if_cond']
            _else = conditional['else']

            def cond_fn(self):
                return eval(_str_replace(
                    self._cond_statement,
                    {conditional_param: hyperparams[conditional_param]
                     for conditional_param in self._conditional_parameters}))

            def true_fn(self):
                return [_suggest(_param, _config, _config['type'])
                        for _param, _config in self._if_cond.items()]

            def false_fn(self):
                [_suggest(_param, _config, _config['type'])
                 for _param, _config in self._else.items()]

        conditional = ConditionalStatement()
        conditionals.append((conditional.cond_fn, conditional.true_fn, conditional.false_fn))

    for cond, true_fn, false_fn in conditionals:
        if cond():
            true_fn()
        else:
            false_fn()

    print("Suggested hyperparameters")
    for key in hyperparams.keys():
        if key != "specs":
            print(f"{key}={hyperparams[key]}")

    return hyperparams


def search(parameters, study_name='study', n_trials=100, report_intermediate=True, log_dir='.'):
    """Search for hyperparameters for the given configuration."""
    outer_params = parameters.pop('outer_loop')
    oracle_params = parameters.pop('oracle')
    method = outer_params.pop('method')
    algorithm = oracle_params.pop('algorithm')
    hyperparams_options = parameters.pop('hyperparameters')

    def optimize_trial(trial):
        oracle_hyperparams = suggest_hyperparameters(trial, hyperparams_options)
        if report_intermediate:
            def callback(step, hypervolume, dominated_hv, discarded_hv, coverage, error):
                trial.report(hypervolume, step)
                if trial.should_prune():
                    raise optuna.TrialPruned()
        else:
            callback = None
        filled_oracle_params = {**oracle_params, **oracle_hyperparams}

        return run_experiment(method, algorithm, parameters, outer_params, filled_oracle_params, callback=callback)

    if isinstance(parameters['env_id'], str):
        env_name = parameters['env_id']
    else:
        parameters['env_name'] = np.random.choice(parameters['env_id'])
        env_name = None

    return optimize_hyperparameters(study_name, env_name, optimize_trial, n_trials=n_trials, log_dir=log_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run a hyperparameter search study')
    parser.add_argument('--params', type=str, default='sn_a2c_dst.yaml',
                        help='path of a yaml file containing the parameters of this study')
    parser.add_argument('--report_intermediate', default=False, action='store_true')
    parser.add_argument('--log_dir', type=str, default='./')
    parser.add_argument('--overwrite_seed', type=int, default=None, help='overwrite the seed in the yaml file')
    args = parser.parse_args()

    with open(args.params, 'r') as file:
        parameters = yaml.safe_load(file)

    if args.overwrite_seed is not None:
        parameters['seed'] = args.overwrite_seed

    search(parameters,
           study_name=parameters.get('study_name', 'IPRO_study'),
           n_trials=parameters['n_trials'],
           report_intermediate=args.report_intermediate,
           log_dir=args.log_dir)
