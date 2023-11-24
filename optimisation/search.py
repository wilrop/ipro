import logging
import random
import os
import sys
import yaml
import time
import argparse
import optuna
import torch

import numpy as np

from optuna._callbacks import RetryFailedTrialCallback

from environments import setup_env, setup_vector_env
from environments.bounding_boxes import get_bounding_box
from linear_solvers import init_linear_solver
from oracles import init_oracle
from outer_loops import init_outer_loop


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

    def optimize_trial(trial):
        env_id = parameters['env_id']
        max_episode_steps = parameters['max_episode_steps']
        one_hot = parameters['one_hot_wrapper']
        gamma = parameters['gamma']
        study_name = parameters['study_name']
        seed = parameters['seed']
        seeds = [seed] if isinstance(seed, int) else seed
        wandb_project_name = parameters['wandb_project_name']
        wandb_entity = parameters['wandb_entity']
        minimals, maximals, ref_point = get_bounding_box(env_id)
        method = parameters['outer_loop'].pop('method')
        algorithm = parameters['oracle'].pop('algorithm')
        oracle_hyperparams = suggest_hyperparameters(trial, parameters.pop('hyperparameters'))

        if 'hidden_size' in oracle_hyperparams:
            hl_actor = (oracle_hyperparams['hidden_size'],) * oracle_hyperparams['num_hidden_layers']
            hl_critic = (oracle_hyperparams['hidden_size'],) * oracle_hyperparams['num_hidden_layers']
            oracle_hyperparams.pop('hidden_size')
            oracle_hyperparams.pop('num_hidden_layers')
        else:
            hl_actor = (oracle_hyperparams['hidden_size_actor'],) * oracle_hyperparams['num_hidden_layers_actor']
            hl_critic = (oracle_hyperparams['hidden_size_critic'],) * oracle_hyperparams['num_hidden_layers_critic']
            oracle_hyperparams.pop('hidden_size_actor')
            oracle_hyperparams.pop('hidden_size_critic')
            oracle_hyperparams.pop('num_hidden_layers_actor')
            oracle_hyperparams.pop('num_hidden_layers_critic')

        if algorithm in ['MO-DQN', 'SN-MO-DQN']:
            oracle_hyperparams['hidden_layers'] = hl_critic
        else:
            oracle_hyperparams['actor_hidden'] = hl_actor
            oracle_hyperparams['critic_hidden'] = hl_critic

        hypervolumes = []

        for seed in seeds:
            # Seeding
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)

            run_name = f"{study_name}__{seed}__{int(time.time())}"

            if algorithm in ['MO-PPO', 'SN-MO-PPO']:
                env, num_objectives = setup_vector_env(env_id,
                                                       oracle_hyperparams['num_envs'],
                                                       seed,
                                                       run_name,
                                                       max_episode_steps=max_episode_steps,
                                                       one_hot=one_hot,
                                                       capture_video=False)
            else:
                env, num_objectives = setup_env(env_id,
                                                max_episode_steps=max_episode_steps,
                                                one_hot=one_hot,
                                                capture_video=False,
                                                run_name=run_name)

            linear_solver = init_linear_solver('known_box', minimals=minimals, maximals=maximals)
            oracle = init_oracle(algorithm,
                                 env,
                                 gamma,
                                 seed=seed,
                                 **parameters['oracle'],
                                 **oracle_hyperparams)
            ol = init_outer_loop(method,
                                 env,
                                 num_objectives,
                                 oracle,
                                 linear_solver,
                                 ref_point=ref_point,
                                 exp_name=run_name,
                                 wandb_project_name=wandb_project_name,
                                 wandb_entity=wandb_entity,
                                 seed=seed,
                                 **parameters['outer_loop'])
            if report_intermediate:
                def callback(step, hypervolume, dominated_hv, discarded_hv, coverage, error):
                    trial.report(hypervolume, step)
                    if trial.should_prune():
                        raise optuna.TrialPruned()
            else:
                callback = None
            ol.solve(callback=callback)
            hypervolumes.append(ol.hv)
        return np.mean(hypervolumes)

    if isinstance(parameters['env_id'], str):
        env_name = parameters['env_id']
    else:
        parameters['env_name'] = np.random.choice(parameters['env_id'])
        env_name = None

    return optimize_hyperparameters(study_name, env_name, optimize_trial, n_trials=n_trials, log_dir=log_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run a hyperparameter search study')
    parser.add_argument('--params', type=str, default='sn_dqn_dst.yaml',
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
           log_dir=args.log_dir,
           )
