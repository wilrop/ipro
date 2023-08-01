import logging
import os
import sys
import yaml
import time
import argparse
import optuna
import wandb
import numpy as np

from torch.utils.tensorboard import SummaryWriter
from optuna._callbacks import RetryFailedTrialCallback

from environments import setup_env, setup_vector_env
from linear_solvers import init_linear_solver
from oracles import init_oracle
from outer_loops import init_outer_loop


def optimize_hyperparameters(study_name, env_name, optimize_trial, storage=None, n_trials=100):
    # Add stream handler of stdout to show the messages
    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))

    if storage is None:
        if not os.path.exists(os.path.join('studies')):
            os.makedirs(os.path.join('studies'))
        storage = f'sqlite:///studies/{study_name}.db'

    sqlite_timeout = 300
    storage = optuna.storages.RDBStorage(
        storage,
        engine_kwargs={
            'connect_args': {'timeout': sqlite_timeout},
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


def suggest_hyperparameters(trial, parameters):
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

    for param, config in parameters['hyperparameters'].items():
        if param != 'conditionals':
            assert 'type' in config, f'Please provide a suggestion type for parameter {param}'
            suggestion_type = config['type']
            _suggest(param, config, suggestion_type)

    def _str_replace(str_, dict_):
        for word, replacement in dict_.items():
            str_ = str_.replace(word, str(replacement))
        return str_

    conditionals = []
    conditional_params = parameters['hyperparameters'].get('conditionals', [])
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


def search(
        parameters,
        study_name='study',
        n_trials=100,
):
    def optimize_trial(trial):
        env_id = parameters['env_id']
        seed = parameters['seed']
        oracle_name = parameters['oracle']
        outer_loop_name = parameters['outer_loop']
        max_episode_steps = parameters['max_episode_steps']
        run_name = f"{study_name}__{seed}__{int(time.time())}"

        if parameters['track']:
            wandb.init(
                project=parameters['wandb_project_name'],
                entity=parameters['wandb_entity'],
                sync_tensorboard=True,
                config=parameters,
                name=run_name,
                monitor_gym=False,
                save_code=True,
            )
        writer = SummaryWriter(f"runs/{run_name}")
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in parameters.items()])),
        )

        nadirs = np.array(parameters['nadirs'])
        ideals = np.array(parameters['ideals'])
        hyperparameters = suggest_hyperparameters(trial, parameters)
        hidden_layers = (hyperparameters['hidden_size'],) * hyperparameters['num_hidden_layers']
        hyperparameters.pop('hidden_size')
        hyperparameters.pop('num_hidden_layers')
        if oracle_name == 'MO-DQN':
            hyperparameters['hidden_layers'] = hidden_layers
        else:
            hyperparameters['actor_hidden'] = hidden_layers
            hyperparameters['critic_hidden'] = hidden_layers

        if oracle_name == 'MO-PPO':
            env, num_objectives = setup_vector_env(env_id, hyperparameters['num_envs'], seed, run_name, False,
                                                   max_episode_steps=max_episode_steps)
        else:
            env, num_objectives = setup_env(env_id, max_episode_steps, capture_video=False, run_name=run_name)

        linear_solver = init_linear_solver('known_box', nadirs=nadirs, ideals=ideals)
        oracle = init_oracle(oracle_name,
                             env,
                             parameters['gamma'],
                             writer,
                             seed=seed,
                             **hyperparameters)
        ol = init_outer_loop(outer_loop_name,
                             env,
                             num_objectives,
                             oracle,
                             linear_solver,
                             writer,
                             warm_start=parameters['warm_start'],
                             seed=seed)
        ol.solve()
        writer.close()
        wandb.finish()
        return ol.dominated_hv

    if type(parameters['env_id']) == str:
        env_name = parameters['env_id']
    else:
        parameters['env_name'] = np.random.choice(parameters['env_id'])
        env_name = None

    return optimize_hyperparameters(study_name, env_name, optimize_trial, n_trials=n_trials)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run a hyperparameter search study')
    parser.add_argument('--params', type=str, default='ppo_highway.yaml',
                        help='path of a yaml file containing the parameters of this study')
    args = parser.parse_args()

    with open(args.params, 'r') as file:
        parameters = yaml.safe_load(file)

    search(parameters, parameters.get('study_name', 'IPRO_study'), parameters['n_trials'])
