import json
import argparse
import numpy as np

import torch
import torch.nn as nn
from environments import setup_env
from experiments.reproduce_experiment import get_env_info
from environments.bounding_boxes import get_bounding_box
from morl_baselines.multi_policy.pcn.pcn import PCN
from morl_baselines.multi_policy.gpi_pd.gpi_pd import GPILS
from morl_baselines.multi_policy.gpi_pd.gpi_pd_continuous_action import GPILSContinuousAction
from morl_baselines.multi_policy.envelope.envelope import Envelope
from morl_baselines.multi_policy.capql.capql import CAPQL


class DSTModel(nn.Module):

    def __init__(self, observation_dim, nA, reward_dim, scaling_factor, hidden_dim=64):
        super(DSTModel, self).__init__()

        self.scaling_factor = nn.Parameter(torch.tensor(scaling_factor).float(), requires_grad=False)
        self.s_emb = nn.Sequential(nn.Linear(121, 64),
                                   nn.Sigmoid())
        self.c_emb = nn.Sequential(nn.Linear(3, 64),
                                   nn.Sigmoid())
        self.fc = nn.Sequential(nn.Linear(64, nA),
                                nn.LogSoftmax(1))

    def forward(self, state, desired_return, desired_horizon):
        c = torch.cat((desired_return, desired_horizon), dim=-1)
        # commands are scaled by a fixed factor
        c = c * self.scaling_factor
        s = self.s_emb(state.float())
        c = self.c_emb(c)
        # element-wise multiplication of state-embedding and command
        log_prob = self.fc(s * c)
        return log_prob


def get_kwargs(alg_id, env_id, min_vals, max_vals):
    """Get the keyword arguments for the baseline."""
    if alg_id == 'PCN' and env_id == 'deep-sea-treasure-concave-v0':
        total_timesteps = 100000
        setup_kwargs = {
            'scaling_factor': np.array([0.1, 0.1, 0.04]),
            'learning_rate': 5e-3,
            'batch_size': 256,
            'model_class': DSTModel,
        }
        train_kwargs = {
            'max_return': np.array([124, -1.0]),
            'max_buffer_size': 200,
            'num_model_updates': 10,
            'num_er_episodes': 50,
        }
    elif alg_id == 'PCN' and env_id == 'minecart-v0':
        total_timesteps = 2000000
        setup_kwargs = {
            'scaling_factor': np.array([1.0, 1.0, 0.1, 0.1]),
        }
        train_kwargs = {
            'max_return': np.array([1.5, 1.5, 0.0]),
            'max_buffer_size': 200
        }
    elif alg_id == 'PCN' and env_id == 'mo-reacher-v4':
        total_timesteps = 200000
        setup_kwargs = {
            'scaling_factor': np.array([0.1, 0.1, 0.1, 0.1, 0.1])
        }
        train_kwargs = {
            'max_return': np.array([50.0, 50.0, 50.0, 50.0]),
            'max_buffer_size': 200
        }
    elif alg_id == 'GPI-LS' and env_id == 'deep-sea-treasure-concave-v0':
        total_timesteps = 100000
        setup_kwargs = {
            'per': False,
            'initial_epsilon': 1.0,
            'final_epsilon': 0.05,
            'epsilon_decay_steps': 50000,
            'target_net_update_freq': 200,
            'gradient_updates': 10
        }
        train_kwargs = {}
    elif alg_id == 'GPI-LS' and env_id == 'minecart-v0':
        total_timesteps = 200000
        setup_kwargs = {
            'per': True,
            'initial_epsilon': 1.0,
            'final_epsilon': 0.05,
            'epsilon_decay_steps': 100000,
            'target_net_update_freq': 200,
            'gradient_updates': 10
        }
        train_kwargs = {}
    elif alg_id == 'GPI-LS' and env_id == 'mo-reacher-v4':
        total_timesteps = 200000
        setup_kwargs = {
            'per': False,
            'initial_epsilon': 1.0,
            'final_epsilon': 0.05,
            'epsilon_decay_steps': 100000,
            'target_net_update_freq': 200,
            'gradient_updates': 10
        }
        train_kwargs = {}
    elif alg_id == 'GPILSContinuousAction' and env_id == 'mo-walker2d-v4':
        total_timesteps = 200000
        setup_kwargs = {
            'per': False,
        }
        train_kwargs = {}
    elif alg_id == 'Envelope' and env_id == 'deep-sea-treasure-concave-v0':
        total_timesteps = 100000
        setup_kwargs = {
            'initial_epsilon': 1.0,
            'final_epsilon': 0.05,
            'epsilon_decay_steps': 50000,
            'target_net_update_freq': 500,
            'num_sample_w': 4,
            'batch_size': 64
        }
        train_kwargs = {}
    elif alg_id == 'Envelope' and env_id == 'minecart-v0':
        total_timesteps = 400000
        setup_kwargs = {
            'per': True,
            'batch_size': 32,
            'buffer_size': 1486469,
            'epsilon_decay_steps': 95463,
            'final_epsilon': 0.5540,
            'final_homotopy_lambda': 0.8728,
            'gradient_updates': 5,
            'homotopy_decay_steps': 51843,
            'initial_epsilon': 0.7293,
            'initial_homotopy_lambda': 0.9021,
            'learning_rate': 0.00024,
            'learning_starts': 167,
            'max_grad_norm': 1.2262,
            'num_sample_w': 3,
            'per_alpha': 0.3036,
            'target_net_update_freq': 3022,
            'tau': 0.1294
        }
        train_kwargs = {}
    elif alg_id == 'Envelope' and env_id == 'mo-reacher-v4':
        total_timesteps = 200000
        setup_kwargs = {
            'initial_epsilon': 1.0,
            'final_epsilon': 0.05,
            'epsilon_decay_steps': 100000,
            'target_net_update_freq': 1000,
            'num_sample_w': 4,
            'batch_size': 64
        }
        train_kwargs = {}
    elif alg_id == 'CAPQL' and env_id == 'mo-walker2d-v4':
        # These parameters are taken from the original paper.
        total_timesteps = 2000000
        setup_kwargs = {
            'learning_rate': 3e-4,
            'tau': 0.005,
            'buffer_size': 1e6,
            'net_arch': [256, 256],
            'batch_size': 128,
            'num_q_nets': 2,
            'alpha': 0.05,
            'learning_starts': 1000,
            'gradient_updates': 1,
        }
        train_kwargs = {}
    else:
        raise NotImplementedError

    min_val = np.min(min_vals, axis=0)
    max_val = np.max(max_vals, axis=0)
    setup_kwargs['min_val'] = min_val
    setup_kwargs['max_val'] = max_val

    return total_timesteps, setup_kwargs, train_kwargs


def setup_agent(alg_id, env, gamma, seed, setup_kwargs):
    """Setup the agent using MORL-baselines."""
    if alg_id == 'GPI-LS':
        agent = GPILS(env,
                      gamma=gamma,
                      seed=seed,
                      **setup_kwargs)
    elif alg_id == 'PCN':
        agent = PCN(env,
                    gamma=gamma,
                    seed=seed,
                    **setup_kwargs)
    elif alg_id == 'Envelope':
        agent = Envelope(env,
                         gamma=gamma,
                         seed=seed,
                         **setup_kwargs)
    elif alg_id == 'GPILSContinuousAction':
        agent = GPILSContinuousAction(env,
                                      gamma=gamma,
                                      seed=seed,
                                      **setup_kwargs)
    elif alg_id == 'CAPQL':
        agent = CAPQL(env,
                      gamma=gamma,
                      seed=seed,
                      **setup_kwargs)
    else:
        raise NotImplementedError
    return agent


def run_baseline(exp_id, exp_dir):
    """Run a baseline on the environment."""
    id_exp_dict = json.load(open(f'{exp_dir}/baselines.json', 'r'))
    baseline, env_id, seed = id_exp_dict[str(exp_id)]
    gamma, max_episode_steps, one_hot_wrapper, _ = get_env_info(env_id)
    min_vals, max_vals, ref_point = get_bounding_box(env_id)
    total_timesteps, setup_kwargs, train_kwargs = get_kwargs(baseline, env_id, min_vals, max_vals)

    if env_id == 'deep-sea-treasure-concave-v0':
        one_hot = True
    else:
        one_hot = False
    env, _ = setup_env(env_id, gamma=gamma, max_episode_steps=max_episode_steps, one_hot=one_hot)
    eval_env, _ = setup_env(env_id, gamma=gamma, max_episode_steps=max_episode_steps, one_hot=one_hot)

    agent = setup_agent(baseline, env, gamma, seed, setup_kwargs)
    agent.train(total_timesteps, eval_env, ref_point, **train_kwargs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run baseline.')
    parser.add_argument('--exp_id', type=str, default=1)
    parser.add_argument('--exp_dir', type=str, default='./evaluation')
    args = parser.parse_args()

    run_baseline(args.exp_id, args.exp_dir)
