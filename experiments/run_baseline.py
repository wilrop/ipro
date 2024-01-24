import json
import argparse
import numpy as np

from environments import setup_env
from experiments.reproduce_experiment import get_env_info
from environments.bounding_boxes import get_bounding_box
from morl_baselines.multi_policy.pcn.pcn import PCN
from morl_baselines.multi_policy.gpi_pd.gpi_pd import GPILS
from morl_baselines.multi_policy.envelope.envelope import Envelope


def get_kwargs(alg_id, env_id):
    """Get the keyword arguments for the baseline."""
    if alg_id == 'PCN' and env_id == 'deep-sea-treasure-concave-v0':
        total_timesteps = 100000
        setup_kwargs = {
            'scaling_factor': np.array([0.1, 0.1, 0.04]),
            'learning_rate': 5e-3,
            'batch_size': 256,
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
    elif alg_id == 'Envelope' and env_id == 'deep-sea-treasure-concave-v0':
        total_timesteps = 100000
        setup_kwargs = {
            'initial_epsilon': 1.0,
            'final_epsilon': 0.05,
            'epsilon_decay_steps': 50000,
            'target_net_update_freq': 200,
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
    else:
        raise NotImplementedError
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
    else:
        raise NotImplementedError
    return agent


def run_baseline(exp_id, exp_dir):
    """Run a baseline on the environment."""
    id_exp_dict = json.load(open(f'{exp_dir}/baselines.json', 'r'))
    baseline, env_id, seed = id_exp_dict[str(exp_id)]
    gamma, max_episode_steps, one_hot_wrapper, _ = get_env_info(env_id)
    _, _, ref_point = get_bounding_box(env_id)
    total_timesteps, setup_kwargs, train_kwargs = get_kwargs(baseline, env_id)

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
    parser.add_argument('--exp_id', type=str, default=2)
    parser.add_argument('--exp_dir', type=str, default='./evaluation')
    args = parser.parse_args()

    run_baseline(args.exp_id, args.exp_dir)
