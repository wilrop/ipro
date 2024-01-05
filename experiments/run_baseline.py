import json
import argparse
import numpy as np
import mo_gymnasium as mo_gym

from mo_gymnasium.utils import MORecordEpisodeStatistics
from experiments.reproduce_experiment import get_env_info
from environments.bounding_boxes import get_bounding_box
from morl_baselines.multi_policy.pcn.pcn import PCN
from morl_baselines.multi_policy.gpi_pd.gpi_pd import GPILS
from morl_baselines.multi_policy.envelope.envelope import Envelope


def get_timesteps(alg_id, env_id):
    if alg_id == 'PCN' and env_id == 'deep-sea-treasure-concave-v0':
        total_timesteps = 100000
    elif alg_id == 'PCN' and env_id == 'minecart-v0':
        total_timesteps = 2000000
    elif alg_id == 'PCN' and env_id == 'mo-reacher-v4':
        total_timesteps = 200000
    elif alg_id == 'GPI-PD' and env_id == 'deep-sea-treasure-concave-v0':
        total_timesteps = 100000
    elif alg_id == 'GPI-PD' and env_id == 'minecart-v0':
        total_timesteps = 100000
    elif alg_id == 'GPI-PD' and env_id == 'mo-reacher-v4':
        total_timesteps = 200000
    elif alg_id == 'Envelope' and env_id == 'deep-sea-treasure-concave-v0':
        total_timesteps = 100000
    elif alg_id == 'Envelope' and env_id == 'minecart-v0':
        total_timesteps = 400000
    elif alg_id == 'Envelope' and env_id == 'mo-reacher-v4':
        total_timesteps = 200000
    else:
        raise NotImplementedError
    return total_timesteps


def setup_agent(alg_id, env, gamma, seed, run_name):
    """Setup the agent using MORL-baselines."""
    if alg_id == 'GPI-PD':
        agent = GPILS(env,
                      gamma=gamma,
                      experiment_name=run_name,
                      seed=seed)
    elif alg_id == 'PCN':
        agent = PCN(env,
                    scaling_factor=np.array([0.1, 0.1, 0.01]),
                    gamma=gamma,
                    experiment_name=run_name,
                    seed=seed)
    elif alg_id == 'Envelope':
        agent = Envelope(env,
                         gamma=gamma,
                         experiment_name=run_name,
                         seed=seed)
    else:
        raise NotImplementedError
    return agent


def run_baseline(exp_id, exp_dir):
    """Run a baseline on the environment."""
    id_exp_dict = json.load(open(f'{exp_dir}/baselines.json', 'r'))
    baseline, env_id, seed = id_exp_dict[str(exp_id)]
    gamma, max_episode_steps, one_hot_wrapper, _ = get_env_info(env_id)
    _, _, ref_point = get_bounding_box(env_id)
    total_timesteps = get_timesteps(baseline, env_id)
    run_name = f'{baseline}__{env_id}__{seed}'

    env = mo_gym.make(env_id)
    env = MORecordEpisodeStatistics(env, gamma=gamma)  # wrapper for recording statistics
    eval_env = mo_gym.make(env_id)  # environment used for evaluation

    agent = setup_agent(baseline, env, gamma, seed, run_name)
    agent.train(total_timesteps, eval_env, ref_point)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run baseline.')
    parser.add_argument('--exp_id', type=str, default=1)
    parser.add_argument('--exp_dir', type=str, default='./evaluation')
    args = parser.parse_args()

    run_baseline(args.exp_id, args.exp_dir)
