import mo_gymnasium as mo_gym
from environments import setup_env
from environments.bounding_boxes import get_bounding_box
from morl_baselines.multi_policy.pcn.pcn import PCN
from morl_baselines.multi_policy.gpi_pd.gpi_pd import GPILS
from morl_baselines.multi_policy.envelope.envelope import Envelope


def select_gamma(env_id):
    if env_id == 'minecart-v0':
        return 0.98
    elif env_id == 'mo-reacher-v4':
        return 0.99
    elif env_id == 'deep-sea-treasure-concave-v0':
        return 1.0
    else:
        raise NotImplementedError


def setup_agent(alg_id, env):
    if alg_id == 'GPI-LS':
        agent = GPILS(
            env,
            per=True,
            initial_epsilon=1.0,
            final_epsilon=0.05,
            epsilon_decay_steps=200000,
            target_net_update_freq=200,
            gradient_updates=10
        )
        kwargs = {}
    elif alg_id == 'PCN':
        agent = PCN(
            env,
            per=True,
            initial_epsilon=1.0,
            final_epsilon=0.05,
            epsilon_decay_steps=200000,
            target_net_update_freq=200,
            gradient_updates=10
        )
        kwargs = {}
    elif alg_id == 'Envelope':
        agent = Envelope(
            env,
            per=True,
            initial_epsilon=1.0,
            final_epsilon=0.05,
            epsilon_decay_steps=200000,
            target_net_update_freq=200,
            gradient_updates=10
        )
        kwargs = {}
    else:
        raise NotImplementedError
    return agent, kwargs


def run_baseline(alg_id, env_id):
    gamma = select_gamma(env_id)
    env = setup_env(env_id, gamma=gamma, max_episode_steps=1000, one_hot=False, capture_video=False, run_name='test')
    eval_env = mo_gym.make(env_id)
    agent, kwargs = setup_agent(alg_id, env)

    _, _, ref_point = get_bounding_box(env_id)

    agent.train(**kwargs)
