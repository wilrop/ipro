import os

import gymnasium as gym
import mo_gymnasium as mo_gym

from gymnasium.wrappers import TimeLimit
from mo_gymnasium.envs.deep_sea_treasure.deep_sea_treasure import CONCAVE_MAP

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def setup_env(args):
    """Setup the environment."""
    if args.env == 'deep-sea-treasure-v0':
        env = gym.make(args.env, float_state=args.continuous_state)
    elif args.env == 'deep-sea-treasure-concave-v0':
        env = mo_gym.make('deep-sea-treasure-v0', float_state=args.continuous_state, dst_map=CONCAVE_MAP)
    elif args.env == 'mo-mountaincar-v0':
        env = mo_gym.make(args.env)
    else:
        raise ValueError("Unknown environment: {}".format(args.env))
    env = TimeLimit(env, max_episode_steps=50)
    return env, env.reward_space.shape[0]
