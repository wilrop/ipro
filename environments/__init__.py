import os

import gymnasium as gym
import mo_gymnasium as mo_gym

from gymnasium.wrappers import TimeLimit
from mo_gymnasium.envs.deep_sea_treasure.deep_sea_treasure import CONCAVE_MAP
from environments.highway_custom import HighwayCustom

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def setup_env(env_id, max_episode_steps):
    """Setup the environment."""
    if env_id == 'deep-sea-treasure-concave-v0':
        env = mo_gym.make('deep-sea-treasure-v0', dst_map=CONCAVE_MAP)
    elif env_id == 'mo-highway-v0':
        env = HighwayCustom()
    else:
        env = mo_gym.make(env_id)
    if max_episode_steps is not None:
        env = TimeLimit(env, max_episode_steps=max_episode_steps)
    env = mo_gym.MORecordEpisodeStatistics(env)
    return env, env.reward_space.shape[0]


def make_env(env_id, idx, seed, run_name, capture_video, max_episode_steps=None):
    def thunk():
        env, _ = setup_env(env_id, max_episode_steps)
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        try:
            env.seed(seed)
        except AttributeError:
            pass
        try:
            env.action_space.seed(seed)
        except AttributeError:
            pass
        try:
            env.observation_space.seed(seed)
        except AttributeError:
            pass
        return env

    return thunk


def setup_vector_env(args, run_name):
    envs = mo_gym.MOSyncVectorEnv(
        [make_env(args.env_id, i, args.seed + i, run_name, args.capture_video, max_episode_steps=args.max_episode_steps)
         for i in range(args.num_envs)]
    )
    envs = mo_gym.MORecordEpisodeStatistics(envs)
    return envs, envs.reward_space.shape[0]
