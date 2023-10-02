import os

import gymnasium as gym
import mo_gymnasium as mo_gym

from gymnasium.wrappers import TimeLimit

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def default_timesteps(env_id):
    if env_id == 'deep-sea-treasure-concave-v0':
        max_episode_steps = 50
    elif env_id == 'mo-reacher-v4':
        max_episode_steps = 50
    elif env_id == 'minecart-v0':
        max_episode_steps = 1000
    else:
        raise NotImplementedError
    return max_episode_steps


def setup_env(env_id, max_episode_steps, capture_video=False, run_name='run'):
    """Setup the environment."""
    env = mo_gym.make(env_id)
    if max_episode_steps is not None:
        env = TimeLimit(env, max_episode_steps=max_episode_steps)
    if capture_video:
        env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
    env = mo_gym.MORecordEpisodeStatistics(env)
    env.env_id = env_id
    return env, env.reward_space.shape[0]


def make_env(env_id, idx, seed, run_name, capture_video, max_episode_steps=None):
    """A function used in the vectorised environment generation."""

    def thunk():
        env, _ = setup_env(env_id, max_episode_steps, capture_video=capture_video and idx == 0, run_name=run_name)
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


def setup_vector_env(env_id, num_envs, seed, run_name, capture_video, max_episode_steps=None):
    """Setup a vectorised environment."""
    envs = []
    for i in range(num_envs):
        envs.append(make_env(env_id, i, seed + i, run_name, capture_video, max_episode_steps=max_episode_steps))
    envs = mo_gym.MOSyncVectorEnv(envs)
    envs = mo_gym.MORecordEpisodeStatistics(envs)
    envs.env_id = env_id
    return envs, envs.reward_space.shape[0]
