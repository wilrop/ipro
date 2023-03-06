import argparse
import random

import gymnasium as gym
import mo_gymnasium as mo_gym
import numpy as np
import torch
from gymnasium.wrappers import TimeLimit
from mo_gymnasium.envs.deep_sea_treasure.deep_sea_treasure import CONCAVE_MAP

from mo_dqn import MODQN
from outer_loop import outer_loop


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--track", action="store_true", default=False, help="Track the experiments using wandb")
    parser.add_argument("--log_dir", type=str, default="logs", help="Directory where to save the logs")
    parser.add_argument("--wandb-project-name", type=str, default="cones", help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None, help="the entity (team) of wandb's project")
    parser.add_argument("--algorithm", type=str, default="MO-DQN", help="The algorithm to use.")
    parser.add_argument("--env", type=str, default="deep-sea-treasure-concave-v0", help="The game to use.")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="The learning rate.")
    parser.add_argument("--learning_start", type=int, default=5000,
                        help="The number of global steps before starting the training.")
    parser.add_argument("--train_freq", type=int, default=10,
                        help="The number of global steps between two training steps.")
    parser.add_argument("--target_update_interval", type=int, default=500,
                        help="The number of global steps between two target network updates.")
    parser.add_argument("--epsilon", type=float, default=1.0, help="The initial value of epsilon.")
    parser.add_argument("--epsilon_decay", type=float, default=0.9999, help="The decay of epsilon.")
    parser.add_argument("--final_epsilon", type=float, default=0.05, help="The final value of epsilon.")
    parser.add_argument("--gamma", type=float, default=1., help="The discount factor.")
    parser.add_argument("--tau", type=float, default=1., help="The soft update factor.")
    parser.add_argument("--hidden_layers", type=int, nargs='+', default=[120, 84],
                        help="The sizes of the hidden layers.")
    parser.add_argument("--buffer_size", type=int, default=10000, help="The size of the replay buffer.")
    parser.add_argument("--batch_size", type=int, default=128, help="The size of the batch.")
    parser.add_argument("--num_train_episodes", type=int, default=20000, help="The number of training episodes.")
    parser.add_argument("--num_eval_episodes", type=int, default=10, help="The number of evaluation episodes.")
    parser.add_argument("--log_every", type=int, default=1000, help="The number of episodes between two logs.")
    parser.add_argument("--seed", type=int, default=42, help="The seed for random number generation.")
    parser.add_argument("--save_figs", type=bool, default=False, help="Whether to save figures.")
    parser.add_argument("--tolerance", type=float, default="1e-4", help="The tolerance for the outer loop.")
    args = parser.parse_args()
    return args


def init_inner_loop(args):
    if args.algorithm == "MO-DQN":
        inner_loop = MODQN(env,
                           args.learning_rate,
                           args.learning_start,
                           args.train_freq,
                           args.target_update_interval,
                           args.epsilon,
                           args.epsilon_decay,
                           args.final_epsilon,
                           args.gamma,
                           args.tau,
                           args.hidden_layers,
                           args.buffer_size,
                           args.batch_size,
                           args.num_train_episodes,
                           args.num_eval_episodes,
                           args.log_every,
                           args.seed)
    else:
        raise ValueError("Unknown algorithm: {}".format(args.algorithm))
    return inner_loop


def setup_env(env_name):
    if env_name == 'deep-sea-treasure-v0':
        env = gym.make(env_name, float_state=True)
    elif env_name == 'deep-sea-treasure-concave-v0':
        # Pareto front:
        # (0.0, 0.0),
        # (1.0, -1.0),
        # (2.0, -3.0),
        # (3.0, -5.0),
        # (5.0, -7.0),
        # (8.0, -8.0),
        # (16.0, -9.0),
        # (24.0, -13.0),
        # (50.0, -14.0),
        # (74.0, -17.0),
        # (124.0, -19.0),
        env = mo_gym.make('deep-sea-treasure-v0', float_state=True, dst_map=CONCAVE_MAP)
    elif env_name == 'mo-mountaincar-v0':
        env = mo_gym.make(env_name)
    else:
        raise ValueError("Unknown environment: {}".format(env_name))
    env = TimeLimit(env, max_episode_steps=50)
    return env


if __name__ == '__main__':
    args = parse_args()

    # Seeding
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    env = setup_env(args.env)
    inner_loop = init_inner_loop(args)
    linear_solver = lambda _, x: np.array([124.0, -19.0]) if x[0] == 1 else np.array([0., 0.])
    pf = outer_loop(env,
                    inner_loop,
                    linear_solver,
                    tolerance=args.tolerance,
                    save_figs=args.save_figs,
                    log_dir=args.log_dir)

    print("Pareto front:")
    for point in pf:
        print(point)
