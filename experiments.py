import argparse

import gymnasium as gym
import mo_gymnasium as mo_gym
from outer_loop import outer_loop
from mo_dqn import MODQN
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--track", action="store_true", default=False, help="Track the experiments using wandb")
    parser.add_argument("--log_dir", type=str, default="logs", help="Directory where to save the logs")
    parser.add_argument("--wandb-project-name", type=str, default="cones", help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None, help="the entity (team) of wandb's project")
    parser.add_argument("--algorithm", type=str, default="MO-DQN", help="The algorithm to use.")
    parser.add_argument("--env", type=str, default="MO-Mountaincarcontinuous", help="The game to use.")
    parser.add_argument("--learning_rate", type=float, default=3e-3, help="The learning rate.")
    parser.add_argument("--learning_start", type=int, default=1000, help="The number of steps before starting the training.")
    parser.add_argument("--train_freq", type=int, default=1, help="The number of steps between two training steps.")
    parser.add_argument("--target_update_interval", type=int, default=1000, help="The number of steps between two target network updates.")
    parser.add_argument("--epsilon", type=float, default=1.0, help="The initial value of epsilon.")
    parser.add_argument("--epsilon_decay", type=float, default=0.999, help="The decay of epsilon.")
    parser.add_argument("--gamma", type=float, default=0.99, help="The discount factor.")
    parser.add_argument("--tau", type=float, default=0.001, help="The soft update factor.")
    parser.add_argument("--hidden_layers", type=int, nargs='+', default=[120, 84], help="The sizes of the hidden layers.")
    parser.add_argument("--buffer_size", type=int, default=100000, help="The size of the replay buffer.")
    parser.add_argument("--batch_size", type=int, default=32, help="The size of the batch.")
    parser.add_argument("--num_train_episodes", type=int, default=1000, help="The number of training episodes.")
    parser.add_argument("--num_eval_episodes", type=int, default=100, help="The number of evaluation episodes.")
    parser.add_argument("--log_every", type=int, default=10, help="The number of episodes between two logs.")
    parser.add_argument("--seed", type=int, default=42, help="The seed for random number generation.")
    parser.add_argument("--save_figs", type=bool, default=False, help="Whether to save figures.")
    parser.add_argument("tolerance", type=float, default="1e-4", help="The tolerance for the outer loop.")
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


if __name__ == '__main__':
    args = parse_args()
    env = mo_gym.make(args.env)  # Ensure positive rewards
    inner_loop = init_inner_loop(args)
    linear_solver = lambda x: np.array([1, 0]) if x[0] == 1 else np.array([0, 1])
    pf = outer_loop(env,
                    inner_loop,
                    linear_solver,
                    tolerance=args.tolerance,
                    save_figs=args.save_figs,
                    log_dir=args.log_dir)
