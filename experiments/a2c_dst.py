import random
import torch
import argparse

import numpy as np

from experiments import setup_env
from linear_solvers import init_linear_solver
from oracles import init_oracle
from outer_loops import init_outer_loop


def parse_args():
    parser = argparse.ArgumentParser()

    # Logging arguments.
    parser.add_argument("--track", action="store_true", default=False, help="Track the experiments using wandb")
    parser.add_argument("--log_dir", type=str, default="logs", help="Directory where to save the logs")
    parser.add_argument("--wandb-project-name", type=str, default="cones", help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None, help="the entity (team) of wandb's project")
    parser.add_argument("--save_figs", type=bool, default=False, help="Whether to save figures. Only used in 2D")
    parser.add_argument("--log_freq", type=int, default=1000,
                        help="The frequency (in number of steps) at which to log the results.")

    # General arguments.
    parser.add_argument("--seed", type=int, default=1, help="The random seed.")
    parser.add_argument('--outer_loop', type=str, default='2D', help='The outer loop to use.')
    parser.add_argument("--oracle", type=str, default="MO-A2C", help="The algorithm to use.")
    parser.add_argument("--aug", type=float, default=0.005, help="The augmentation term in the utility function.")
    parser.add_argument("--env", type=str, default="deep-sea-treasure-concave-v0", help="The game to use.")
    parser.add_argument("--tolerance", type=float, default="1e-4", help="The tolerance for the outer loop.")
    parser.add_argument("--warm_start", type=bool, default=False, help="Whether to warm start the inner loop.")
    parser.add_argument("--global_steps", type=int, default=50000,
                        help="The total number of steps to run the experiment.")
    parser.add_argument("--eval_episodes", type=int, default=100, help="The number of episodes to use for evaluation.")
    parser.add_argument("--gamma", type=float, default=1., help="The discount factor.")

    # Oracle arguments.
    parser.add_argument("--lrs", nargs='+', type=float, default=(0.0003, 0.001),
                        help="The learning rates for the models.")
    parser.add_argument("--hidden_layers", nargs='+', type=tuple, default=((64, 64), (64, 64),),
                        help="The hidden layers for the model.")
    parser.add_argument("--one_hot", type=bool, default=True, help="Whether to use a one hot state encoding.")

    # Model based arguments.
    parser.add_argument("--model_based", type=bool, default=False, help="Whether to use a model-based DQN.")
    parser.add_argument("--model_lr", type=float, default=0.001, help="The learning rate for the model.")
    parser.add_argument("--model_hidden_layers", type=tuple, default=(64, 64), help="The hidden layers for the model.")
    parser.add_argument("--model_steps", type=int, default=32,
                        help="The number of steps to take for each model training step.")
    parser.add_argument("--model_train_finish", type=int, default=10000,
                        help="The number of steps after which the model training is finished.")
    parser.add_argument("--pe_size", type=int, default=5, help="The size of the policy ensemble.")

    # MO-A2C specific arguments.
    parser.add_argument("--e_coef", type=float, default=0.01, help="The entropy coefficient for A2C.")
    parser.add_argument("--v_coef", type=float, default=0.5, help="The value coefficient for A2C.")
    parser.add_argument("--max_grad_norm", type=float, default=50,
                        help="The maximum norm for the gradient clipping.")
    parser.add_argument("--normalize_advantage", type=bool, default=False,
                        help="Whether to normalize the advantages in A2C.")
    parser.add_argument("--n_steps", type=int, default=10, help="The number of steps for the n-step A2C.")
    parser.add_argument("--gae_lambda", type=float, default=0.5, help="The lambda parameter for the GAE.")

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    # Seeding
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    env, num_objectives = setup_env(args)
    linear_solver = init_linear_solver('known_box',
                                       nadirs=[np.array([0., 0.]), np.array([124.0, -19.])],
                                       ideals=[np.array([124.0, -19.]), np.array([0., 0.])])
    oracle = init_oracle(args.oracle,
                         env,
                         aug=args.aug,
                         lrs=args.lrs,
                         hidden_layers=args.hidden_layers,
                         one_hot=args.one_hot,
                         e_coef=args.e_coef,
                         v_coef=args.v_coef,
                         gamma=args.gamma,
                         max_grad_norm=args.max_grad_norm,
                         normalize_advantage=args.normalize_advantage,
                         n_steps=args.n_steps,
                         gae_lambda=args.gae_lambda,
                         global_steps=args.global_steps,
                         eval_episodes=args.eval_episodes,
                         log_freq=args.log_freq,
                         seed=args.seed
                         )
    ol = init_outer_loop(args.outer_loop,
                         env,
                         num_objectives,
                         oracle,
                         linear_solver,
                         warm_start=args.warm_start,
                         seed=args.seed)
    pf = ol.solve()

    print("Pareto front:")
    for point in pf:
        print(point)
