import os
import random
import torch
import argparse
import time

import numpy as np

from utils.helpers import strtobool
from environments import setup_env
from linear_solvers import init_linear_solver
from oracles import init_oracle
from outer_loops import init_outer_loop


def parse_args():
    parser = argparse.ArgumentParser()

    # Logging arguments.
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
                        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1, help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
                        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
                        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="PRIOL", help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None, help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
                        help="whether to capture videos of the agent performances (check out `videos` folder)")
    parser.add_argument("--log-freq", type=int, default=10000, help="the logging frequency")

    # General arguments.
    parser.add_argument("--env_id", type=str, default="mo-highway-v0", help="The environment to use.")
    parser.add_argument('--outer_loop', type=str, default='2D', help='The outer loop to use.')
    parser.add_argument("--oracle", type=str, default="MO-A2C", help="The algorithm to use.")
    parser.add_argument("--aug", type=float, default=0.005, help="The augmentation term in the utility function.")
    parser.add_argument("--scale", type=float, default=1000, help="The scale of the utility function.")
    parser.add_argument("--tolerance", type=float, default="1e-4", help="The tolerance for the outer loop.")
    parser.add_argument("--warm_start", type=bool, default=False, help="Whether to warm start the inner loop.")
    parser.add_argument("--global_steps", type=int, default=100000,
                        help="The total number of steps to run the experiment.")
    parser.add_argument("--eval_episodes", type=int, default=1, help="The number of episodes to use for evaluation.")
    parser.add_argument("--gamma", type=float, default=1., help="The discount factor.")
    parser.add_argument("--max_episode_steps", type=int, default=30, help="The maximum number of steps per episode.")

    # Oracle arguments.
    parser.add_argument("--lr_actor", type=float, default=0.001, help="The learning rate for the actor.")
    parser.add_argument("--lr_critic", type=float, default=0.001, help="The learning rate for the critic.")
    parser.add_argument("--actor_hidden", type=tuple, default=(64, 64), help="The hidden layers for the actor.")
    parser.add_argument("--critic_hidden", type=tuple, default=(64, 64), help="The hidden layers for the critic.")
    parser.add_argument("--one_hot", type=bool, default=False, help="Whether to use a one hot state encoding.")

    # MO-A2C specific arguments.
    parser.add_argument("--e_coef", type=float, default=0.1, help="The entropy coefficient for A2C.")
    parser.add_argument("--v_coef", type=float, default=0.5, help="The value coefficient for A2C.")
    parser.add_argument("--max_grad_norm", type=float, default=5.,
                        help="The maximum norm for the gradient clipping.")
    parser.add_argument("--normalize_advantage", type=bool, default=False,
                        help="Whether to normalize the advantages in A2C.")
    parser.add_argument("--n_steps", type=int, default=10, help="The number of steps for the n-step A2C.")
    parser.add_argument("--gae_lambda", type=float, default=0.95, help="The lambda parameter for the GAE.")

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"

    # Seeding
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    env, num_objectives = setup_env(args.env_id, args.max_episode_steps)
    max_reward = args.max_episode_steps * 1.
    linear_solver = init_linear_solver('known_box',
                                       nadirs=[np.array([0., max_reward]), np.array([max_reward, 0.])],
                                       ideals=[np.array([max_reward, 0.]), np.array([0., max_reward])])
    oracle = init_oracle(args.oracle,
                         env,
                         args.gamma,
                         track=args.track,
                         aug=args.aug,
                         scale=args.scale,
                         lr_actor=args.lr_actor,
                         lr_critic=args.lr_critic,
                         actor_hidden=args.actor_hidden,
                         critic_hidden=args.critic_hidden,
                         one_hot=args.one_hot,
                         e_coef=args.e_coef,
                         v_coef=args.v_coef,
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
                         track=args.track,
                         exp_name=run_name,
                         wandb_project_name=args.wandb_project_name,
                         wandb_entity=args.wandb_entity,
                         warm_start=args.warm_start,
                         seed=args.seed)
    pf = ol.solve()

    print("Pareto front:")
    for point in pf:
        print(point)
