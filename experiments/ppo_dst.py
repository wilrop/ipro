import os
import random
import torch
import argparse
import time

import numpy as np

from utils.helpers import strtobool
from environments import setup_vector_env
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
    parser.add_argument("--window_size", type=int, default=20, help="the moving average window size")

    # General arguments.
    parser.add_argument("--env_id", type=str, default="deep-sea-treasure-concave-v0", help="The game to use.")
    parser.add_argument('--outer_loop', type=str, default='2D', help='The outer loop to use.')
    parser.add_argument("--oracle", type=str, default="MO-PPO", help="The algorithm to use.")
    parser.add_argument("--aug", type=float, default=0.005, help="The augmentation term in the utility function.")
    parser.add_argument("--scale", type=float, default=1000, help="The scale of the utility function.")
    parser.add_argument("--tolerance", type=float, default="1e-4", help="The tolerance for the outer loop.")
    parser.add_argument("--warm_start", type=bool, default=False, help="Whether to warm start the inner loop.")
    parser.add_argument("--global_steps", type=int, default=300000,
                        help="The total number of steps to run the experiment.")
    parser.add_argument("--eval_episodes", type=int, default=1, help="The number of episodes to use for evaluation.")
    parser.add_argument("--gamma", type=float, default=1., help="The discount factor.")
    parser.add_argument("--max_episode_steps", type=int, default=50, help="The maximum number of steps per episode.")

    # Oracle arguments.
    parser.add_argument("--lr_actor", type=float, default=0.0002, help="The learning rate for the actor.")
    parser.add_argument("--lr_critic", type=float, default=0.0002, help="The learning rate for the critic.")
    parser.add_argument("--actor_hidden", type=tuple, default=(64, 64), help="The hidden layers for the actor.")
    parser.add_argument("--critic_hidden", type=tuple, default=(64, 64), help="The hidden layers for the critic.")
    parser.add_argument("--one_hot", type=bool, default=True, help="Whether to use a one hot state encoding.")

    # MO-PPO specific arguments.
    parser.add_argument("--anneal_lr", type=bool, default=False, help="Whether to anneal the learning rate.")
    parser.add_argument("--e_coef", type=float, default=0.01, help="The entropy coefficient for PPO.")
    parser.add_argument("--v_coef", type=float, default=0.25, help="The value coefficient for PPO.")
    parser.add_argument("--num_envs", type=int, default=4, help="The number of environments to use.")
    parser.add_argument("--num_minibatches", type=int, default=4, help="The number of minibatches to use.")
    parser.add_argument("--update_epochs", type=int, default=4, help="The number of epochs to use for the update.")
    parser.add_argument("--max_grad_norm", type=float, default=0.5,
                        help="The maximum norm for the gradient clipping.")
    parser.add_argument("--target_kl", type=float, default=None, help="The target KL divergence for PPO.")
    parser.add_argument("--normalize_advantage", type=bool, default=False,
                        help="Whether to normalize the advantages in PPO.")
    parser.add_argument("--clip_coef", type=float, default=0.25, help="The clipping coefficient for PPO.")
    parser.add_argument("--clip_range_vf", type=bool, default=0.25, help="Whether to clip the value loss in PPO.")
    parser.add_argument("--n_steps", type=int, default=128, help="The number of steps for the n-step PPO.")
    parser.add_argument("--gae_lambda", type=float, default=.95, help="The lambda parameter for the GAE.")
    parser.add_argument("--eps", type=float, default=1e-8, help="The epsilon parameter for the Adam optimizer.")

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"

    # Seeding
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    envs, num_objectives = setup_vector_env(args.env_id,
                                            args.num_envs,
                                            args.seed,
                                            run_name,
                                            args.capture_video,
                                            max_episode_steps=args.max_episode_steps)
    linear_solver = init_linear_solver('known_box',
                                       nadirs=[np.array([0., 0.]), np.array([124.0, -19.])],
                                       ideals=[np.array([124.0, -19.]), np.array([0., 0.])])
    oracle = init_oracle(args.oracle,
                         envs,
                         args.gamma,
                         track=args.track,
                         aug=args.aug,
                         scale=args.scale,
                         lr_actor=args.lr_actor,
                         lr_critic=args.lr_critic,
                         eps=args.eps,
                         actor_hidden=args.actor_hidden,
                         critic_hidden=args.critic_hidden,
                         one_hot=args.one_hot,
                         anneal_lr=args.anneal_lr,
                         e_coef=args.e_coef,
                         v_coef=args.v_coef,
                         num_envs=args.num_envs,
                         num_minibatches=args.num_minibatches,
                         update_epochs=args.update_epochs,
                         max_grad_norm=args.max_grad_norm,
                         target_kl=args.target_kl,
                         normalize_advantage=args.normalize_advantage,
                         clip_coef=args.clip_coef,
                         clip_range_vf=args.clip_range_vf,
                         gae_lambda=args.gae_lambda,
                         n_steps=args.n_steps,
                         global_steps=args.global_steps,
                         eval_episodes=args.eval_episodes,
                         log_freq=args.log_freq,
                         window_size=args.window_size,
                         seed=args.seed,
                         )

    ol = init_outer_loop(args.outer_loop,
                         envs,
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

