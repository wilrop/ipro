import os
import argparse
import time
import optuna
import joblib
import numpy as np

from utils.helpers import strtobool
from environments import setup_vector_env
from linear_solvers import init_linear_solver
from oracles import init_oracle
from outer_loops import init_outer_loop
from torch.utils.tensorboard import SummaryWriter


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
    parser.add_argument("--env_id", type=str, default="mo-highway-v0", help="The game to use.")
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
    parser.add_argument("--max_episode_steps", type=int, default=100, help="The maximum number of steps per episode.")

    # Oracle arguments.
    parser.add_argument("--lrs", nargs='+', type=float, default=(0.0002, 0.0002),
                        help="The learning rates for the models.")
    parser.add_argument("--hidden_layers", nargs='+', type=tuple, default=((64, 64), (64, 64),),
                        help="The hidden layers for the model.")
    parser.add_argument("--one_hot", type=bool, default=False, help="Whether to use a one hot state encoding.")

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


def objective(trial):
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    aug = trial.suggest_float("aug", 0.001, 0.01, log=True)
    scale = trial.suggest_float("scale", 10, 1000, log=True)
    lr_actor = trial.suggest_float("lrs", 0.0001, 0.001, log=True)
    lr_critic = trial.suggest_float("lrs", 0.0001, 0.001, log=True)
    e_coef = trial.suggest_float("e_coef", 0.001, 0.1, log=True)
    v_coef = trial.suggest_float("v_coef", 0.1, 1.0)
    max_grad_norm = trial.suggest_float("max_grad_norm", 0.5, 50.0, log=True)
    normalize_advantage = trial.suggest_categorical("normalize_advantage", [True, False])
    anneal_lr = trial.suggest_categorical("anneal_lr", [True, False])
    n_steps = trial.suggest_categorical("n_steps", [32, 64, 128])
    num_envs = trial.suggest_categorical("num_envs", [4, 8, 16])
    num_minibatches = trial.suggest_categorical("num_mini_batches", [4, 8, 16])
    update_epochs = trial.suggest_categorical("update_epochs", [4, 8, 16])
    clip_coef = trial.suggest_float("clip_coef", 0.1, 0.5)
    clip_range_vf = trial.suggest_float("clip_range_vf", 0., 0.5)
    args.num_envs = num_envs

    envs, num_objectives = setup_vector_env(args, run_name)
    linear_solver = init_linear_solver('known_box',
                                       nadirs=[np.array([0., 100.0]), np.array([100.0, 0.])],
                                       ideals=[np.array([100.0, 0.]), np.array([0., 100.0])])
    oracle = init_oracle(args.oracle,
                         envs,
                         writer,
                         aug=aug,
                         scale=scale,
                         gamma=args.gamma,
                         lrs=(lr_actor, lr_critic),
                         eps=args.eps,
                         hidden_layers=args.hidden_layers,
                         one_hot=args.one_hot,
                         anneal_lr=anneal_lr,
                         e_coef=e_coef,
                         v_coef=v_coef,
                         num_envs=num_envs,
                         num_minibatches=num_minibatches,
                         update_epochs=update_epochs,
                         max_grad_norm=max_grad_norm,
                         target_kl=args.target_kl,
                         normalize_advantage=normalize_advantage,
                         clip_coef=clip_coef,
                         clip_range_vf=clip_range_vf,
                         gae_lambda=args.gae_lambda,
                         n_steps=n_steps,
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
                         writer,
                         warm_start=args.warm_start,
                         seed=args.seed)
    ol.solve()
    return ol.dominated_hv


if __name__ == '__main__':
    args = parse_args()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=10)
    joblib.dump(study, "opt/study_ppo_highway.pkl")
    print(study.best_params)
