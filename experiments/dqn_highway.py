import os
import random
import torch
import argparse
import time
import wandb

import numpy as np

from utils.helpers import strtobool
from environments import setup_env
from linear_solvers import init_linear_solver
from oracles import init_oracle
from outer_loops import init_outer_loop
from torch.utils.tensorboard import SummaryWriter

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


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
    parser.add_argument("--env_id", type=str, default="mo-highway-v0", help="The game to use.")
    parser.add_argument('--outer_loop', type=str, default='2D', help='The outer loop to use.')
    parser.add_argument("--oracle", type=str, default="MO-DQN", help="The algorithm to use.")
    parser.add_argument("--aug", type=float, default=0.005, help="The augmentation term in the utility function.")
    parser.add_argument("--scale", type=float, default=1000, help="The scale of the utility function.")
    parser.add_argument("--tolerance", type=float, default="1e-4", help="The tolerance for the outer loop.")
    parser.add_argument("--warm_start", type=bool, default=False, help="Whether to warm start the inner loop.")
    parser.add_argument("--global_steps", type=int, default=40000,
                        help="The total number of steps to run the experiment.")
    parser.add_argument("--eval_episodes", type=int, default=1, help="The number of episodes to use for evaluation.")
    parser.add_argument("--gamma", type=float, default=1., help="The discount factor.")
    parser.add_argument("--max_episode_steps", type=int, default=100, help="The maximum number of steps per episode.")

    # Oracle arguments.
    parser.add_argument("--lr", type=float, default=0.0007, help="The learning rates for the models.")
    parser.add_argument("--hidden_layers", type=tuple, default=(64, 64), help="The hidden layers for the model.")
    parser.add_argument("--one_hot", type=bool, default=False, help="Whether to use a one hot state encoding.")

    # MO-DQN specific arguments.
    parser.add_argument("--learning_start", type=int, default=2000,
                        help="The number of steps before starting to train the DQN.")
    parser.add_argument("--train_freq", type=int, default=1, help="The number of steps between two DQN training steps.")
    parser.add_argument("--target_update_freq", type=int, default=1,
                        help="The number of steps between two DQN target network updates.")
    parser.add_argument("--tau", type=float, default=.1,
                        help="The fraction to copy the target network into the Q-network.")
    parser.add_argument("--gradient_steps", type=int, default=1,
                        help="The number of gradient steps to take for each DQN training step.")
    parser.add_argument("--batch_size", type=int, default=32, help="The batch size for the DQN training.")
    parser.add_argument("--buffer_size", type=int, default=10000, help="The size of the replay buffer.")
    parser.add_argument("--per", type=bool, default=True, help="Whether to use prioritized experience replay.")
    parser.add_argument("--alpha_per", type=float, default=0.6,
                        help="The alpha parameter for prioritized experience replay.")
    parser.add_argument("--min_priority", type=float, default=1e-3,
                        help="The minimum priority for prioritized experience replay.")
    parser.add_argument("--epsilon_start", type=float, default=1.0,
                        help="The initial value of epsilon for the epsilon-greedy exploration.")
    parser.add_argument("--epsilon_end", type=float, default=0.05,
                        help="The final value of epsilon for the epsilon-greedy exploration.")
    parser.add_argument("--exploration_frac", type=float, default=0.5,
                        help="The fraction of the total number of steps during which epsilon is linearly decayed.")

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"

    if args.track:
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # Seeding
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    env, num_objectives = setup_env(args.env_id, args.max_episode_steps)
    linear_solver = init_linear_solver('known_box',
                                       nadirs=[np.array([0., 100.0]), np.array([100.0, 0.])],
                                       ideals=[np.array([100.0, 0.]), np.array([0., 100.0])])
    oracle = init_oracle(args.oracle,
                         env,
                         writer,
                         aug=args.aug,
                         scale=args.scale,
                         lr=args.lr,
                         hidden_layers=args.hidden_layers,
                         one_hot=args.one_hot,
                         learning_start=args.learning_start,
                         train_freq=args.train_freq,
                         target_update_freq=args.target_update_freq,
                         gradient_steps=args.gradient_steps,
                         epsilon_start=args.epsilon_start,
                         epsilon_end=args.epsilon_end,
                         exploration_frac=args.exploration_frac,
                         gamma=args.gamma,
                         tau=args.tau,
                         buffer_size=args.buffer_size,
                         per=args.per,
                         alpha_per=args.alpha_per,
                         min_priority=args.min_priority,
                         batch_size=args.batch_size,
                         global_steps=args.global_steps,
                         eval_episodes=args.eval_episodes,
                         log_freq=args.log_freq,
                         seed=args.seed,
                         )
    ol = init_outer_loop(args.outer_loop,
                         env,
                         num_objectives,
                         oracle,
                         linear_solver,
                         writer,
                         warm_start=args.warm_start,
                         seed=args.seed)
    pf = ol.solve()

    print("Pareto front:")
    for point in pf:
        print(point)

    writer.close()
