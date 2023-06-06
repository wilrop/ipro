import os
import argparse
import random
import torch

import numpy as np

from experiments import setup_env
from linear_solvers import init_linear_solver
from oracles import init_oracle
from outer_loops import init_outer_loop

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


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
    parser.add_argument("--oracle", type=str, default="MO-DQN", help="The algorithm to use.")
    parser.add_argument("--aug", type=float, default=0.02, help="The augmentation term in the utility function.")
    parser.add_argument("--env", type=str, default="deep-sea-treasure-concave-v0", help="The game to use.")
    parser.add_argument("--tolerance", type=float, default="1e-4", help="The tolerance for the outer loop.")
    parser.add_argument("--warm_start", type=bool, default=False, help="Whether to warm start the inner loop.")
    parser.add_argument("--global_steps", type=int, default=40000,
                        help="The total number of steps to run the experiment.")
    parser.add_argument("--eval_episodes", type=int, default=100, help="The number of episodes to use for evaluation.")
    parser.add_argument("--gamma", type=float, default=1., help="The discount factor.")

    # Oracle arguments.
    parser.add_argument("--lr", type=float, default=0.001, help="The learning rates for the models.")
    parser.add_argument("--hidden_layers", type=tuple, default=(64, 64), help="The hidden layers for the model.")
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
    parser.add_argument("--init_real_frac", type=float, default=0.8,
                        help="The initial fraction of real data to use for the model training.")
    parser.add_argument("--final_real_frac", type=float, default=0.1,
                        help="The final fraction of real data to use for the model training.")
    parser.add_argument("--buffer_size", type=int, default=100000, help="The size of the replay buffer.")
    parser.add_argument("--per", type=bool, default=True, help="Whether to use prioritized experience replay.")
    parser.add_argument("--alpha_per", type=float, default=0.6,
                        help="The alpha parameter for prioritized experience replay.")
    parser.add_argument("--min_priority", type=float, default=1e-3,
                        help="The minimum priority for prioritized experience replay.")
    parser.add_argument("--epsilon_start", type=float, default=1.0,
                        help="The initial value of epsilon for the epsilon-greedy exploration.")
    parser.add_argument("--epsilon_end", type=float, default=0.1,
                        help="The final value of epsilon for the epsilon-greedy exploration.")
    parser.add_argument("--exploration_frac", type=float, default=0.2,
                        help="The fraction of the total number of steps during which epsilon is linearly decayed.")

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
                         model_based=args.model_based,
                         model_lr=args.model_lr,
                         model_hidden_layers=args.model_hidden_layers,
                         model_steps=args.model_steps,
                         pe_size=args.pe_size,
                         buffer_size=args.buffer_size,
                         per=args.per,
                         alpha_per=args.alpha_per,
                         min_priority=args.min_priority,
                         batch_size=args.batch_size,
                         init_real_frac=args.init_real_frac,
                         final_real_frac=args.final_real_frac,
                         model_train_finish=args.model_train_finish,
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
                         warm_start=args.warm_start,
                         seed=args.seed)
    pf = ol.solve()

    print("Pareto front:")
    for point in pf:
        print(point)
