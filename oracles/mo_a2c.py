import torch
import numpy as np
from oracles.drl_oracle import DRLOracle
from oracles.vector_u import create_batched_aasf


class MOA2C(DRLOracle):
    def __init__(self,
                 env,
                 aug=0.2,
                 gamma=0.99,
                 eval_episodes=100,
                 log_freq=1000,
                 seed=0):
        super().__init__(env, aug=aug, gamma=gamma, eval_episodes=eval_episodes)

        self.seed = seed
        self.rng = np.random.default_rng(seed=seed)

    def reset(self):
        pass

    def train(self):
        pass
