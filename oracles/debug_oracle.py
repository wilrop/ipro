import numpy as np


class DebugOracle:
    """An oracle meant for debugging that replays a precomputed Pareto front."""

    def __init__(self, wandb_summary):
        self.wandb_summary = wandb_summary
        self.iter = 0

    def config(self):
        pass

    def init_oracle(self, nadir=None, ideal=None):
        pass

    def solve(self, referent, nadir=None, ideal=None):
        self.iter += 1
        return np.array(self.wandb_summary[f'pareto_point_{self.iter}'])
