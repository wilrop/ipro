import numpy as np

from ipro.linear_solvers.linear_solver import LinearSolver


class Finite(LinearSolver):
    def __init__(
            self,
            problem: np.ndarray,
            direction: str = 'maximize'
    ):
        super().__init__(problem, direction=direction)
        self.optimizer = np.argmax if direction == 'maximize' else np.argmin

    def solve(self, weights):
        """Find the points that optimise the score function."""
        best_score = self.optimizer(np.dot(self.problem, weights))
        return self.problem[best_score]
