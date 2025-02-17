from typing import Any

import numpy as np


class LinearSolver:
    def __init__(
            self,
            problem: Any,
            *args: Any,
            direction: str = 'maximize',
            **kwargs: Any
    ):
        self.problem = problem
        self.direction = direction

    def config(self):
        return {
            'direction': self.direction
        }

    def solve(self, weight: np.ndarray) -> tuple[np.ndarray, Any]:
        raise NotImplementedError
