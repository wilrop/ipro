from typing import Any, Optional

import numpy as np


class Oracle:
    def __init__(
            self,
            problem: Any,
            **kwargs: Any):
        self.problem = problem
        self.nadir = None
        self.ideal = None

    def config(self) -> dict:
        return {}

    def init_oracle(self, nadir: np.ndarray, ideal: np.ndarray):
        self.nadir = nadir
        self.ideal = ideal

    def solve(
            self,
            referent: np.ndarray,
            nadir: Optional[np.ndarray] = None,
            ideal: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        raise NotImplementedError
