from typing import Optional, Any

import numpy as np

from ipro.oracles.oracle import Oracle


class DebugOracle(Oracle):
    """An oracle meant for debugging.

     Takes as input a dictionary of the vector and solution found at each iteration and replays the sequence.
     """

    def __init__(self, problem: dict, **kwargs: Any):
        super().__init__(problem, **kwargs)
        self.iter = 0

    def solve(
            self,
            referent: np.ndarray,
            nadir: Optional[np.ndarray] = None,
            ideal: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, Any]:
        self.iter += 1
        vec, sol = self.problem.get(f'pareto_point_{self.iter}', referent)
        return vec, sol
