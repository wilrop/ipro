from typing import Optional, Any

import numpy as np

from ipro.oracles.oracle import Oracle


class DebugOracle(Oracle):
    """An oracle meant for debugging.

     It takes as input a dictionary of the Pareto point found in each iteration and replays the same sequence.
     """

    def __init__(self, problem: dict, **kwargs: Any):
        super().__init__(problem, **kwargs)
        self.iter = 0

    def solve(
            self,
            referent: np.ndarray,
            nadir: Optional[np.ndarray] = None,
            ideal: Optional[np.ndarray] = None
    ) -> np.ndarray:
        self.iter += 1
        res = self.problem.get(f'pareto_point_{self.iter}', referent)
        return np.array(res)
