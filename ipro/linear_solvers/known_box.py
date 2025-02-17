import numpy as np

from numpy.typing import ArrayLike
from typing import Any, Optional



class KnownBox:
    """A linear solver for settings with a known bounding box."""

    def __init__(
            self,
            problem: np.ndarray,
            nadir_vecs: ArrayLike,
            ideal_vecs: ArrayLike,
            ideal_sols: Optional[list[Any]] = None,
            direction: str = 'maximize',
    ):
        super().__init__(problem, direction=direction)
        self.nadir_vecs = np.array(nadir_vecs)
        self.ideal_vecs = np.array(ideal_vecs)
        self.ideal_sols = ideal_sols if ideal_sols else [None for _ in range(len(ideal_vecs))]

    def solve(self, weight: np.ndarray) -> tuple[np.ndarray, Any]:
        """Returns a solution to form known bounding box"""
        if np.sum(weight) > 0:
            obj_idx = np.argmax(weight)
            vec = self.ideal_vecs[obj_idx]
            sol = self.ideal_sols[obj_idx]
        else:
            obj_idx = np.argmin(weight)
            vec = self.nadir_vecs[obj_idx]
            sol = None

        return vec, sol