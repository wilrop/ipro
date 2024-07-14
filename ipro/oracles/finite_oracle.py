from typing import Any, Optional

import numpy as np

from ipro.oracles.asf_oracle import ASFOracle
from ipro.oracles.vector_u import create_batched_aasf


class FiniteOracle(ASFOracle):
    """A simple oracle for the known problem setting."""

    def __init__(
            self,
            problem: np.ndarray,
            direction: str = 'maximize',
            **kwargs: Any
    ):
        super().__init__(problem, **kwargs)

        self.direction = direction
        self.sign = 1 if direction == 'maximize' else -1

    def config(self) -> dict:
        """Return the configuration of the oracle."""
        conf = super().config()
        return {
            **conf,
            'direction': self.direction
        }

    def solve(
            self,
            referent: np.ndarray,
            nadir: Optional[np.ndarray] = None,
            ideal: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """The inner loop solver for the basic setting."""
        params = self.get_asf_params(referent, nadir, ideal)
        referent, nadir, ideal = [self.sign * param for param in params]
        u_f = create_batched_aasf(referent, nadir, ideal, aug=self.aug, scale=self.scale)
        utilities = u_f(self.sign * self.problem)
        best_point = np.argmax(utilities)
        return self.problem[best_point]
