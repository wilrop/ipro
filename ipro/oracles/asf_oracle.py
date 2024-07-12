from typing import Any, Optional

import torch

import numpy as np

from ipro.oracles.oracle import Oracle


class ASFOracle(Oracle):
    def __init__(
            self,
            problem: Any,
            aug: float = 0.1,
            scale: float = 100,
            vary_nadir: bool = False,
            vary_ideal: bool = False,
            **kwargs: Any
    ):
        super().__init__(problem, **kwargs)

        self.aug = aug
        self.scale = scale
        self.vary_nadir = vary_nadir
        self.vary_ideal = vary_ideal

        self.nadir = None
        self.ideal = None

    def config(self) -> dict:
        conf = super().config()
        return {
            **conf,
            'aug': self.aug,
            'scale': self.scale,
            'vary_nadir': self.vary_nadir,
            'vary_ideal': self.vary_ideal,
        }

    def get_asf_params(
            self,
            referent: np.ndarray,
            nadir: Optional[np.ndarray] = None,
            ideal: Optional[np.ndarray] = None,
            backend: str = 'numpy'
    ) -> tuple[np.ndarray | torch.Tensor, np.ndarray | torch.Tensor, np.ndarray | torch.Tensor]:
        """Get the parameters for the utility function."""
        nadir = nadir if nadir is not None and self.vary_nadir else self.nadir
        ideal = ideal if ideal is not None and self.vary_ideal else self.ideal

        if backend == 'torch':
            referent = torch.tensor(referent, dtype=torch.float32)
            nadir = torch.tensor(nadir, dtype=torch.float32)
            ideal = torch.tensor(ideal, dtype=torch.float32)
        return referent, nadir, ideal

    def solve(
            self,
            referent: np.ndarray,
            nadir: Optional[np.ndarray] = None,
            ideal: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        raise NotImplementedError
