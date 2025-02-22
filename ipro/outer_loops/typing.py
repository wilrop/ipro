import numpy as np

from typing import Callable, TypeAlias, Any
from dataclasses import dataclass

from ipro.outer_loops.box import Box


@dataclass
class Subproblem:
    """A subproblem"""
    referent: np.ndarray
    nadir: np.ndarray
    ideal: np.ndarray


Subsolution: TypeAlias = tuple[Subproblem, np.ndarray, Any]
IPROCallback: TypeAlias = Callable[[int, float, float, float, float, float], Any]
