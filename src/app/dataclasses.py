import os
import numpy as np
from dataclasses import dataclass, astuple
from typing import List

@dataclass
class Label:
    pixel: [int, int]
    depth: np.ndarray
    camera: np.ndarray

    def __iter__(self):
        return iter(astuple(self))

@dataclass
class Scene:
    index: int
    name: str
    path: os.PathLike
    labels: List[Label]
