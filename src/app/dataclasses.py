import os
import numpy as np
from dataclasses import dataclass
from typing import List

@dataclass
class Label:
    pixel: [int, int]
    depth: np.ndarray
    camera: np.ndarray

@dataclass
class Scene:
    index: int
    name: str
    path: os.PathLike
    labels: List[Label]
