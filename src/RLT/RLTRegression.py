from typing import Any
import numpy as np
from RLT import RLT


class RLTRegression(RLT):
    def __init__(
        self, max_depth: int, min_samples_split: int = 2, random_state: int = 42
    ):
        super().__init__(max_depth, min_samples_split, random_state)

    def _get_loss(self, y: np.ndarray) -> float:
        if len(y) == 0:
            return 0
        y_mean = np.mean(y)
        mse = np.mean((y - y_mean) ** 2)
        return mse

    def _get_node_value(self, y: np.ndarray) -> float:
        return np.mean(y)
