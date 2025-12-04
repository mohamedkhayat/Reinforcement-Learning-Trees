import numpy as np
from RLT import RLT


class RLTRegression(RLT):
    def __init__(
        self,
        task_type: str,
        n_estimators: int,
        muting_rate: float,
        min_protected: int,
        n_thresholds_to_try: int,
        max_depth: int,
        min_samples_split: int = 2,
        random_state: int = 42,
    ):
        super().__init__(
            task_type,
            n_estimators,
            muting_rate,
            min_protected,
            n_thresholds_to_try,
            max_depth,
            min_samples_split,
            random_state,
        )

    def _get_loss(self, y: np.ndarray) -> float:
        if len(y) == 0:
            return 0
        y_mean = np.mean(y)
        mse = np.mean((y - y_mean) ** 2)
        return mse

    def _get_node_value(self, y: np.ndarray) -> float:
        return np.mean(y)