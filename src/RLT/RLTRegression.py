import numpy as np
from RLT import BaseRLT


class RLTRegression(BaseRLT):
    def __init__(
        self,
        max_depth,
        min_samples_split=2,
        n_estimators=50,
        muting_rate=0.5,
        protected_count=2,
        random_state=42,
    ):
        super().__init__(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            n_estimators=n_estimators,
            muting_rate=muting_rate,
            protected_count=protected_count,
            random_state=random_state,
            task_type="regression",
        )

    def _get_loss(self, y: np.ndarray) -> float:
        if len(y) == 0:
            return 0
        y_mean = np.mean(y)
        mse = np.mean((y - y_mean) ** 2)
        return mse

    def _get_node_value(self, y: np.ndarray) -> float:
        return np.mean(y)
