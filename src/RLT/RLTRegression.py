import numpy as np
from RLT.ReinforcementLearningTree import ReinforcementLearningTree


class RLTRegression(ReinforcementLearningTree):
    """
    Regression variant of Reinforcement Learning Tree.

    Uses Mean Squared Error (MSE) as the loss function and mean value for prediction.
    """

    def __init__(
        self,
        task_type: str,
        n_estimators: int,
        muting_rate: float,
        min_protected: int,
        k: int,
        alpha: int,
        n_thresholds_to_try: int,
        max_depth: int,
        min_samples_split: int = 2,
        random_state: int = 42,
        n_jobs: int = 1,
    ):
        super().__init__(
            task_type,
            n_estimators,
            muting_rate,
            min_protected,
            k,
            alpha,
            n_thresholds_to_try,
            max_depth,
            min_samples_split,
            random_state,
            n_jobs,
        )

    def _get_loss(self, y: np.ndarray) -> float:
        """
        Compute Mean Squared Error (MSE) for the target values.

        Parameters
        ----------
        y : np.ndarray
            Target values.

        Returns
        -------
        float
            The MSE value.
        """
        if len(y) == 0:
            return 0
        y_mean = np.mean(y)
        mse = np.mean((y - y_mean) ** 2)
        return mse

    def _get_node_value(self, y: np.ndarray) -> float:
        """
        Compute the mean value of the targets.

        Parameters
        ----------
        y : np.ndarray
            Target values.

        Returns
        -------
        float
            The mean of y.
        """
        return np.mean(y)
