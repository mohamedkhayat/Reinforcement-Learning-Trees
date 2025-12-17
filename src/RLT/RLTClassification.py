from typing import Any, Dict
import numpy as np
from RLT.ReinforcementLearningTree import ReinforcementLearningTree


class RLTClassification(ReinforcementLearningTree):
    """
    Classification variant of Reinforcement Learning Tree.

    Uses Gini impurity as the loss function and soft voting (probability averaging)
    for prediction when used in an ensemble.
    """

    def __init__(
        self,
        task_type: str,
        embedded_model: str,
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
        use_bandit: bool = False,
        bandit_exploration: float = 1.0,
        bandit_selection_rate: float = 0.5,
    ):
        super().__init__(
            task_type,
            embedded_model,
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
            use_bandit,
            bandit_exploration,
            bandit_selection_rate,
        )

    def _get_loss(self, y: np.ndarray) -> float:
        """
        Compute Gini impurity for the target values.

        Parameters
        ----------
        y : np.ndarray
            Target values.

        Returns
        -------
        float
            The Gini impurity.
        """
        if len(y) <= 1:
            return 0

        # calcul nombre d'occurence chaque classe:
        # par example : y = [0, 1, 1, 1]
        classes, counts = np.unique(y, return_counts=True)
        # classes = [0, 1]
        # counts = [1, 3]
        probabilities = counts / len(y)
        """
        len(y) = 4
        probabilities = pour chaque classe,
        on divise nombre d'occurence sur nombre total d'observation
        [1 /4, 3/4]
        gini = 1 - somme(probalities ** 2)
        somme = 0
        for proba in probabilities:
            proba_carré = probabilities ** 2
            somme += proba_carré
        gini = 1 - gini
        """
        gini = 1 - np.sum(probabilities**2)
        return gini

    def _get_node_value(self, y: np.ndarray) -> Any:
        """
        Find the most frequent class (majority vote).

        Parameters
        ----------
        y : np.ndarray
            Target values.

        Returns
        -------
        Any
            The most frequent class label.
        """
        values, counts = np.unique(y, return_counts=True)
        idx = np.argmax(counts)
        label = values[idx]
        return label

    def _get_node_probabilities(self, y: np.ndarray) -> Dict[Any, float]:
        """
        Compute class probabilities at a node.

        Parameters
        ----------
        y : np.ndarray
            Target values at this node.

        Returns
        -------
        Dict[Any, float]
            Dictionary mapping class labels to their probabilities.
        """
        if len(y) == 0:
            return {}
        classes, counts = np.unique(y, return_counts=True)
        probs = counts / len(y)
        return dict(zip(classes, probs))