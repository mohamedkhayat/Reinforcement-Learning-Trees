from typing import Any
import numpy as np
from RLT import BaseRLT


class RLTClassification(BaseRLT):
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
            task_type="classification",
        )

    def _get_loss(self, y: np.ndarray) -> float:
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
        values, counts = np.unique(y, return_counts=True)
        idx = np.argmax(counts)
        label = values[idx]
        return label
