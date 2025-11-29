from typing import Any
import numpy as np
from RLT import RLT


class RLTClassification(RLT):
    def __init__(
        self, max_depth: int, min_samples_split: int = 2, random_state: int = 42
    ):
        super().__init__(max_depth, min_samples_split, random_state)

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
