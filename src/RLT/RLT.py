import numpy as np
from Node import Node
import pandas as pd
from typing import Any, Tuple, Union
from abc import ABC, abstractmethod


class RLT(ABC):
    def __init__(
        self, max_depth: int, min_samples_split: int = 2, random_state: int = 42
    ) -> None:
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None
        self._set_seed(random_state)

    def _set_seed(self, seed: int) -> None:
        np.random.seed(seed)

    def _get_score(
        self,
        y: np.ndarray,
        indice_left: np.ndarray,
        indice_right: np.ndarray,
    ) -> float:
        y_left = y[indice_left]
        y_right = y[indice_right]

        score_gauche = self._get_loss(y_left)
        score_droite = self._get_loss(y_right)

        nombre_observations_total = len(y)

        proportion_a_gauche = len(y_left) / nombre_observations_total
        proportion_a_droite = len(y_right) / nombre_observations_total

        score_total = (
            proportion_a_gauche * score_gauche + proportion_a_droite * score_droite
        )
        return score_total

    def _find_best_split(self, X: np.ndarray, y: np.ndarray) -> Tuple[int, int]:
        best_feature = best_threshold = None
        best_score = float("inf")
        # nlawjou ahsen variable
        for variable in range(X.shape[1]):
            # nlawjou ahsen seuil
            for threshold in sorted(np.unique(X[:, variable])):
                # nkasmou donnes mte3na sur 2, eli a gauche w a droite
                indice_left = np.where(X[:, variable] <= threshold)
                indice_right = np.where(X[:, variable] > threshold)

                if len(indice_left) == 0 or len(indice_right) == 0:
                    continue
                """
                X : colonnes : toul w age : {
                                            [toul : 178, 180, 150],
                                            [age : 20, 22, 19]
                                            }
                supposans ahna wselna feature : age, threshold = 20
                indice_a_gauche = [2]
                indice_a_droite = [0, 1]
                
                X_left = X[indice_a_gauche, : ]
                X_right X[indice_a_droite,  : ]
                
                """

                score = self._get_score(y, indice_left, indice_right)
                if score < best_score:
                    best_score = score
                    best_feature = variable
                    best_threshold = threshold

        return best_feature, best_threshold

    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int = 0) -> Node:
        # nchoufou est ce que noeud terminal wale bich nwakfou
        # example : max_depth = 3, min_samples_split = 1
        # len(y) = 4
        # wselna depth = 3
        # donc iwali noeud terminal
        # [0,1,2], [[0,1,2]]
        if (
            depth >= self.max_depth
            or len(X) <= self.min_samples_split
            or len(np.unique(y)) == 1
        ):
            valeur = self._get_node_value(y)
            return Node(valeur=valeur)

        best_feature, best_threshold = self._find_best_split(X, y)

        if best_feature is None:
            valeur = self._get_node_value(y)
            return Node(valeur=valeur)

        indice_left = np.where(X[:, best_feature] <= best_threshold)[0]
        indice_right = np.where(X[:, best_feature] > best_threshold)[0]

        x_left, y_left = X[indice_left, : ], y[indice_left]
        x_right, y_right = X[indice_right, : ], y[indice_right]

        left_node = self._build_tree(x_left, y_left, depth + 1)
        right_node = self._build_tree(x_right, y_right, depth + 1)
        return Node(
            best_feature,
            best_threshold,
            left_node,
            right_node,
        )

    @abstractmethod
    def _get_loss(self, y: np.ndarray) -> float:
        pass

    @abstractmethod
    def _get_node_value(self, y: np.ndarray) -> Any:
        pass

    def fit(
        self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]
    ) -> Node:
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values

        self.root = self._build_tree(X, y, depth=0)
        return self.root

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            X = X.values
        predictions = [self._traverse_tree(x, self.root) for x in X]
        return np.array(predictions)

    def _traverse_tree(self, x: np.ndarray, node: Node) -> Any:
        if node.is_terminal():
            return node.valeur

        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)