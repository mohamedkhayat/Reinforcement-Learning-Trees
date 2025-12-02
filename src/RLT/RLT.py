import numpy as np
from Node import Node
import pandas as pd
from typing import Any, Tuple, Union
from abc import ABC, abstractmethod
from EmbeddedModel import EmbeddedModel


class BaseRLT(ABC):
    def __init__(
        self,
        max_depth: int,
        min_samples_split: int = 2,
        n_estimators=50,
        muting_rate=0.5,
        protected_count=2,
        random_state: int = 42,
        *,
        task_type,
    ) -> None:
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None
        self._set_seed(random_state)
        self.task_type = task_type
        self.n_estimators = n_estimators
        self.muting_rate = muting_rate
        self.protected_count = protected_count

    def _set_seed(self, seed: int) -> None:
        np.random.seed(seed)

    def _get_score(
        self,
        y: np.ndarray,
        indice_left: np.ndarray,
        indice_right: np.ndarray,
    ) -> float:
        y_left, y_right = y[indice_left], y[indice_right]

        score_gauche = self._get_loss(y_left)
        score_droite = self._get_loss(y_right)

        nombre_observations_total = len(y)

        proportion_a_gauche = len(y_left) / nombre_observations_total
        proportion_a_droite = len(y_right) / nombre_observations_total

        score_total = (
            proportion_a_gauche * score_gauche + proportion_a_droite * score_droite
        )
        return score_total

    def _find_best_threshold(self, X, y):
        candidates = np.random.uniform(np.min(X), np.max(X), size=5)
        best_thresh = None
        best_score = float("inf")

        for t in candidates:
            indice_left = np.where(X <= t)
            indice_right = np.where(X > t)
            score = self._get_score(y, indice_left, indice_right)
            if score < best_score:
                best_score = score
                best_thresh = t

        return best_thresh

    def _build_tree(
        self, X: np.ndarray, y: np.ndarray, muted_set, protected_set, depth: int = 0
    ) -> Node:
        # nchoufou est ce que noeud terminal wale bich nwakfou
        # example : max_depth = 3, min_samples_split = 1
        # len(y) = 4
        # wselna depth = 3
        # donc iwali noeud terminal
        # [0,1,2], [[0,1,2]]
        n_samples = X.shape[0]
        n_classes = len(np.unique(y))

        if (
            depth >= self.max_depth
            or n_samples <= self.min_samples_split
            or n_classes == 1
        ):
            valeur = self._get_node_value(y)
            return Node(valeur=valeur)

        all_features = set(range(X.shape[1]))
        valid_features = list(all_features - muted_set)

        embedded_model = EmbeddedModel(
            self.task_type, self.n_estimators, self.min_samples_split
        )
        importances = embedded_model.get_feature_importance(X, y, valid_features)
        sorted_feature_importance = sorted(
            importances, key=importances.get, reverse=True
        )
        best_feature = sorted_feature_importance[0]

        if best_feature is None:
            valeur = self._get_node_value(y)
            return Node(valeur=valeur)

        top_features = sorted_feature_importance[: self.protected_count]
        new_protected_set = protected_set.union(top_features)

        num_to_mute = int(len(valid_features) * self.muting_rate)
        candidates_to_mute = sorted_feature_importance[-num_to_mute:]
        new_muted_set = muted_set.copy()

        for feat in candidates_to_mute:
            if feat not in new_protected_set:
                new_muted_set.add(feat)

        best_threshold = self._find_best_threshold(X[:, best_feature], y)

        indice_left = X[:, best_feature] <= best_threshold
        indice_right = X[:, best_feature] > best_threshold
        x_left, y_left = X[indice_left, :], y[indice_left]
        x_right, y_right = X[indice_right, :], y[indice_right]

        left_node = self._build_tree(
            x_left, y_left, new_muted_set, new_protected_set, depth + 1
        )
        right_node = self._build_tree(
            x_right, y_right, new_muted_set, new_protected_set, depth + 1
        )
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

        initial_muted = set()
        initial_protected = set()

        self.root = self._build_tree(X, y, initial_muted, initial_protected, depth=0)
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
