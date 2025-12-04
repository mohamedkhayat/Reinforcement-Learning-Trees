import numpy as np
from Node import Node
import pandas as pd
from typing import Any, Tuple, Union, List, Set
from abc import ABC, abstractmethod
from EmbeddedModel import EmbeddedModel


class RLT(ABC):
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
    ) -> None:
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None
        self.random_state = random_state
        self.task_type = task_type
        self.n_estimators = n_estimators
        self.muting_rate = muting_rate
        self.min_protected = min_protected
        self.n_thresholds_to_try = n_thresholds_to_try
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

    def _find_best_split(
        self, X: np.ndarray, y: np.ndarray, valid_features: List
    ) -> Tuple[int, int]:
        embedded_model = EmbeddedModel(
            self.task_type,
            self.n_estimators,
            self.max_depth,
            self.min_samples_split,
        )

        VI_scores = embedded_model.get_variables_importances(X, y, valid_features)

        variables_sorted_by_importance = sorted(
            VI_scores, key=VI_scores.get, reverse=True
        )
        best_feature = variables_sorted_by_importance[0]

        return best_feature, variables_sorted_by_importance, VI_scores

    def _find_best_threshold(self, X, y, coefficients, valid_features, best_feature):
        best_score = float("inf")
        q = 0.2

        idx = valid_features.index(best_feature)

        best_coeff = coefficients[idx]

        Z = X[:, best_feature] * best_coeff

        # Z = np.dot(X[:, valid_features], coefficients)

        min_threshold = np.quantile(Z, q)
        max_threshold = np.quantile(Z, 1 - q)

        if min_threshold >= max_threshold:
            return min_threshold

        thresholds = np.random.uniform(
            low=min_threshold, high=max_threshold, size=self.n_thresholds_to_try
        )

        best_threshold = thresholds[0]

        for threshold in thresholds:
            indice_left = np.where(Z <= threshold)[0]
            indice_right = np.where(Z > threshold)[0]

            if len(indice_left) == 0 or len(indice_right) == 0:
                continue

            score = self._get_score(y, indice_left, indice_right)
            if score < best_score:
                best_score = score
                best_threshold = threshold

        return best_threshold

    def _get_coefficients(self, X, y, valid_features, VI_scores):
        coeffs = []

        for col_idx in valid_features:
            feature_col = X[:, col_idx]

            if np.std(feature_col) == 0 or np.std(y) == 0:
                rho = 0
            else:
                rho = np.corrcoef(feature_col, y)[0, 1]

            direction = np.sign(rho)
            if direction == 0:
                direction = 1

            magnitude = np.sqrt(VI_scores[col_idx])

            beta = direction * magnitude
            coeffs.append(beta)

        return np.array(coeffs)

    def _build_tree(
        self,
        X: np.ndarray,
        y: np.ndarray,
        protected_set: Set,
        muted_set: Set,
        depth: int = 0,
    ) -> Node:
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

        all_features = set(range(X.shape[1]))
        valid_features = list(all_features - muted_set)
        num_valid_features = len(valid_features)

        best_feature, variables_sorted_by_importance, VI_scores = self._find_best_split(
            X, y, valid_features
        )

        if best_feature is None:
            valeur = self._get_node_value(y)
            return Node(valeur=valeur)

        if depth == 0:
            protected_set.update(variables_sorted_by_importance[: self.min_protected])

        protected_set.add(best_feature)

        num_features_to_mute = int(self.muting_rate * num_valid_features)
        features_to_mute = variables_sorted_by_importance[-num_features_to_mute:]

        for feature in features_to_mute:
            if feature not in protected_set:
                muted_set.add(feature)

        coefficients = self._get_coefficients(X, y, valid_features, VI_scores)

        best_threshold = self._find_best_threshold(
            X, y, coefficients, valid_features, best_feature
        )

        # valid_features = [2, 4, 8, 10, 11, 18]
        # best_feature   = 10
        # coefficients   = [10, 8.4, 7, 6,  2, 19]

        idx_best_feature = valid_features.index(best_feature)
        coef_best_feature = coefficients[idx_best_feature]

        indice_left = np.where(
            X[:, best_feature] * coef_best_feature <= best_threshold
        )[0]
        indice_right = np.where(
            X[:, best_feature] * coef_best_feature > best_threshold
        )[0]

        if len(indice_left) == 0 or len(indice_right) == 0:
            valeur = self._get_node_value(y)
            return Node(valeur=valeur)

        x_left, y_left = X[indice_left, :], y[indice_left]
        x_right, y_right = X[indice_right, :], y[indice_right]

        left_node = self._build_tree(
            x_left, y_left, protected_set.copy(), muted_set.copy(), depth + 1
        )
        right_node = self._build_tree(
            x_right, y_right, protected_set.copy(), muted_set.copy(), depth + 1
        )

        return Node(
            best_feature,
            best_threshold,
            coef_best_feature,
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

        self.root = self._build_tree(X, y, set(), set(), depth=0)
        return self.root

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            X = X.values
        predictions = [self._traverse_tree(x, self.root) for x in X]
        return np.array(predictions)

    def _traverse_tree(self, x: np.ndarray, node: Node) -> Any:
        if node.is_terminal():
            return node.valeur

        if x[node.feature] * node.coefficient <= node.threshold:
            return self._traverse_tree(x, node.left)

        return self._traverse_tree(x, node.right)
