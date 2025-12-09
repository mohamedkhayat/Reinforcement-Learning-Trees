import numpy as np
from RLT.Node import Node
import pandas as pd
from typing import Any, Dict, Tuple, Union, List, Set
from abc import ABC, abstractmethod
from RLT.EmbeddedModel import EmbeddedModel


class ReinforcementLearningTree(ABC):
    """
    Base class for Reinforcement Learning Tree.

    Parameters
    ----------
    task_type : str
        Type of task: "classification" or "regression".
    n_estimators : int
        Number of estimators for the embedded model.
    muting_rate : float
        Rate at which features are muted.
    min_protected : int
        Minimum number of protected features.
    k : int
        Number of top features to consider.
    alpha : float
        Threshold multiplier for feature selection.
    n_thresholds_to_try : int
        Number of random thresholds to try for splitting.
    max_depth : int
        Maximum depth of the tree.
    min_samples_split : int, default=2
        Minimum number of samples required to split an internal node.
    random_state : int, default=42
        Random seed.
    n_jobs : int, default=1
        Number of parallel jobs.
    """

    def __init__(
        self,
        task_type: str,
        n_estimators: int,
        muting_rate: float,
        min_protected: int,
        k: int,
        alpha: float,
        n_thresholds_to_try: int,
        max_depth: int,
        min_samples_split: int = 2,
        random_state: int = 42,
        n_jobs: int = 1,
    ) -> None:
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None
        self.random_state = random_state
        self.task_type = task_type
        self.k = k
        self.alpha = alpha
        self.n_estimators = n_estimators
        self.muting_rate = muting_rate
        self.min_protected = min_protected
        self.n_thresholds_to_try = n_thresholds_to_try
        self.n_jobs = n_jobs
        self.rng = np.random.default_rng(random_state)

    def _get_score(
        self,
        y: np.ndarray,
        indice_left: np.ndarray,
        indice_right: np.ndarray,
    ) -> float:
        """
        Compute the weighted loss score for a split.

        Parameters
        ----------
        y : np.ndarray
            Target values.
        indice_left : np.ndarray
            Indices of samples in the left child.
        indice_right : np.ndarray
            Indices of samples in the right child.

        Returns
        -------
        float
            The weighted sum of losses for the split.
        """
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
        self, X: np.ndarray, y: np.ndarray, valid_features: List[int]
    ) -> Tuple[int, int]:
        """
        Find the best features to split on using the embedded model.

        Parameters
        ----------
        X : np.ndarray
            Input features.
        y : np.ndarray
            Target values.
        valid_features : List
            List of valid feature indices to consider.

        Returns
        -------
        Tuple[List, dict]
            A tuple containing the list of variables sorted by importance and the dictionary of VI scores.
        """
        embedded_model = EmbeddedModel(
            self.task_type,
            self.n_estimators,
            self.max_depth,
            2,#self.min_samples_split,
            self.n_jobs,
            self.random_state,
        )

        VI_scores = embedded_model.get_variables_importances(X, y, valid_features)

        variables_sorted_by_importance = sorted(
            VI_scores, key=VI_scores.get, reverse=True
        )

        return variables_sorted_by_importance, VI_scores

    def _find_best_threshold(
        self, X: np.ndarray, y: np.ndarray, best_features: List[int], coeffs: np.ndarray
    ) -> float:
        """
        Find the best threshold for the linear combination of features.

        Parameters
        ----------
        X : np.ndarray
            Input features.
        y : np.ndarray
            Target values.
        best_features : list
            Indices of the selected features.
        coeffs : np.ndarray
            Coefficients for the linear combination.

        Returns
        -------
        float
            The best threshold value found.
        """
        best_score = float("inf")
        q = 0.2

        Z = np.dot(X[:, best_features], coeffs)

        min_threshold = np.quantile(Z, q)
        max_threshold = np.quantile(Z, 1 - q)

        if min_threshold >= max_threshold:
            return min_threshold

        thresholds = self.rng.uniform(
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

    def _get_coefficients(
        self,
        X: np.ndarray,
        y: np.ndarray,
        valid_features: List[int],
        VI_scores: Dict[int, float],
        variables_sorted_by_importance: List[int],
    ) -> Tuple[np.ndarray, List]:
        """
        Calculate coefficients for the linear combination split.

        Parameters
        ----------
        X : np.ndarray
            Input features.
        y : np.ndarray
            Target values.
        valid_features : list
            List of valid feature indices.
        VI_scores : dict
            Variable importance scores.
        variables_sorted_by_importance : list
            List of feature indices sorted by importance.

        Returns
        -------
        Tuple[np.ndarray, list]
            A tuple containing the coefficients array and the list of best feature indices.
        """
        coeffs = []
        best_features = []
        if (
            len(variables_sorted_by_importance) == 0
            or variables_sorted_by_importance is None
        ):
            return np.zeros(
                len(valid_features),
            )

        index_top_kth_vi = min(self.k, len(valid_features)) - 1
        top_kth_vi = VI_scores[variables_sorted_by_importance[index_top_kth_vi]]

        max_vi = VI_scores[variables_sorted_by_importance[0]]

        for col_idx in valid_features:
            col_score = VI_scores[col_idx]
            if (
                col_score <= 0
                or col_score < self.alpha * max_vi
                or col_score < top_kth_vi
            ):
                continue

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
            best_features.append(col_idx)

        return np.array(coeffs), best_features

    def _build_tree(
        self,
        X: np.ndarray,
        y: np.ndarray,
        protected_set: Set,
        muted_set: Set,
        depth: int = 0,
    ) -> Node:
        """
        Recursively build the Reinforcement Learning Tree.

        Parameters
        ----------
        X : np.ndarray
            Input features.
        y : np.ndarray
            Target values.
        protected_set : Set
            Set of protected features (cannot be muted).
        muted_set : Set
            Set of muted features (cannot be used for splitting).
        depth : int, default=0
            Current depth of the tree.

        Returns
        -------
        Node
            The root node of the subtree.
        """
        # nchoufou est ce que noeud terminal wale bich nwakfou
        # example : max_depth = 3, min_samples_split = 1
        # len(y) = 4
        # wselna depth = 3
        # donc iwali noeud terminal
        # [0,1,2], [[0,1,2]]
        if (
            (self.max_depth and depth >= self.max_depth)
            or len(X) <= self.min_samples_split
            or len(np.unique(y)) == 1
        ):
            valeur = self._get_node_value(y)
            probabilities = (
                self._get_node_probabilities(y)
                if hasattr(self, "_get_node_probabilities")
                else None
            )
            return Node(valeur=valeur, probabilities=probabilities)

        all_features = set(range(X.shape[1]))
        valid_features = list(all_features - muted_set)
        num_valid_features = len(valid_features)

        variables_sorted_by_importance, VI_scores = self._find_best_split(
            X, y, valid_features
        )

        coefficients, best_features = self._get_coefficients(
            X, y, valid_features, VI_scores, variables_sorted_by_importance
        )

        if best_features is None or len(best_features) == 0:
            valeur = self._get_node_value(y)
            probabilities = (
                self._get_node_probabilities(y)
                if hasattr(self, "_get_node_probabilities")
                else None
            )
            return Node(valeur=valeur, probabilities=probabilities)

        best_threshold = self._find_best_threshold(X, y, best_features, coefficients)

        if depth == 0:
            protected_set.update(variables_sorted_by_importance[: self.min_protected])

        protected_set.update(best_features)

        num_features_to_mute = int(self.muting_rate * num_valid_features)

        muting_candidates = [f for f in valid_features if f not in protected_set]
        muting_candidates_sorted = sorted(muting_candidates, key=lambda f: VI_scores.get(f, 0))
        features_to_mute = muting_candidates_sorted[:num_features_to_mute] if num_features_to_mute > 0 else []

        muted_set.update(features_to_mute)

        # valid_features = [2, 4, 8, 10, 11, 18]
        # best_feature   = 10
        # coefficients   = [10, 8.4, 7, 6,  2, 19]

        indice_left = np.where(
            np.dot(X[:, best_features], coefficients) <= best_threshold
        )[0]
        indice_right = np.where(
            np.dot(X[:, best_features], coefficients) > best_threshold
        )[0]

        if len(indice_left) == 0 or len(indice_right) == 0:
            valeur = self._get_node_value(y)
            probabilities = (
                self._get_node_probabilities(y)
                if hasattr(self, "_get_node_probabilities")
                else None
            )
            return Node(valeur=valeur, probabilities=probabilities)

        x_left, y_left = X[indice_left, :], y[indice_left]
        x_right, y_right = X[indice_right, :], y[indice_right]

        left_node = self._build_tree(
            x_left, y_left, protected_set.copy(), muted_set.copy(), depth + 1
        )
        right_node = self._build_tree(
            x_right, y_right, protected_set.copy(), muted_set.copy(), depth + 1
        )

        return Node(
            best_features,
            best_threshold,
            coefficients,
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
        """
        Build the tree from the training set (X, y).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.
        y : array-like of shape (n_samples,)
            The target values.

        Returns
        -------
        Node
            The root node of the built tree.
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values

        self.root = self._build_tree(X, y, set(), set(), depth=0)
        return self.root

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Predict class or regression value for X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        np.ndarray
            The predicted values.
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        predictions = [self._traverse_tree(x, self.root) for x in X]
        return np.array(predictions)

    def _traverse_tree(self, x: np.ndarray, node: Node) -> Any:
        """
        Traverse the tree to make a prediction for a single sample.

        Parameters
        ----------
        x : np.ndarray
            Input feature vector.
        node : Node
            Current node in the traversal.

        Returns
        -------
        Any
            The prediction value found at the leaf node.
        """
        if node.is_terminal():
            return node.valeur

        if np.dot(x[node.features], node.coefficients) <= node.threshold:
            return self._traverse_tree(x, node.left)

        return self._traverse_tree(x, node.right)

    def _traverse_tree_proba(self, x: np.ndarray, node: Node) -> Dict[Any, float]:
        """
        Traverse the tree to get class probabilities for a single sample.

        Parameters
        ----------
        x : np.ndarray
            Input feature vector.
        node : Node
            Current node in the traversal.

        Returns
        -------
        Dict[Any, float]
            Dictionary mapping class labels to their probabilities.
        """
        if node.is_terminal():
            return node.probabilities if node.probabilities is not None else {}

        if np.dot(x[node.features], node.coefficients) <= node.threshold:
            return self._traverse_tree_proba(x, node.left)

        return self._traverse_tree_proba(x, node.right)

    def predict_proba(
        self, X: Union[pd.DataFrame, np.ndarray], classes: np.ndarray
    ) -> np.ndarray:
        """
        Predict class probabilities for X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.
        classes : np.ndarray
            Array of all possible class labels.

        Returns
        -------
        np.ndarray of shape (n_samples, n_classes)
            The class probabilities for each sample.
        """
        if isinstance(X, pd.DataFrame):
            X = X.values

        n_samples = X.shape[0]
        n_classes = len(classes)
        class_to_idx = {c: i for i, c in enumerate(classes)}

        proba = np.zeros((n_samples, n_classes))

        for i, x in enumerate(X):
            node_probs = self._traverse_tree_proba(x, self.root)
            for cls, prob in node_probs.items():
                if cls in class_to_idx:
                    proba[i, class_to_idx[cls]] = prob

        return proba