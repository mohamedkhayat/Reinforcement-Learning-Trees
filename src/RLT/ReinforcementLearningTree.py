import numpy as np
import pandas as pd
from typing import Any, Dict, Tuple, Union, List, Set
from abc import ABC, abstractmethod
from RLT.EmbeddedModel import EmbeddedModel
from RLT.Node import Node


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
    ) -> Tuple[List[int], Dict[int, float]]:
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
        embedded_seed = int(self.rng.integers(0, 10**9))

        embedded_model = EmbeddedModel(
            self.task_type,
            self.n_estimators,
            self.max_depth,
            self.min_samples_split,
            self.n_jobs,
            embedded_seed,
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
        """
        best_score = float("inf")

        Z = np.dot(X[:, best_features], coeffs)

        min_z = np.min(Z)
        max_z = np.max(Z)
        interval_length = max_z - min_z

        if interval_length <= 1e-9:
            return min_z

        q = max(0.05, self.min_samples_split / (len(y) + 1))
        q = min(q, 0.45)

        margin = interval_length * q
        lower_bound = min_z + margin
        upper_bound = max_z - margin

        if lower_bound >= upper_bound:
            lower_bound = min_z
            upper_bound = max_z

        n_candidates = max(self.n_thresholds_to_try, 1)
        thresholds = self.rng.uniform(
            low=lower_bound, high=upper_bound, size=n_candidates
        )

        best_threshold = thresholds[0]
        found_valid_split = False

        min_leaf = max(1, self.min_samples_split // 2)

        for threshold in thresholds:
            mask_left = Z <= threshold
            indice_left = np.where(mask_left)[0]
            indice_right = np.where(~mask_left)[0]

            if len(indice_left) < min_leaf or len(indice_right) < min_leaf:
                continue

            score = self._get_score(y, indice_left, indice_right)

            if score < best_score:
                best_score = score
                best_threshold = threshold
                found_valid_split = True

        if not found_valid_split:
            best_threshold = np.median(Z)

        return best_threshold

    def _get_coefficients(
        self,
        X: np.ndarray,
        y: np.ndarray,
        VI_scores: Dict[int, float],
        variables_sorted_by_importance: List[int],
    ) -> Tuple[np.ndarray, List]:
        """
        Calculate coefficients for the linear combination split.
        Matches C implementation: Normalizes weights relative to the strongest variable.
        """
        if not variables_sorted_by_importance:
            return np.array([]), []

        max_vi = VI_scores.get(variables_sorted_by_importance[0], 0)

        if max_vi <= 0:
            return np.array([]), []

        candidates = []
        for var_idx in variables_sorted_by_importance:
            score = VI_scores.get(var_idx, 0)
            if score <= 0 or score < self.alpha * max_vi:
                break
            candidates.append(var_idx)

        selected_vars = candidates[: self.k]

        if not selected_vars:
            selected_vars = [variables_sorted_by_importance[0]]

        raw_coeffs = []
        best_features = []

        if self.task_type.lower() == "classification":
            classes = np.unique(y)
            if len(classes) < 2:
                return np.array([]), []

        for col_idx in selected_vars:
            feature_col = X[:, col_idx]
            random_direction = 1 if self.rng.random() > 0.5 else -1
            if self.task_type.lower() == "classification":
                classes = np.unique(y)

                if len(classes) < 2:
                    direction = 1
                else:
                    mask_1 = y == classes[1]
                    mask_0 = y == classes[0]

                    mean_1 = np.mean(feature_col[mask_1]) if np.any(mask_1) else 0
                    mean_0 = np.mean(feature_col[mask_0]) if np.any(mask_0) else 0

                    diff = mean_1 - mean_0

                    if abs(diff) > 1e-9:
                        direction = np.sign(diff)
                    else:
                        direction = random_direction

            elif self.task_type.lower() == "regression":
                if np.std(feature_col) < 1e-9 or np.std(y) < 1e-9:
                    direction = random_direction
                else:
                    # np.corrcoef returns matrix [[1, r], [r, 1]]
                    rho = np.corrcoef(feature_col, y)[0, 1]
                    if abs(rho) > 1e-9:
                        direction = np.sign(rho)
                    else:
                        direction = random_direction

            magnitude = np.sqrt(VI_scores[col_idx])
            beta = direction * magnitude

            raw_coeffs.append(beta)
            best_features.append(col_idx)

        if not raw_coeffs:
            return np.array([]), []

        first_coeff = raw_coeffs[0]

        if abs(first_coeff) < 1e-9:
            first_coeff = 1.0 if first_coeff >= 0 else -1.0

        normalized_coeffs = [c / first_coeff for c in raw_coeffs]
        normalized_coeffs[0] = 1.0

        return np.array(normalized_coeffs), best_features

    def _find_standard_split(self, X, y, valid_features):
        """
        Fallback: Standard Random Forest greedy split logic.
        Used when node is too small for Reinforcement Learning.
        """
        best_score = float("inf")
        best_feature = []
        best_threshold = None
        best_coeff = None

        n_features = len(valid_features)
        mtry = int(np.sqrt(n_features))
        mtry = max(1, min(mtry, n_features))

        candidate_features = self.rng.choice(valid_features, mtry, replace=False)

        for feat_idx in candidate_features:
            coeffs = np.array([1.0])

            threshold = self._find_best_threshold(X, y, [feat_idx], coeffs)

            mask_left = X[:, feat_idx] <= threshold
            indices_left = np.where(mask_left)[0]
            indices_right = np.where(~mask_left)[0]

            if len(indices_left) == 0 or len(indices_right) == 0:
                continue

            score = self._get_score(y, indices_left, indices_right)

            if score < best_score:
                best_score = score
                best_feature = [feat_idx]
                best_threshold = threshold
                best_coeff = coeffs

        return best_feature, best_threshold, best_coeff

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

        n_node = len(y)
        all_features = set(range(X.shape[1]))
        valid_features = list(all_features - muted_set)

        # Check if we should use Reinforcement Learning or Fallback
        if n_node >= self.min_samples_split and len(valid_features) > 1:
            # 1. Run Embedded Model (RLT)
            variables_sorted_by_importance, VI_scores = self._find_best_split(
                X, y, valid_features
            )

            # 2. Try Linear Combination
            coefficients, best_features = self._get_coefficients(
                X, y, VI_scores, variables_sorted_by_importance
            )

            # 3. Find Threshold
            if best_features is not None and len(best_features) > 0:
                best_threshold = self._find_best_threshold(
                    X, y, best_features, coefficients
                )
            else:
                # Fallback if VI <= 0
                best_features, best_threshold, coefficients = self._find_standard_split(
                    X, y, valid_features
                )

        else:
            # Fallback: Node too small for Embedded Model
            best_features, best_threshold, coefficients = self._find_standard_split(
                X, y, valid_features
            )

        if best_features is None or len(best_features) == 0:
            valeur = self._get_node_value(y)
            probabilities = (
                self._get_node_probabilities(y)
                if hasattr(self, "_get_node_probabilities")
                else None
            )
            return Node(valeur=valeur, probabilities=probabilities)

        if depth == 0:
            if "variables_sorted_by_importance" in locals():
                protected_set.update(
                    variables_sorted_by_importance[: self.min_protected]
                )

        protected_set.update(best_features)

        if "VI_scores" in locals():
            num_valid = len(valid_features)
            num_to_mute = int(self.muting_rate * num_valid)

            muting_candidates = [f for f in valid_features if f not in protected_set]

            muting_candidates.sort(key=lambda f: VI_scores.get(f, 0.0))

            newly_muted = set(muting_candidates[:num_to_mute])

            next_muted_set = muted_set.union(newly_muted)
        else:
            next_muted_set = muted_set

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
            x_left, y_left, protected_set.copy(), next_muted_set.copy(), depth + 1
        )
        right_node = self._build_tree(
            x_right, y_right, protected_set.copy(), next_muted_set.copy(), depth + 1
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

        all_features = set(range(X.shape[1]))
        valid_features = list(all_features)

        _, VI_scores = self._find_best_split(X, y, valid_features)

        self.variable_importances_ = np.zeros(X.shape[1])

        for idx in range(X.shape[1]):
            self.variable_importances_[idx] = VI_scores.get(idx, 0.0)

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
