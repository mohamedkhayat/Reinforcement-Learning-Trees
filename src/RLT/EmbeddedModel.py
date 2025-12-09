from typing import Dict, List, Tuple
from sklearn.tree import ExtraTreeClassifier, ExtraTreeRegressor
import numpy as np
from joblib import Parallel, delayed


class EmbeddedModel:
    """
    Embedded model for calculating variable importance using ExtraTrees.

    Parameters
    ----------
    task_type : str
        Type of task: "classification" or "regression".
    n_estimators : int, default=30
        Number of trees in the ExtraTrees ensemble.
    max_depth : int, optional
        Maximum depth of the trees.
    min_samples_split : int, default=2
        Minimum number of samples required to split an internal node.
    n_jobs : int, default=1
        Number of jobs to run in parallel.
    seed : int, default=42
        Random seed for reproducibility.
    """

    def __init__(
        self,
        task_type: str,
        n_estimators: int = 30,
        max_depth: int = None,
        min_samples_split: int = 2,
        n_jobs: int = 1,
        seed: int = 42,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.task_type = task_type
        self.n_jobs = n_jobs
        self.seed = seed

    def get_variables_importances(
        self, X: np.ndarray, y: np.ndarray, valid_features: List[int]
    ) -> Dict[int, float]:
        """
        Calculate variable importance scores (VI) for valid features.

        Parameters
        ----------
        X : np.ndarray
            Input features.
        y : np.ndarray
            Target values.
        valid_features : list of int
            List of feature indices to consider.

        Returns
        -------
        dict
            Dictionary mapping feature index to its importance score.
        """
        n_samples = X.shape[0]

        if n_samples < self.min_samples_split:
            return {feat: 0.0 for feat in valid_features}

        rng = np.random.default_rng(self.seed)
        seeds = rng.integers(0, 10**9, size=self.n_estimators)

        results = Parallel(n_jobs=self.n_jobs, prefer="threads")(
            delayed(self._fit_single_tree)(
                X,
                y,
                valid_features,
                seeds[i],
            )
            for i in range(self.n_estimators)
        )

        somme_MSE = 0.0
        somme_PMSE = np.zeros(len(valid_features))

        for res in results:
            if res is None:
                continue
            mse, pmse = res
            somme_MSE += mse
            somme_PMSE += pmse

        if somme_MSE == 0:
            somme_MSE = 1e-10

        vi_scores = {}
        for idx_local, idx_global in enumerate(valid_features):
            ratio = somme_PMSE[idx_local] / somme_MSE
            vi = max(ratio - 1, 0)
            vi_scores[idx_global] = vi

        return vi_scores

    def _fit_single_tree(
        self, X: np.ndarray, y: np.ndarray, valid_features: List, seed: int
    ) -> Tuple[float, np.ndarray]:
        """
        Fit a single ExtraTree to estimate variable importance.

        Parameters
        ----------
        i : int
            Index of the tree (unused, but kept for parallel execution context).
        X : np.ndarray
            Input features.
        y : np.ndarray
            Target values.
        valid_features : list
            List of valid feature indices.
        seed : int
            Random seed.

        Returns
        -------
        Tuple[float, np.ndarray] or None
            A tuple containing the MSE contribution and PMSE contributions, or None if split failed.
        """
        rng = np.random.default_rng(seed)
        n_samples = X.shape[0]
        indices = np.arange(n_samples)

        train_size = int(0.85 * n_samples)
        train_idx = rng.choice(indices, train_size, replace=False)

        mask = np.ones((n_samples), dtype=bool)
        mask[train_idx] = False
        test_idx = indices[mask]

        if len(test_idx) == 0:
            return None

        X_subset = X[:, valid_features]
        X_train, y_train = X_subset[train_idx, :], y[train_idx]
        X_oob, y_oob = X_subset[test_idx, :], y[test_idx]

        params = {
            "min_samples_split": 2,
            "max_depth": None,
            "random_state": seed,
            "max_features": None,
        }
        if self.task_type.lower() == "classification":
            model = ExtraTreeClassifier(**params)

        elif self.task_type.lower() == "regression":
            model = ExtraTreeRegressor(**params)
        else:
            raise ValueError(f"Unknown task_type: {self.task_type}")

        model.fit(X_train, y_train)
        y_pred = model.predict(X_oob)

        if self.task_type.lower() == "classification":
            mse_contribution = np.mean(y_pred != y_oob)

        elif self.task_type.lower() == "regression":
            mse_contribution = np.mean((y_oob - y_pred) ** 2)

        pmse_contributions = np.zeros(len(valid_features))
        for idx_variable in range(len(valid_features)):
            col = X_oob[:, idx_variable].copy()
            rng.shuffle(X_oob[:, idx_variable])
            y_pred = model.predict(X_oob)
            X_oob[:, idx_variable] = col

            if self.task_type.lower() == "classification":
                pmse_contributions[idx_variable] = np.mean(y_pred != y_oob)

            elif self.task_type.lower() == "regression":
                pmse_contributions[idx_variable] = np.mean((y_oob - y_pred) ** 2)

        return mse_contribution, pmse_contributions