from typing import Dict, Self, Union
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from RLT.RLTClassification import RLTClassification
from RLT.RLTRegression import RLTRegression
from scipy.stats import mode

from RLT.ReinforcementLearningTree import ReinforcementLearningTree


class ReinforcementLearningTrees:
    """
    Ensemble of Reinforcement Learning Trees (Forest).

    Parameters
    ----------
    task_type : str
        Type of task: "classification" or "regression".
    n_rlt_trees : int, default=10
        Number of RLT trees in the forest.
    n_extra_trees : int, default=50
        Number of ExtraTrees in the embedded model.
    muting_rate : float, default=0.5
        Rate at which features are muted.
    min_protected : int, default=3
        Minimum number of protected features.
    k : int, default=1
        Number of top features to consider.
    alpha : float, default=0.1
        Threshold multiplier for feature selection.
    n_thresholds_to_try : int, default=10
        Number of random thresholds to try.
    max_depth : int, default=10
        Maximum depth of the trees.
    min_samples_split : int, default=5
        Minimum number of samples required to split.
    n_jobs : int, default=1
        Number of parallel jobs.
    random_state : int, default=42
        Random seed.
    """

    def __init__(
        self,
        task_type: str,
        n_rlt_trees: int = 10,
        n_extra_trees: int = 50,
        muting_rate: float = 0.5,
        min_protected: int = 3,
        k: int = 1,
        alpha: int = 0.1,
        n_thresholds_to_try: int = 10,
        max_depth: int = 10,
        min_samples_split: int = 5,
        n_jobs: int = 1,
        random_state: int = 42,
    ):
        self.task_type = task_type
        self.n_rlt_trees = n_rlt_trees
        self.n_extra_trees = n_extra_trees
        self.muting_rate = muting_rate
        self.min_protected = min_protected
        self.k = k
        self.alpha = alpha
        self.n_thresholds_to_try = n_thresholds_to_try
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.trees = []

    def _fit_single_tree(
        self,
        X_boot: np.ndarray,
        y_boot: np.ndarray,
        params: Dict[str, Union[str, int]],
        seed: int,
        embedded_n_jobs: int,
    ) -> ReinforcementLearningTree:
        """
        Fit a single RLT tree on a bootstrap sample.

        Parameters
        ----------
        X_boot : np.ndarray
            Bootstrap sample of features.
        y_boot : np.ndarray
            Bootstrap sample of targets.
        params : dict
            Dictionary of tree hyperparameters.
        seed : int
            Random seed for this specific tree.
        embedded_n_jobs : int
            Number of jobs for the embedded model within this tree.

        Returns
        -------
        RLTRegression or RLTClassification
            The fitted tree instance.
        """
        if self.task_type == "regression":
            model = RLTRegression(
                "regression", random_state=seed, n_jobs=embedded_n_jobs, **params
            )
        else:
            model = RLTClassification(
                "classification", random_state=seed, n_jobs=embedded_n_jobs, **params
            )

        model.fit(X_boot, y_boot)
        return model

    def _build_forest(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Orchestrate the parallel building of the forest.

        Creates bootstrap samples and fits trees in parallel.

        Parameters
        ----------
        X : np.ndarray
            Training features.
        y : np.ndarray
            Training targets.
        """
        n_samples = len(y)
        rng = np.random.default_rng(self.random_state)

        if self.n_rlt_trees > 1:
            outer_n_jobs = self.n_jobs
            inner_n_jobs = 1
        else:
            outer_n_jobs = 1
            inner_n_jobs = self.n_jobs

        seeds = rng.integers(0, 10**9, size=self.n_rlt_trees)

        tree_params = {
            "n_estimators": self.n_extra_trees,
            "muting_rate": self.muting_rate,
            "min_protected": self.min_protected,
            "k": self.k,
            "alpha": self.alpha,
            "n_thresholds_to_try": self.n_thresholds_to_try,
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
        }

        bootstraps = []
        for _ in range(self.n_rlt_trees):
            indices = rng.choice(n_samples, n_samples, replace=True)
            bootstraps.append((X[indices], y[indices]))

        self.trees = Parallel(n_jobs=outer_n_jobs)(
            delayed(self._fit_single_tree)(
                bootstraps[i][0],
                bootstraps[i][1],
                tree_params,
                seeds[i],
                inner_n_jobs,
            )
            for i in range(self.n_rlt_trees)
        )

    def fit(
        self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.DataFrame, np.ndarray]
    ) -> Self:
        """
        Build the forest of trees from the training set (X, y).

        Parameters
        ----------
        X : array-like
            The training input samples.
        y : array-like
            The target values.

        Returns
        -------
        self
            Returns self.
        """
        self._build_forest(X, y)
        return self

    def _aggregate_results(self, X: np.ndarray) -> Union[int, float]:
        """
        Aggregate predictions from all trees in the forest.

        Parameters
        ----------
        X : np.ndarray
            Input features.

        Returns
        -------
        np.ndarray
            Aggregated predictions (mean for regression, mode for classification).
        """
        predictions = Parallel(n_jobs=self.n_jobs)(
            delayed(tree.predict)(X) for tree in self.trees
        )
        predictions = np.array(predictions)

        if self.task_type == "regression":
            return np.mean(predictions, axis=0)
        else:
            most_frequent, _ = mode(predictions, axis=0, keepdims=False)

            return most_frequent.ravel()

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Predict class or regression value for X.

        Parameters
        ----------
        X : array-like
            The input samples.

        Returns
        -------
        np.ndarray
            The predicted values.
        """
        return self._aggregate_results(X)