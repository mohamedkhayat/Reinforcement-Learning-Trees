from typing import Dict, Self, Union
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from scipy.stats import mode
from RLT.RLTClassification import RLTClassification
from RLT.RLTRegression import RLTRegression

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
        embedded_model: str = "extra_trees",
        n_rlt_trees: int = 10,
        n_extra_trees: int = 100,
        muting_rate: float = 0.5,
        min_protected: int = 3,
        k: int = 1,
        alpha: int = 0.25,
        n_thresholds_to_try: int = 2,
        max_depth: int = 10,
        min_samples_split: int = 5,
        n_jobs: int = 1,
        random_state: int = 42,
        use_bandit: bool = False,
        bandit_exploration: float = 1.0,
        bandit_selection_rate: float = 0.5,
    ):
        self.task_type = task_type
        self.embedded_model = embedded_model
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
        self.use_bandit = use_bandit
        self.bandit_exploration = bandit_exploration
        self.bandit_selection_rate = bandit_selection_rate

    @property
    def feature_importances_(self):
        import numpy as np

        all_importances = []
        for tree in self.trees:
            if hasattr(tree, "variable_importances_"):
                all_importances.append(tree.variable_importances_)

        if not all_importances:
            raise AttributeError("No variable_importances_ found in trees.")

        importances_matrix = np.array(all_importances)

        avg_ratios = np.mean(importances_matrix, axis=0)

        final_importance = avg_ratios - 1.0

        final_importance = np.maximum(final_importance, 0.0)

        total = np.sum(final_importance)
        if total > 0:
            final_importance /= total

        return final_importance

    def _fit_single_tree(
        self,
        X_boot: np.ndarray,
        y_boot: np.ndarray,
        X_oob: np.ndarray,
        y_oob: np.ndarray,
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
        if self.task_type.lower() == "regression":
            model = RLTRegression(
                "regression", random_state=seed, n_jobs=embedded_n_jobs, **params
            )
        else:
            model = RLTClassification(
                "classification", random_state=seed, n_jobs=embedded_n_jobs, **params
            )

        model.fit(X_boot, y_boot, X_oob, y_oob)
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
        if isinstance(y, pd.Series):
            y = y.values

        if self.task_type.lower() == "classification":
            self.classes_ = np.unique(y)
        else:
            self.classes_ = None

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
            "embedded_model": self.embedded_model,
            "muting_rate": self.muting_rate,
            "min_protected": self.min_protected,
            "k": self.k,
            "alpha": self.alpha,
            "n_thresholds_to_try": self.n_thresholds_to_try,
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
            "use_bandit": self.use_bandit,
            "bandit_exploration": self.bandit_exploration,
            "bandit_selection_rate": self.bandit_selection_rate,
        }

        bootstraps = []
        oobs = []
        for _ in range(self.n_rlt_trees):
            indices = rng.choice(n_samples, n_samples, replace=True)
            mask = np.ones(n_samples, dtype=bool)
            mask[indices] = False
            X_boot, y_boot = X[indices], y[indices]
            X_oob, y_oob = X[mask], y[mask]
            bootstraps.append((X_boot, y_boot))
            oobs.append((X_oob, y_oob))

        self.trees = Parallel(n_jobs=outer_n_jobs)(
            delayed(self._fit_single_tree)(
                bootstraps[i][0],
                bootstraps[i][1],
                oobs[i][0],
                oobs[i][1],
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
        vi_counts = [t.vi_split_count for t in self.trees]
        fallback_counts = [t.fallback_split_count for t in self.trees]

        self.average_vi_split_count = np.mean(vi_counts)
        self.average_fallback_split_count = np.mean(fallback_counts)

        self.total_vi_splits = np.sum(vi_counts)
        self.total_fallback_splits = np.sum(fallback_counts)
        self.vi_usage_ratio = (
            self.total_vi_splits / (self.total_vi_splits + self.total_fallback_splits)
            if (self.total_vi_splits + self.total_fallback_splits) > 0
            else 0.0
        )

        return self

    def _aggregate_results(self, X: np.ndarray) -> np.ndarray:
        """
        Aggregate predictions from all trees in the forest.

        Parameters
        ----------
        X : np.ndarray
            Input features.

        Returns
        -------
        np.ndarray
            Aggregated predictions (mean for regression, soft voting for classification).
        """
        predictions = Parallel(n_jobs=self.n_jobs)(
            delayed(tree.predict)(X) for tree in self.trees
        )
        if self.task_type.lower() == "regression":
            predictions = np.array(predictions)
            return np.mean(predictions, axis=0)
        else:
            predictions = np.array(predictions)
            majority_votes, _ = mode(predictions, axis=0, keepdims=False)
            return majority_votes

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Predict class probabilities for X (classification only).

        Uses soft voting by averaging probabilities across all trees.

        Parameters
        ----------
        X : array-like
            The input samples.

        Returns
        -------
        np.ndarray of shape (n_samples, n_classes)
            The class probabilities for each sample.

        Raises
        ------
        ValueError
            If called on a regression model.
        """
        if self.task_type.lower() == "regression":
            raise ValueError(
                "predict_proba is only available for classification tasks."
            )

        if isinstance(X, pd.DataFrame):
            X = X.values

        proba_list = Parallel(n_jobs=self.n_jobs)(
            delayed(tree.predict_proba)(X, self.classes_) for tree in self.trees
        )
        return np.mean(proba_list, axis=0)

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
        if self.task_type.lower() == "classification":
            # Use argmax of averaged probabilities for consistency with predict_proba
            proba = self.predict_proba(X)
            return self.classes_[np.argmax(proba, axis=1)]
        else:
            return self._aggregate_results(X)

    def print_split_statistics(self):
        """Print detailed statistics about VI vs fallback split usage."""
        print("=" * 70)
        print("🌳 RLT FOREST SPLIT STATISTICS")
        print("=" * 70)
        print(f"Number of trees:              {self.n_rlt_trees}")
        print(f"Average VI splits per tree:   {self.average_vi_split_count:.2f}")
        print(f"Average fallback splits:      {self.average_fallback_split_count:.2f}")
        print(f"Total VI splits (all trees):  {self.total_vi_splits}")
        print(f"Total fallback splits:        {self.total_fallback_splits}")
        print(f"VI usage ratio:               {self.vi_usage_ratio:.2%}")
        print("=" * 70)

        if self.vi_usage_ratio < 0.3:
            print("⚠️  WARNING: Less than 30% of splits use VI!")
            print("   → Consider lowering min_samples_split")
        elif self.vi_usage_ratio > 0.9:
            print("✅ EXCELLENT: >90% of splits use VI-based selection!")
        else:
            print("✔️  GOOD: Healthy mix of VI and fallback splits")

        print("=" * 70)