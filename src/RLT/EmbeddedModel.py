from typing import Dict, List, Tuple
import warnings
from sklearn.tree import ExtraTreeClassifier, ExtraTreeRegressor
import numpy as np
from joblib import Parallel, delayed
from lightgbm import LGBMClassifier, LGBMRegressor
import lightgbm as lgb

warnings.filterwarnings("ignore", message=".*X does not have valid feature names.*")
warnings.filterwarnings(
    "ignore", category=UserWarning, module="sklearn.utils.validation"
)


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
        model: str = "extra_trees",
        n_estimators: int = 30,
        max_depth: int = None,
        min_samples_split: int = 2,
        n_jobs: int = 1,
        seed: int = 42,
    ):
        self.n_estimators = n_estimators
        self.model = model
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

        sum_ratios = {feat_idx: 0.0 for feat_idx in valid_features}
        valid_tree_count = 0

        for res in results:
            if res is None:
                continue

            base_mse, pmse_dict = res
            if base_mse < 1e-10:
                for local_idx, feat_idx in enumerate(valid_features):
                    sum_ratios[feat_idx] += 1.0
            else:
                for local_idx, feat_idx in enumerate(valid_features):
                    ratio = pmse_dict[local_idx] / base_mse
                    sum_ratios[feat_idx] += ratio

            valid_tree_count += 1

        vi_scores = {}

        if valid_tree_count == 0:
            return {feat: 0.0 for feat in valid_features}

        for feat_idx in valid_features:
            avg_ratio = sum_ratios[feat_idx] / valid_tree_count
            vi = avg_ratio - 1

            vi_scores[feat_idx] = vi
            
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

        try:
            rng = np.random.default_rng(seed)
            n_samples = X.shape[0]
            indices = np.arange(n_samples) # [0, 1, 2, 3, 4, ..., n_samples - 1]
                                                                     # X : 10 lignes
            train_idx = rng.choice(indices, n_samples, replace=True) # train_idx :               [0,0,0, 1, 2 ,2 ,2 ,3 ,4 ,5 ,5]
                                                                     # mask      :               [0,0,0, 0, 0 ,0 ,1 ,1 ,1 ,1 ,1]
                                                                     # mask[train_idx] = False : [1,1,1, 1, 1 ,1 ,1 ,1 ,1 ,1 ,1]
            mask = np.ones((n_samples), dtype=bool)
            mask[train_idx] = False
            test_idx = indices[mask]

            if len(test_idx) == 0:
                return None

            X_subset = X[:, valid_features].copy()
            X_train, y_train = X_subset[train_idx, :], y[train_idx]
            X_oob, y_oob = X_subset[test_idx, :], y[test_idx]

            model_type_str = getattr(
                self, "model_type", getattr(self, "model", "extra_trees")
            ).lower()

            if self.task_type.lower() == "classification":
                unique_train = np.unique(y_train)
                if len(unique_train) < 2:
                    return None

                if model_type_str == "lightgbm":
                    unique_oob = np.unique(y_oob)
                    if not np.isin(unique_oob, unique_train).all():
                        return None

            ex_params = {
                "min_samples_split": self.min_samples_split,
                "max_depth": None,
                "random_state": seed,
                "max_features": 0.5,
            }

            lgbm_params = {
                "n_estimators": 25,
                "max_depth": self.max_depth if self.max_depth else 3,
                "min_child_samples": max(1, int(0.1 * len(train_idx))),
                "random_state": seed,
                "n_jobs": 1,
                "verbose": -1,
                "silent": True,
                "force_col_wise": True,
            }

            if self.task_type.lower() == "classification":
                if model_type_str == "extra_trees":
                    model = ExtraTreeClassifier(**ex_params)
                elif model_type_str == "lightgbm":
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        model = LGBMClassifier(**lgbm_params)
                else:
                    return None

            elif self.task_type.lower() == "regression":
                if model_type_str == "extra_trees":
                    model = ExtraTreeRegressor(**ex_params)
                elif model_type_str == "lightgbm":
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        model = LGBMRegressor(**lgbm_params)
                else:
                    raise ValueError("Model not recognised")
            else:
                return None

            if model_type_str == "lightgbm":
                callbacks = [
                    lgb.early_stopping(stopping_rounds=5, verbose=False),
                    lgb.log_evaluation(period=0),
                ]

                metric = (
                    "binary_logloss"
                    if self.task_type.lower() == "classification"
                    else "l2"
                )

                model.fit(
                    X_train,
                    y_train,
                    eval_set=[(X_oob, y_oob)],
                    eval_metric=metric,
                    callbacks=callbacks,
                )
            else:
                model.fit(X_train, y_train)

            y_pred = model.predict(X_oob)

            if self.task_type.lower() == "classification":
                base_error = np.mean(y_pred != y_oob)
            elif self.task_type.lower() == "regression":
                base_error = np.mean((y_oob - y_pred) ** 2)

            if base_error < 1e-15:
                return 0.0, np.zeros(len(valid_features))

            pmse_contributions = np.zeros(len(valid_features))
            for local_idx, global_idx in enumerate(valid_features):
                original_col = X_oob[:, local_idx].copy()
                rng.shuffle(X_oob[:, local_idx])

                y_pred_perm = model.predict(X_oob)
                X_oob[:, local_idx] = original_col

                if self.task_type.lower() == "classification":
                    pmse = np.mean(y_pred_perm != y_oob)
                elif self.task_type.lower() == "regression":
                    pmse = np.mean((y_oob - y_pred_perm) ** 2)

                pmse_contributions[local_idx] = pmse

            return base_error, pmse_contributions

        except Exception:
            return None