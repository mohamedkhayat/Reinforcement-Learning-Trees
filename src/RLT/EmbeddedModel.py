import numpy as np
from sklearn.tree import ExtraTreeClassifier, ExtraTreeRegressor


class EmbeddedModel:
    def __init__(self, task_type, n_estimators=50, min_samples_split=2):
        self.task_type = task_type
        self.n_estimators = n_estimators
        self.min_samples_split = min_samples_split

    def get_feature_importance(self, X, y, features):
        n_samples = X.shape[0]

        if n_samples < self.min_samples_split:
            return {feat: 0.0 for feat in features}

        X_subset_features = X[:, features]

        sum_baseline_error = 0.0

        trees_with_oob = 0

        perm_errors = np.zeros(len(features))
        for i in range(self.n_estimators):
            indices = np.arange(n_samples)
            train_idx = np.random.choice(indices, n_samples, replace=True)

            mask = np.ones(n_samples, dtype=bool)
            mask[train_idx] = False
            oob_idx = indices[mask]

            if len(oob_idx) == 0:
                continue

            trees_with_oob += 1

            X_train, y_train = X_subset_features[train_idx], y[train_idx]
            X_oob, y_oob = X_subset_features[oob_idx], y[oob_idx]

            if self.task_type == "regression":
                model = ExtraTreeRegressor(
                    min_samples_split=self.min_samples_split, max_features="sqrt"
                )
            else:
                model = ExtraTreeClassifier(
                    min_samples_split=self.min_samples_split, max_features="sqrt"
                )

            model.fit(X_train, y_train)

            baseline_preds = model.predict(X_oob)

            if self.task_type == "regression":
                error = np.sum((baseline_preds - y_oob) ** 2)
            else:
                error = np.sum(baseline_preds != y_oob)

            sum_baseline_error += error

            for local_feature_idx in range(len(features)):
                original_col = X_oob[:, local_feature_idx].copy()
                np.random.shuffle(X_oob[:, local_feature_idx])
                permutation_preds = model.predict(X_oob)
                X_oob[:, local_feature_idx] = original_col

                if self.task_type == "regression":
                    perm_error = np.sum((permutation_preds - y_oob) ** 2)
                else:
                    perm_error = np.sum(permutation_preds != y_oob)

                perm_errors[local_feature_idx] += perm_error

        if trees_with_oob == 0:
            return {feat: 0.0 for feat in features}

        if sum_baseline_error == 0:
            sum_baseline_error = 1e-10

        vi_scores = {}

        for local_idx, actual_idx in enumerate((features)):
            vi = perm_errors[local_idx] / sum_baseline_error - 1
            vi_scores[actual_idx] = max(0, vi)

        return vi_scores
