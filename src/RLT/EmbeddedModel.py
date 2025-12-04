from sklearn.tree import ExtraTreeClassifier, ExtraTreeRegressor
import numpy as np


class EmbeddedModel:
    def __init__(
        self,
        task_type,
        n_estimators=30,
        max_depth=None,
        min_samples_split=2,
        random_state=42,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        np.random.seed(random_state)
        self.task_type = task_type

    def get_variables_importances(self, X, y, valid_features):
        n_samples = X.shape[0]

        if n_samples < self.min_samples_split:
            return np.zeros((len(valid_features),))

        X_valid = X[:, valid_features]

        somme_PMSE = np.zeros(
            (len(valid_features)),
        )
        somme_MSE = 0.0

        for i in range(self.n_estimators):
            indices = np.arange(n_samples)

            # bagging
            train_idx = np.random.choice(indices, n_samples, replace=True)
            # indices   = [0, 1, 2, 3, 4, 5]
            # train_idx = [1, 1, 1, 2, 4, 4]
            # mask      = [1, 1, 1, 1, 1, 1]
            # indice[mask] = indices[5, False, 0, False, 3, False, False]
            # result = indices[5, 0, 3]
            mask = np.ones((n_samples), dtype=bool)  # [1,1 ,1,1,1,1]
            mask[train_idx] = False
            # [ 0 0 0 0 1 0 1 0] # 0 si pour train, 1 si pour oob
            # indices[mask] = [0 1 2 3 4 5 6], [False, False, False, False, 1]
            test_idx = indices[mask]

            if len(test_idx) == 0:
                continue

            X_train, y_train = X_valid[train_idx, : ], y[train_idx]
            X_oob, y_oob = X_valid[test_idx, : ], y[test_idx]

            if self.task_type == "classification":
                model = ExtraTreeClassifier(
                    min_samples_split=self.min_samples_split,
                    max_depth=self.max_depth,
                    random_state=self.random_state,
                )

            elif self.task_type == "regression":
                model = ExtraTreeRegressor(
                    min_samples_split=self.min_samples_split,
                    max_depth=self.max_depth,
                    random_state=self.random_state,
                )
                somme_MSE += np.mean((y_oob - y_pred) ** 2)

            model.fit(X_train, y_train)
            y_pred = model.predict(X_oob)

            if self.task_type == "classification":
                somme_MSE += np.mean(y_pred != y_oob)

            elif self.task_type == "regression":
                somme_MSE += np.mean((y_oob - y_pred) ** 2)

            for idx_variable in range(len(valid_features)):
                col = X_oob[:, idx_variable].copy()
                np.random.shuffle(X_oob[:, idx_variable])
                y_pred = model.predict(X_oob)
                X_oob[:, idx_variable] = col

                if self.task_type == "classification":
                    somme_PMSE[idx_variable] += np.mean(y_pred != y_oob)

                elif self.task_type == "regression":
                    somme_PMSE[idx_variable] += np.mean((y_oob - y_pred) ** 2)

        if somme_MSE == 0:
            somme_MSE = 1e-10

        vi_scores = {}

        # valid features : [1, 2, 6, 8]
        # PMSE : 0, 1, 2, 3

        for idx_local, idx_global in enumerate(valid_features):
            ratio = somme_PMSE[idx_local] / somme_MSE
            vi = ratio - 1
            vi = max(vi, 0)
            vi_scores[idx_global] = vi

        return vi_scores
