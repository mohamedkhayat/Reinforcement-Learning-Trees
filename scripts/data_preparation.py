from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def prepare_data(wrapper):
    df = wrapper.df

    X = df[wrapper.quantitatives_variables].copy()
    y = df[wrapper.target].copy()

    missing_pct = X.isnull().mean() * 100
    cols_to_drop = missing_pct[missing_pct > 60].index
    X_clean = X.drop(columns=cols_to_drop)

    wrapper.clean_variables = X_clean.columns.tolist()

    if X_clean.shape[1] > 0:
        row_missing_pct = X_clean.isnull().mean(axis=1)
        X_clean = X_clean[row_missing_pct <= 0.5]
        y_clean = y.loc[X_clean.index]
    else:
        y_clean = y

    task_type = (
        "classification" if wrapper.type_target == "Categorical" else "regression"
    )

    if task_type == "classification":
        le = LabelEncoder()
        y_clean = le.fit_transform(y_clean)
        class_names = list(le.classes_)
        wrapper.class_names = class_names

    stratify_type = None if task_type == "regression" else y_clean

    X_train, X_test, y_train, y_test = train_test_split(
        X_clean,
        y_clean,
        train_size=150,
        shuffle=True,
        random_state=42,
        stratify=stratify_type,
    )

    imputer = KNNImputer(n_neighbors=5)
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imputed)
    X_test_scaled = scaler.transform(X_test_imputed)

    wrapper.scaler = scaler

    return X_train_scaled, X_test_scaled, y_train, y_test
