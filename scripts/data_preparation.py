import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split


def prepare_data(wrapper):
    df = wrapper.df

    X = df[wrapper.quantitatives_variables].copy()
    y = df[wrapper.target].copy()

    missing_pct = X.isnull().mean() * 100
    cols_to_drop = missing_pct[missing_pct > 60].index
    X_clean = X.drop(columns=cols_to_drop)

    if X_clean.shape[1] > 0:
        row_missing_pct = X_clean.isnull().mean(axis=1)
        X_clean = X_clean[row_missing_pct <= 0.5]
        y_clean = y.loc[X_clean.index]
    else:
        y_clean = y

    X_train, X_test, y_train, y_test = train_test_split(
        X_clean, y_clean, test_size=0.2, shuffle=True, random_state=42
    )

    imputer = KNNImputer(n_neighbors=5)
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imputed)
    X_test_scaled = scaler.transform(X_test_imputed)

    print(f"Shape of X_train: {X_train_scaled.shape}")
    print(f"Shape of X_test: {X_test_scaled.shape}")
    print(f"Shape of y_train: {y_train.shape}")
    print(f"Shape of y_test: {y_test.shape}")

    return X_train_scaled, X_test_scaled, y_train, y_test
