import numpy as np
import time
import pandas as pd
from sklearn.datasets import make_regression, make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

# Local Import
# Ensure ReinforcementLearningTrees.py is in the same folder or in python path
from RLT import ReinforcementLearningTrees


def test_regression():
    print("\n" + "=" * 60)
    print("--- 1. Testing Regression (Sparse Signal) ---")
    print("=" * 60)

    # 1. Generate Data (High dimensional, Sparse)
    # 50 features, only 5 informative. This is RLT's "home turf".
    X, y = make_regression(
        n_samples=500, n_features=50, n_informative=5, noise=5.0, random_state=42
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 2. Fit RLT Forest
    print(f"Fitting RLT Forest (10 Trees) on {len(X_train)} samples...")
    start_rlt = time.time()

    rlt_forest = ReinforcementLearningTrees(
        task_type="regression",
        n_rlt_trees=10,  # Ensemble size
        n_extra_trees=50,  # Embedded model size (Heavy computation)
        muting_rate=0.5,  # Mute 50% of noise variables at every split
        min_protected=3,
        k=2,  # Linear Combinations enabled
        alpha=0.1,
        n_thresholds_to_try=10,
        max_depth=8,
        min_samples_split=5,
        n_jobs=-1,
        random_state=42,
    )
    rlt_forest.fit(X_train, y_train)
    rlt_time = time.time() - start_rlt

    preds_rlt = rlt_forest.predict(X_test)
    mse_rlt = mean_squared_error(y_test, preds_rlt)

    # 3. Fit Sklearn Random Forest
    print("Fitting Random Forest (10 Trees)...")

    # We set n_jobs=1 to make the comparison fair against single-threaded Python RLT
    rf = RandomForestRegressor(
        n_estimators=10, max_depth=8, min_samples_split=5, n_jobs=1, random_state=42
    )

    start_rf = time.time()
    rf.fit(X_train, y_train)
    rf_time = time.time() - start_rf  # Calculate Duration

    preds_rf = rf.predict(X_test)
    mse_rf = mean_squared_error(y_test, preds_rf)

    # 4. Results
    print(f"\n{'Model':<20} | {'Time (s)':<10} | {'MSE':<10}")
    print("-" * 45)
    print(f"{'RLT Forest':<20} | {rlt_time:<10.2f} | {mse_rlt:<10.4f}")
    print(f"{'Random Forest':<20} | {rf_time:<10.4f} | {mse_rf:<10.4f}")

    improvement = ((mse_rf - mse_rlt) / mse_rf) * 100
    print("-" * 45)
    if improvement > 0:
        print(f"✅ RLT reduced error by {improvement:.2f}%")
    else:
        print(f"❌ RLT was worse by {abs(improvement):.2f}%")

    print(
        f"ℹ️  Time Ratio: RLT is {rlt_time / rf_time:.1f}x slower (Expected due to Embedded Models)"
    )


def test_classification():
    print("\n" + "=" * 60)
    print("--- 2. Testing Classification (Correlated Features) ---")
    print("=" * 60)

    # 1. Generate Data
    # n_redundant=10 creates correlations where Linear Combo (k=2) shines
    X, y = make_classification(
        n_samples=500,
        n_features=30,
        n_informative=5,
        n_redundant=10,
        n_classes=2,
        random_state=42,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 2. Fit RLT Forest
    print(f"Fitting RLT Forest (10 Trees) on {len(X_train)} samples...")
    start_rlt = time.time()

    rlt_forest = ReinforcementLearningTrees(
        task_type="classification",
        n_rlt_trees=10,
        n_extra_trees=50,
        muting_rate=0.5,
        min_protected=3,
        k=2,  # Linear Combinations
        alpha=0.1,
        n_thresholds_to_try=10,
        max_depth=8,
        min_samples_split=5,
        n_jobs=-1,
        random_state=42,
    )
    rlt_forest.fit(X_train, y_train)
    rlt_time = time.time() - start_rlt

    preds_rlt = rlt_forest.predict(X_test)
    acc_rlt = accuracy_score(y_test, preds_rlt)

    # 3. Fit Sklearn Random Forest
    print("Fitting Random Forest (10 Trees)...")
    rf = RandomForestClassifier(
        n_estimators=10, max_depth=8, min_samples_split=5, n_jobs=1, random_state=42
    )

    start_rf = time.time()
    rf.fit(X_train, y_train)
    rf_time = time.time() - start_rf  # Calculate Duration

    preds_rf = rf.predict(X_test)
    acc_rf = accuracy_score(y_test, preds_rf)

    # 4. Results
    print(f"\n{'Model':<20} | {'Time (s)':<10} | {'Accuracy':<10}")
    print("-" * 45)
    print(f"{'RLT Forest':<20} | {rlt_time:<10.2f} | {acc_rlt:<10.4f}")
    print(f"{'Random Forest':<20} | {rf_time:<10.4f} | {acc_rf:<10.4f}")

    print("-" * 45)
    if acc_rlt > acc_rf:
        print(f"✅ RLT is {(acc_rlt - acc_rf) * 100:.2f}% more accurate")
    else:
        print(f"❌ RLT is {(acc_rf - acc_rlt) * 100:.2f}% less accurate")

    print(f"ℹ️  Time Ratio: RLT is {rlt_time / rf_time:.1f}x slower")


if __name__ == "__main__":
    test_regression()
    test_classification()