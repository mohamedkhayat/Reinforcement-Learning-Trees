import os
from matplotlib import pyplot as plt
import numpy as np
from scipy.stats import norm
from scipy.linalg import toeplitz
import pandas as pd
import matplotlib.patches as mpatches
from scripts import data_preparation
import matplotlib.image as mpimg

def generate_scenario_data(
    scenario_id: int, n_samples: int, p: int, random_state: int = None
):
    """
    Generates data based on the 4 scenarios described in the RLT paper (Zhu et al., 2015).

    Parameters:
    - scenario_id: int (1, 2, 3, or 4)
    - n_samples: int (Number of observations)
    - p: int (Number of features)
    - random_state: int, optional

    Returns:
    - X: np.array of shape (n_samples, p)
    - y: np.array of shape (n_samples,)
    """
    rng = np.random.default_rng(random_state)

    epsilon = rng.normal(0, 1, n_samples)

    if scenario_id == 1:
        X = rng.uniform(0, 1, (n_samples, p))

        term = 10 * (X[:, 0] - 1) + 20 * np.abs(X[:, 1] - 0.5)
        mu = norm.cdf(term)

        y = rng.binomial(1, mu)

        return X, y

    elif scenario_id == 2:
        X = rng.uniform(0, 1, (n_samples, p))

        pos_part = np.maximum(0, X[:, 1] - 0.25)
        y = 100 * ((X[:, 0] - 0.5) ** 2) * pos_part + epsilon

        return X, y

    elif scenario_id == 3:
        col = 0.9 ** np.arange(p)
        Sigma = toeplitz(col)

        X = rng.multivariate_normal(np.zeros(p), Sigma, n_samples)

        y = 2 * X[:, 49] * X[:, 99] + 2 * X[:, 149] * X[:, 199] + epsilon

        return X, y

    elif scenario_id == 4:
        col = 0.5 ** np.arange(p)
        Sigma = toeplitz(col)
        off_diag_mask = ~np.eye(p, dtype=bool)
        Sigma[off_diag_mask] += 0.2

        X = rng.multivariate_normal(np.zeros(p), Sigma, n_samples)

        y = 2 * X[:, 49] + 2 * X[:, 99] + 4 * X[:, 149] + epsilon

        return X, y

    else:
        raise ValueError("Scenario ID must be 1, 2, 3, or 4")


def format_table(df_results, p_dim=200):
    """
    Transforms the aggregated results to match the specific layout
    of Table 4 in the RLT paper (for a specific dimension P).
    """
    df = df_results.copy()

    if "Main_Metric" not in df.columns:
        df["Main_Metric"] = df.apply(
            lambda x: x["Error Rate"] if x["Task"] == "Classification" else x["MSE"],
            axis=1,
        )

    summary = (
        df.groupby(["Scenario", "P", "Muting", "K"])["Main_Metric"]
        .agg(["mean", "std"])
        .reset_index()
    )

    def make_string(row):
        m, s = row["mean"], row["std"]
        if pd.isna(s):
            s = 0.0

        if row["Scenario"] == 1:
            return f"{m * 100:.1f}% ({s * 100:.1f}%)"
        else:
            return f"{m:.2f} ({s:.2f})"

    summary["Result_String"] = summary.apply(make_string, axis=1)

    pivot_df = summary.pivot_table(
        index=["Muting", "K"],
        columns=["Scenario", "P"],
        values="Result_String",
        aggfunc=lambda x: x,
    )

    try:
        df_display = pivot_df.xs(p_dim, level="P", axis=1).copy()
    except KeyError:
        print(f"Dimension P={p_dim} not found in data.")
        return None

    muting_map = {0.0: "None", 0.5: "Moderate", 0.8: "Aggressive"}
    df_display = df_display.rename(index=muting_map, level="Muting")
    df_display.index.names = ["Muting", "Linear\ncombination"]

    df_display.columns = pd.MultiIndex.from_product([["RLT"], df_display.columns])

    styler = df_display.style.set_table_styles(
        [
            {
                "selector": "th",
                "props": [
                    ("text-align", "center"),
                    ("vertical-align", "middle"),
                    ("border-bottom", "1px solid black"),
                    ("font-weight", "bold"),
                ],
            },
            {
                "selector": "td",
                "props": [("text-align", "center"), ("vertical-align", "middle")],
            },
            {
                "selector": "caption",
                "props": [
                    ("caption-side", "top"),
                    ("font-size", "1.1em"),
                    ("font-weight", "bold"),
                    ("text-align", "center"),
                    ("color", "black"),
                ],
            },
        ]
    ).set_caption(f"Table : Classification/prediction error (SD), p = {p_dim}")

    return styler


def save_df_as_image(df_raw, path, p_dim=200):
    """
    1. Prepares the raw RLT results into a flat table (filtering by P).
    2. Dynamically handles column names (so it works even if you only ran Scenario 1).
    3. Saves as PNG using Matplotlib (no browser dependency).
    """

    df = df_raw.copy()

    if "Main_Metric" not in df.columns:
        df["Main_Metric"] = df.apply(
            lambda x: x["Error Rate"] if x["Task"] == "Classification" else x["MSE"],
            axis=1,
        )

    summary = (
        df.groupby(["Scenario", "P", "Muting", "K"])["Main_Metric"]
        .agg(["mean", "std"])
        .reset_index()
    )

    def make_string(row):
        m, s = row["mean"], row["std"]
        if pd.isna(s):
            s = 0.0
        if row["Scenario"] == 1:
            return f"{m * 100:.1f}% ({s * 100:.1f}%)"
        else:
            return f"{m:.2f} ({s:.2f})"

    summary["Result_String"] = summary.apply(make_string, axis=1)

    pivot_df = summary.pivot_table(
        index=["Muting", "K"],
        columns=["Scenario", "P"],
        values="Result_String",
        aggfunc=lambda x: x,
    )

    try:
        df_flat = pivot_df.xs(p_dim, level="P", axis=1).copy()
    except KeyError:
        print(f"Error: Dimension P={p_dim} not found in data.")
        return

    df_flat = df_flat.reset_index()

    muting_map = {0.0: "None", 0.5: "Moderate", 0.8: "Aggressive"}
    if "Muting" in df_flat.columns:
        df_flat["Muting"] = df_flat["Muting"].replace(muting_map)

    new_columns = []

    current_cols = list(df_flat.columns)

    new_columns.append("Muting")
    new_columns.append("Linear Comb.")

    for col in current_cols[2:]:
        new_columns.append(f"Scenario {col}")

    df_flat.columns = new_columns

    df_flat = df_flat.astype(str)

    os.makedirs(path, exist_ok=True)
    filename = os.path.join(path, f"table_{p_dim}.png")

    n_rows, n_cols = df_flat.shape
    fig_width = max(6, n_cols * 1.6)
    fig_height = max(2.5, n_rows * 0.8)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=200)
    ax.axis("off")

    table = ax.table(
        cellText=df_flat.values,
        colLabels=df_flat.columns,
        loc="center",
        cellLoc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.4)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("black")
        cell.set_linewidth(0.5)
        if row == 0:
            cell.set_facecolor("#d9d9d9")
            cell.get_text().set_weight("bold")

    ax.set_title(f"RLT Results (p = {p_dim})", fontsize=12, fontweight="bold", pad=12)

    fig.tight_layout()

    try:
        fig.savefig(filename, bbox_inches="tight")
        print(f"Image saved to {filename}")
    except Exception as e:
        print(f"Error saving image: {e}")
    finally:
        plt.close(fig)


def create_interval_labels(instance_row, train_data, feature_names):
    """
    Look at the training distribution to find which 'bin' the instance falls into.
    Returns a list of strings like: "radius_mean (10.5 < x < 15.2)"
    """
    new_labels = []

    for i, feat in enumerate(feature_names):
        val = instance_row[i]
        col_data = train_data[:, i]

        quantiles = np.percentile(col_data, [0, 25, 50, 75, 100])

        if val <= quantiles[1]:
            label = f"{feat} (x ≤ {quantiles[1]:.2f})"
        elif val <= quantiles[2]:
            label = f"{feat} ({quantiles[1]:.2f} < x ≤ {quantiles[2]:.2f})"
        elif val <= quantiles[3]:
            label = f"{feat} ({quantiles[2]:.2f} < x ≤ {quantiles[3]:.2f})"
        else:
            label = f"{feat} (x > {quantiles[3]:.2f})"

        new_labels.append(label)

    return new_labels


def _validate_and_report(wrapper):
    X_train_scaled, X_test_scaled, y_train, y_test = data_preparation.prepare_data(
        wrapper
    )

    name = (
        wrapper.name
        if hasattr(wrapper, "name")
        else getattr(wrapper, "dataset_name", "unknown")
    )
    task = wrapper.task_type

    print("\n" + "=" * 80)
    print(f"Dataset: {name} | Task: {task}")
    print(f"X_train shape: {getattr(X_train_scaled, 'shape', None)}")
    print(f"X_test  shape: {getattr(X_test_scaled, 'shape', None)}")
    print(f"y_train shape: {getattr(y_train, 'shape', (len(y_train),))}")
    print(f"y_test  shape: {getattr(y_test, 'shape', (len(y_test),))}")

    assert X_train_scaled is not None and X_train_scaled.size > 0, (
        f"{name}: X_train_scaled is empty"
    )
    assert X_test_scaled is not None and X_test_scaled.size > 0, (
        f"{name}: X_test_scaled is empty"
    )
    assert y_train is not None and len(y_train) > 0, f"{name}: y_train is empty"
    assert y_test is not None and len(y_test) > 0, f"{name}: y_test is empty"
    print("  ✔ Non-empty checks passed")

    assert X_train_scaled.shape[0] == len(y_train), (
        f"{name}: n_rows(X_train) != len(y_train)"
    )
    assert X_test_scaled.shape[0] == len(y_test), (
        f"{name}: n_rows(X_test) != len(y_test)"
    )
    print("  ✔ Row/label consistency checks passed")

    assert X_train_scaled.ndim == 2 and X_test_scaled.ndim == 2, (
        f"{name}: X arrays must be 2D"
    )
    assert X_train_scaled.shape[1] == X_test_scaled.shape[1], (
        f"{name}: n_features differ between train/test"
    )
    print("  ✔ Feature-dimension consistency passed")

    assert np.isfinite(X_train_scaled).all(), f"{name}: X_train contains NaN/Inf"
    assert np.isfinite(X_test_scaled).all(), f"{name}: X_test contains NaN/Inf"
    assert np.isfinite(y_train).all(), f"{name}: y_train contains NaN/Inf"
    assert np.isfinite(y_test).all(), f"{name}: y_test contains NaN/Inf"
    print("  ✔ No NaN/Inf values")

    try:
        mean_train = np.nanmean(X_train_scaled, axis=0)
        std_train = np.nanstd(X_train_scaled, axis=0)
        mean_dev = np.nanmax(np.abs(mean_train))
        std_dev = np.nanmax(np.abs(std_train - 1.0))

        assert mean_dev < 1e-1, (
            f"{name}: X_train feature means deviate too much from 0 (max abs mean={mean_dev:.4f})"
        )
        assert std_dev < 1e-1, (
            f"{name}: X_train feature std deviate too much from 1 (max abs std-1={std_dev:.4f})"
        )
        print(
            f"  ✔ Scaling sanity (train) passed (max abs mean={mean_dev:.4f}, max abs std-1={std_dev:.4f})"
        )
    except Exception:
        print("  ℹ Skipping scaling sanity checks (non-numeric or insufficient data)")

    if str(task).lower().startswith("c"):
        unique_classes = np.unique(y_train)
        assert unique_classes.size >= 2, (
            f"{name}: classification target must have >=2 classes"
        )
        print(
            f"  ✔ Classification target sanity passed (classes={unique_classes.tolist()})"
        )
    else:
        var_y = np.var(y_train)
        assert var_y > 0, f"{name}: regression target has zero variance"
        print(f"  ✔ Regression target sanity passed (variance={var_y:.6f})")

    print(f"Validation PASSED for dataset: {name}")
    return True


def show_global_feature_importances(feature_names, feature_importances):
    features_df = pd.DataFrame(
        {"Feature": feature_names, "Importance": feature_importances}
    )
    features_df = features_df.sort_values(by="Importance", ascending=True).reset_index(
        drop=True
    )

    colors = []
    n_features = len(features_df)
    top_k = 5

    for i in range(n_features):
        score = features_df.loc[i, "Importance"]

        is_top_k = i >= (n_features - top_k)

        if is_top_k:
            colors.append("#2ca02c")
        elif score > 1e-5:
            colors.append("#1f77b4")
        else:
            colors.append("lightgray")

    plt.figure(figsize=(10, 10))
    bars = plt.barh(
        features_df["Feature"],
        features_df["Importance"],
        color=colors,
        edgecolor="grey",
        linewidth=0.3,
    )

    first_nonzero_idx = features_df[features_df["Importance"] > 1e-5].index[0]

    if first_nonzero_idx > 0:
        plt.axhline(
            y=first_nonzero_idx - 0.5,
            color="red",
            linestyle="--",
            linewidth=1,
            alpha=0.5,
        )
        plt.text(
            features_df["Importance"].max() * 0.8,
            first_nonzero_idx - 0.5,
            " Zero Importance Threshold",
            va="center",
            color="red",
            fontsize=9,
        )

    n_zero = (features_df["Importance"] <= 1e-5).sum()
    n_weak = n_features - n_zero - top_k

    legend_handles = [
        mpatches.Patch(color="#2ca02c", label=f"Dominant Signal (Top {top_k})"),
        mpatches.Patch(color="#1f77b4", label=f"Weak Signal ({n_weak} vars)"),
        mpatches.Patch(color="lightgray", label=f"Muted / Noise ({n_zero} vars)"),
    ]
    plt.legend(handles=legend_handles, loc="lower right")

    plt.title("Global Feature Importance: Signal vs. Noise")
    plt.xlabel("Permutation Importance Score")
    plt.grid(axis="x", linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.show()

def display_tables_side_by_side(base_path, p):
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    import os
    import numpy as np

    def trim_whitespace(img, padding=10):
        """Crops out the white background from an image array."""
        if len(img.shape) == 3:
            is_white = np.all(img > 0.95, axis=-1)
        else:
            is_white = img > 0.95

        non_white_rows = np.where(~np.all(is_white, axis=1))[0]
        non_white_cols = np.where(~np.all(is_white, axis=0))[0]

        if len(non_white_rows) == 0 or len(non_white_cols) == 0:
            return img

        y1, y2 = non_white_rows[0], non_white_rows[-1]
        x1, x2 = non_white_cols[0], non_white_cols[-1]

        h, w = img.shape[:2]
        y1, y2 = max(0, y1-padding), min(h, y2+padding)
        x1, x2 = max(0, x1-padding), min(w, x2+padding)

        return img[y1:y2, x1:x2]

    # Map p to the R table index
    r_index_map = {200: 4, 500: 5, 1000: 6}
    r_idx = r_index_map.get(p, "")

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    
    path1 = os.path.join(base_path, f'python_table_p{p}.png')
    path2 = os.path.join(base_path, f'r_package_table{r_idx}_p{p}.png')
    
    # --- Python Table ---
    try:
        img1 = mpimg.imread(path1)
        axes[0].imshow(trim_whitespace(img1))
        # Added Title Here
        axes[0].set_title("Our Implementation (Python)", fontsize=14, fontweight='bold', pad=15)
    except Exception as e:
        axes[0].text(0.5, 0.5, f'Missing:\n{os.path.basename(path1)}', ha='center', va='center')
        axes[0].set_title("Our Implementation (Missing)", fontsize=14, fontweight='bold')
    
    # --- R Table ---
    try:
        img2 = mpimg.imread(path2)
        axes[1].imshow(trim_whitespace(img2))
        # Added Title Here
        axes[1].set_title("Official R Package", fontsize=14, fontweight='bold', pad=15)
    except Exception as e:
        axes[1].text(0.5, 0.5, f'Missing:\n{os.path.basename(path2)}', ha='center', va='center')
        axes[1].set_title("Official R Package (Missing)", fontsize=14, fontweight='bold')

    # Formatting to ensure alignment
    for ax in axes:
        ax.axis('off')
        ax.set_anchor('N') 

    plt.subplots_adjust(wspace=0.1) # Slight gap between tables
    plt.show()