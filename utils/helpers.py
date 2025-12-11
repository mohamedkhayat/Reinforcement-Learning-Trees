import os
from matplotlib import pyplot as plt
import numpy as np
from scipy.stats import norm
from scipy.linalg import toeplitz
import pandas as pd


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
    ).set_caption(f"Table 4: Classification/prediction error (SD), p = {p_dim}")

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