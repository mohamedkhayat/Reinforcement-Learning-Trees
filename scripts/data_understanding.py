import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def understand_data(wrapper):
    df = wrapper.df
    target_variable = wrapper.target
    type_target = wrapper.type_target

    categorical_variables = wrapper.categorical_variables
    quantitatives_variables = wrapper.quantitatives_variables

    print(f"\n{'=' * 80}")
    print(f"\nNumber of Rows is : {len(df)}\n")
    print("===" * 20)
    print(f"\nNumber of Columns is : {df.shape[1]}\n")
    print("===" * 20)
    print("\nQualitatives Columns are : \n")

    if categorical_variables:
        for col in categorical_variables:
            print(col)
    else:
        print("None")
    print("===" * 20)

    print("\nQuantatatives Columns are : ")

    if quantitatives_variables:
        for col in quantitatives_variables:
            print(col)
    else:
        print("None")

    print("===" * 20)
    print(f"\nThe Target is : {target_variable} (Type: {type_target})\n")
    print("===" * 20)

    valeurs_manquantes = (df.isna().sum() / len(df)) * 100
    if any(valeurs_manquantes != 0):
        for name, val in zip(valeurs_manquantes.index, valeurs_manquantes):
            if val != 0:
                print(f"{name} : {val:.2f} % missing values")
    else:
        print("\nNo missing values\n")
    print("===" * 20)

    duplicated_values_pct = (df.duplicated().sum() / len(df)) * 100
    if duplicated_values_pct != 0:
        print(f"\n{duplicated_values_pct:.2f} % of rows are duplicated\n")
    else:
        print("\nNo duplicated values\n")
    print("===" * 20)

    features_with_outliers = [
        col for col in quantitatives_variables if has_outliers_iqr(df[col])
    ]

    if not features_with_outliers:
        print("\nNo significant outliers were found in any quantitative features.")
    else:
        print(
            f"\nOutliers detected in the following {len(features_with_outliers)} features: {features_with_outliers}"
        )

        plots_per_fig = 2
        num_features = len(features_with_outliers)
        if num_features > 10:
            features_with_outliers = features_with_outliers[:10]
            num_features = 10
            print("(Plotting first 10 only...)")

        for i in range(0, num_features, plots_per_fig):
            features_to_plot = features_with_outliers[i : i + plots_per_fig]
            num_subplots = len(features_to_plot)

            fig, axes = plt.subplots(1, num_subplots, figsize=(5 * num_subplots, 5))
            if num_subplots == 1:
                axes = [axes]
            elif not isinstance(axes, (list, np.ndarray)):
                axes = [axes]

            for j, col in enumerate(features_to_plot):
                sns.boxplot(y=df[col], ax=axes[j])
                axes[j].set_title(f"Boxplot for {col}", fontsize=14)
                axes[j].set_ylabel("Values", fontsize=12)

            plt.tight_layout()
            plt.show()

    print("===" * 20)
    if quantitatives_variables:
        print("\nHistplots for features (First 5)\n")
        for col in quantitatives_variables[:5]:
            plot_quantitative_histogram(df, col, bins=30)
        print("===" * 20)

    if categorical_variables:
        print("\nCountplots for features\n")
        for col in categorical_variables:
            plot_qualitative_countplot(df, col, top_n=10)
        print("===" * 20)

    if type_target == "Categorical":
        print("\nCountplot for Target\n")
        plot_qualitative_countplot(df, target_variable, top_n=10)
    else:
        print("\nHistplot for Target\n")
        plot_quantitative_histogram(df, target_variable, bins=30)

    print("===" * 20)
    if quantitatives_variables:
        plot_pairplot(
            df,
            quantitatives_variables,
            target_variable,
            max_features=10,
            is_categorical_target=(type_target == "Categorical"),
        )

    print("===" * 20)


def has_outliers_iqr(data_column):
    if data_column.dropna().empty:
        return False
    q1 = data_column.quantile(0.25)
    q3 = data_column.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers_exist = (data_column < lower_bound) | (data_column > upper_bound)
    return outliers_exist.any()


def plot_quantitative_histogram(df, col, bins=30):
    sns.set_style("whitegrid")
    plt.figure(figsize=(8, 5))
    sns.histplot(
        data=df, x=col, bins=bins, kde=True, color="skyblue", edgecolor="black"
    )
    plt.title(f"Distribution of {col}", fontsize=16, fontweight="bold")
    plt.show()


def plot_qualitative_countplot(df, col, top_n=10):
    sns.set_style("whitegrid")
    plt.figure(figsize=(10, 6))
    category_counts = df[col].value_counts().nlargest(top_n)
    sns.countplot(
        data=df,
        hue=col,
        y=col,
        order=category_counts.index,
        palette="viridis",
        legend=False,
    )
    plt.title(f"Frequency of Categories in {col}", fontsize=16, fontweight="bold")
    plt.show()


def plot_pairplot(
    df,
    quantitatives_variables,
    target_variable,
    max_features=10,
    is_categorical_target=False,
):
    sns.set_style("ticks")
    cols = quantitatives_variables[:max_features] + [target_variable]
    kwargs = {"hue": target_variable, "height": 2.5, "diag_kind": "kde"}
    if is_categorical_target and df[target_variable].nunique() <= 3:
        kwargs["markers"] = ["o", "s", "D"][: df[target_variable].nunique()]
    try:
        sns.pairplot(df[cols], **kwargs)
        plt.show()
    except Exception:
        pass
