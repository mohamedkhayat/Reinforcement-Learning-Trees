import pandas as pd
import numpy as np
from ucimlrepo import fetch_ucirepo 
import seaborn as sns
import matplotlib.pyplot as plt
datasets_dict = {"breast_cancer" :  17}

def understand_data(nom_dataset):
    df = fetch_ucirepo(id=datasets_dict[nom_dataset]) 
    data = replace_missing_values(df)
    
    target_variable = df.variables.loc[df.variables.index[df.variables["role"]=="Target"],"name"].item()
    categorical_variables = [name for (name,var_type) in zip(df.variables["name"],df.variables["type"]) if (var_type=="Categorical" and name!=target_variable and name!="ID")]
    quantitatives_variables = [name for name in df.variables["name"] if (name not in categorical_variables and name!=target_variable and name!="ID")]
    
    print(f"\nNumber of Lignes is : {len(data)}\n")
    print("==="*20)
    print(f"\nNumber of Columns is : {data.shape[1]}\n")
    print("==="*20)
    print("\nQualitatives Columns are : \n")

    if categorical_variables:
        for col in categorical_variables:
            print(col)
    print("==="*20)
    
    print("\nQuantatatives Columns are : ")

    if quantitatives_variables:
        for col in quantitatives_variables:
            print(col)
            
    print("==="*20)
    print(f"\nThe Target is : {target_variable}\n")
    print("==="*20)
    valeurs_manquantes = (data.isna().sum() / len(data)) * 100
    
    if any(valeurs_manquantes != 0):
        for name, val in zip(list(valeurs_manquantes.index, list(valeurs_manquantes))):
            if val != 0:
                print(f"{name} : {val} % missing values")
    else:
        print("\nNo missing values\n")
    print("==="*20)
    
    duplicated_values_pct = (data.duplicated().sum() / len(data)) * 100
    
    if duplicated_values_pct != 0:
        print(f"\n{duplicated_values_pct} % of lignes are duplicated\n")
    else:
        print("\nNo duplicated values\n")
    print("==="*20) 
    
    features_with_outliers = [
    col for col in quantitatives_variables if has_outliers_iqr(data[col])
]

    if not features_with_outliers:
        print("\nNo significant outliers were found in any quantitative features.")
    else:
        print(f"\nOutliers detected in the following {len(features_with_outliers)} features: {features_with_outliers}")
        
        plots_per_fig = 2
        num_features = len(features_with_outliers)
        
        for i in range(0, num_features, plots_per_fig):
            features_to_plot = features_with_outliers[i : i + plots_per_fig]
            
            num_subplots = len(features_to_plot)
            
            fig, axes = plt.subplots(1, num_subplots, figsize=(5 * num_subplots, 7))
            
            if num_subplots == 1:
                axes = [axes]
                
            for j, col in enumerate(features_to_plot):
                sns.boxplot(y=data[col], ax=axes[j])
                axes[j].set_title(f'Boxplot for {col}', fontsize=14)
                axes[j].set_xlabel('')
                axes[j].set_ylabel('Values', fontsize=12)

            plt.tight_layout()
            plt.show()
        
        print("==="*20)    
    
    if quantitatives_variables:
        for col in quantitatives_variables:
            plot_quantitative_histogram(data, col, bins=30)
    
    if categorical_variables:
        for col in categorical_variables:
            plot_qualitative_countplot(data, col, top_n=10)
    
    return categorical_variables,quantitatives_variables, target_variable


# les histogrames features quantitatives w countplot lil features qualitatives
# pairplot

def has_outliers_iqr(data_column):
    q1 = data_column.quantile(0.25)
    q3 = data_column.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers_exist = (data_column < lower_bound) | (data_column > upper_bound)
    return outliers_exist.any()


def boxplot(data,col):
    sns.set_style("whitegrid")

    plt.figure(figsize=(6, 8))

    sns.boxplot(y=data[col])

    plt.title(f'Boxplot for {col}', fontsize=16)
    plt.ylabel('Values', fontsize=12)

    plt.show()
    
    
def plot_quantitative_histogram(df, col, bins=30):
    sns.set_style("whitegrid")
    plt.figure(figsize=(8, 5))

    sns.histplot(
        data=df,
        x=col,
        bins=bins,
        kde=True,
        color='skyblue',
        edgecolor='black',
        line_kws={'linewidth': 2, 'color': 'navy', 'alpha': 0.8}
    )

    plt.title(f'Distribution of {col}', fontsize=16, fontweight='bold')
    plt.xlabel(col, fontsize=12)
    plt.ylabel('Frequency / Density', fontsize=12)

    mean_val = df[col].mean()
    plt.axvline(mean_val, color='red', linestyle='--', linewidth=1.5, label=f'Mean: {mean_val:.2f}')
    plt.legend()

    plt.tight_layout()
    plt.show()

def plot_qualitative_countplot(df, col, top_n=10):
    sns.set_style("whitegrid")
    plt.figure(figsize=(10, 6))

    category_counts = df[col].value_counts().nlargest(top_n)
    
    ax = sns.countplot(
        data=df,
        y=col,
        order=category_counts.index,
        palette='viridis'
    )

    total = len(df)
    for p in ax.patches:
        count = p.get_width()
        
        percentage = f'{(100 * count / total):.1f}%'
        
        x = p.get_width()
        y = p.get_y() + p.get_height() / 2
        
        ax.annotate(f'{count} ({percentage})', (x, y), ha='left', va='center', 
                    xytext=(5, 0), textcoords='offset points', fontsize=10)

    plt.title(f'Frequency of Categories in {col}', fontsize=16, fontweight='bold')
    plt.xlabel('Count', fontsize=12)
    plt.ylabel(col, fontsize=12)
    
    plt.tight_layout()
    plt.show()
    
    
def replace_missing_values(df):
    missing_symbol = df.metadata.get('missing_values_symbol')
    
    X = df.data.features 
    y = df.data.targets
    
    if missing_symbol is not None and missing_symbol != '':
        symbol_str = str(missing_symbol)
        print(f"Replacing custom missing value symbol '{symbol_str}' with np.nan...")
        X.replace(symbol_str, np.nan, inplace=True)
        y.replace(symbol_str, np.nan, inplace=True)
        
    data = pd.concat([X, y], axis=1)

    return data

# ------------------- New: Generative / Augmentation utilities -------------------
def generate_synthetic_dataset(n=1000, p=50, scenario="classification_signal", 
                               signal_features=None, noise_scale=1.0, random_state=None):
    """
    Returns (X: DataFrame, y: Series) synthetic dataset for quick benchmarking.
    Scenarios:
      - "classification_signal": binary target from linear signal on signal_features
      - "nonlinear": binary target from nonlinear function of signal_features
      - "correlated": features generated with latent factors for correlation
      - "regression_linear": continuous target as linear combination + noise
    """
    rng = np.random.default_rng(random_state)
    if signal_features is None:
        signal_features = [0, 1]  # default signal columns

    # base noise features
    X = pd.DataFrame(rng.normal(size=(n, p)), columns=[f"X{i+1}" for i in range(p)])

    if scenario == "classification_signal":
        score = np.zeros(n)
        for idx in signal_features:
            score += X.iloc[:, idx]
        score += rng.normal(scale=noise_scale, size=n)
        y = (score > np.median(score)).astype(int)
    elif scenario == "nonlinear":
        score = np.zeros(n)
        for idx in signal_features:
            xcol = X.iloc[:, idx]
            score += np.sin(xcol) + (xcol ** 2) * 0.1
        score += rng.normal(scale=noise_scale, size=n)
        y = (score > np.median(score)).astype(int)
    elif scenario == "correlated":
        # create k latent factors and project to features to induce correlation
        k = max(2, len(signal_features))
        factors = rng.normal(size=(n, k))
        loadings = rng.normal(scale=0.8, size=(k, p))
        X = pd.DataFrame(factors @ loadings + rng.normal(scale=0.1, size=(n, p)),
                         columns=[f"X{i+1}" for i in range(p)])
        # signal from some linear combination of specific features
        score = X.iloc[:, signal_features].sum(axis=1) + rng.normal(scale=noise_scale, size=n)
        y = (score > np.median(score)).astype(int)
    elif scenario == "regression_linear":
        coef = np.zeros(p)
        for idx in signal_features:
            coef[idx] = 1.0
        y = X.values.dot(coef) + rng.normal(scale=noise_scale, size=n)
    else:
        raise ValueError(f"Unknown scenario: {scenario}")

    # return DataFrame and Series
    return X, pd.Series(y, name="target")

def add_noisy_covariates(df, target_col="target", target_p=500, correlated=False, corr_strength=0.9, random_state=None):
    """
    Expand df (pandas DataFrame including target_col) to have target_p total features (excluding target).
    If correlated=True, generate covariates that are correlated with existing features.
    """
    rng = np.random.default_rng(random_state)
    features = [c for c in df.columns if c != target_col]
    current_p = len(features)
    needed = max(0, target_p - current_p)
    if needed == 0:
        return df.copy()

    new_cols = {}
    for i in range(needed):
        name = f"noise_{current_p + i + 1}"
        if correlated and current_p > 0:
            # correlate with a random existing feature
            base = df[rng.choice(features)].values
            new_cols[name] = corr_strength * base + (1 - corr_strength) * rng.normal(size=len(df))
        else:
            new_cols[name] = rng.normal(size=len(df))
    augmented = df.copy()
    for k, v in new_cols.items():
        augmented[k] = v
    return augmented

def mixup_augmentation(df, target_col="target", alpha=0.2, n_aug=1000, random_state=None):
    """
    Simple MixUp augmentation for numeric features.
    Returns augmented DataFrame (original + synthetic n_aug samples).
    """
    rng = np.random.default_rng(random_state)
    features = [c for c in df.columns if c != target_col]
    X = df[features].values
    y = df[target_col].values
    n = len(df)

    aug_X = []
    aug_y = []
    for _ in range(n_aug):
        i, j = rng.integers(0, n, size=2)
        lam = rng.beta(alpha, alpha)
        x_new = lam * X[i] + (1 - lam) * X[j]
        y_new = lam * y[i] + (1 - lam) * y[j]
        aug_X.append(x_new)
        aug_y.append(y_new)

    aug_df = pd.DataFrame(np.vstack([X, np.array(aug_X)]), columns=features)
    aug_target = pd.Series(np.hstack([y, np.array(aug_y)]), name=target_col)
    result = pd.concat([aug_df, aug_target], axis=1)
    return result

def smote_augment(df, target_col="target", sampling_strategy="auto", random_state=None):
    """
    Wrapper for SMOTE augmentation if imblearn is installed.
    If imblearn is missing, returns the original df and prints an install hint.
    """
    try:
        from imblearn.over_sampling import SMOTE
    except Exception:
        print("imblearn not available. To use SMOTE install: pip install imbalanced-learn")
        return df.copy()

    features = [c for c in df.columns if c != target_col]
    X = df[features].values
    y = df[target_col].values
    sm = SMOTE(sampling_strategy=sampling_strategy, random_state=random_state)
    X_res, y_res = sm.fit_resample(X, y)
    res_df = pd.DataFrame(X_res, columns=features)
    res_df[target_col] = y_res
    return res_df