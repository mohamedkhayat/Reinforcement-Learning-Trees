import pandas as pd
import numpy as np
from ucimlrepo import fetch_ucirepo 
import seaborn as sns
import matplotlib.pyplot as plt

datasets_dict = {
    "breast_cancer" :  17   
}

def understand_data(nom_dataset):
    df = fetch_ucirepo(id=datasets_dict[nom_dataset]) 
    data = replace_missing_values(df)
    
    target_index = df.variables.index[df.variables["role"]=="Target"]
    target_variable = df.variables.loc[target_index,"name"].item()
    type_target = df.variables.loc[target_index,"type"].item()
    
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
        print("\nHistplots for features\n")
        for col in quantitatives_variables:
            plot_quantitative_histogram(data, col, bins=30)
        print("==="*20)        

     
    if categorical_variables:
        print("\nCountplots for features\n")
        for col in categorical_variables:
            plot_qualitative_countplot(data, col, top_n=10)
        print("==="*20)        

    if type_target == "Categorical":
        print("\nCountplot for Target\n")
        plot_qualitative_countplot(data, target_variable, top_n=10)
        
    else:
        print("\nHistplot for features\n")
        plot_quantitative_histogram(data, target_variable, bins=30)
    
    print("==="*20)
    if quantitatives_variables:
        plot_pairplot(data, quantitatives_variables, target_variable, max_features=10)
            
    print("==="*20)
    return categorical_variables,quantitatives_variables, target_variable

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
        hue=col,
        y=col,
        legend=False,
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

def plot_pairplot(df, quantitatives_variables, target_variable, max_features=10):
    sns.set_style("ticks")
    
    if len(quantitatives_variables) > max_features:
        features_to_plot = quantitatives_variables[:max_features]
        print(f"\nNote: Dataset has {len(quantitatives_variables)} quantitative features. Plotting a subset of the first {max_features} for Pair Plot readability.")
    else:
        features_to_plot = quantitatives_variables
        
    plot_cols = features_to_plot + [target_variable]
    
    plot_size = 2 if len(features_to_plot) > 5 else 2.5
    
    print(f"Generating Pair Plot for {len(features_to_plot)} features...")
    
    g = sns.pairplot(
        data=df[plot_cols], 
        hue=target_variable, 
        height=plot_size, 
        diag_kind="kde", 
        markers=["o", "s"] 
    )
    
    g.figure.suptitle(
        f'Pair Plot of Key Quantitative Features (Colored by {target_variable})', 
        y=1.02, 
        fontsize=16, 
        fontweight='bold'
    )
    plt.tight_layout(rect=[0, 0, 1, 1.0])
    plt.show()