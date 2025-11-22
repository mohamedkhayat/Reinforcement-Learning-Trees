import pandas as pd
import numpy as np
from sklearn.preprocessing import *
from sklearn.impute import *
from sklearn.model_selection import *


datasets_dict = {
    "breast_cancer" : ("breast_cancer.csv", "diagnosis"),
    "concrete_compressive_strength": ("concrete_data.csv","Strength"),
    "parkinsons":("Parkinsson disease.csv","status"),
    "sonar": ("sonar.all-data-uci.csv","Label"),
    "eighthr": ("eighthr.csv","Class"),
    "winequality-red": ("winequality-red.csv","quality"),
    "winequality-white": ("winequality-white.csv","quality"),
    "auto-mpg": ("auto-mpg.csv","mpg"),
    "housing": ("housing.csv","MEDV")
    }

def prepare_data(nom_dataset,nom_target):
    df = pd.read_csv(nom_dataset)
    df = df.drop_duplicates()

    df = df.replace("?",np.NaN)
    X = df.drop(columns=[nom_target],axis=1)
    X = X.select_dtypes(include="number")
    y = df[nom_target]
    missing_pct = X.isnull().mean() * 100

 
    cols_to_drop = missing_pct[missing_pct > 60].index
    X_clean = X.drop(columns=cols_to_drop)

    if(X_clean.shape[1]>20):
        row_missing_pct = X_clean.isnull().mean(axis=1)
        X_clean = X_clean[row_missing_pct <= 0.5]
    
    
    y_clean = y.loc[X_clean.index]
    X_train,X_test,y_train,y_test = train_test_split(X_clean,y_clean,test_size=0.2,shuffle=True,random_state=42)
    
    imputer = KNNImputer(n_neighbors=5)
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)


    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imputed)
    X_test_scaled = scaler.transform(X_test_imputed)

    return X_train_scaled,X_test_scaled,y_train,y_test


