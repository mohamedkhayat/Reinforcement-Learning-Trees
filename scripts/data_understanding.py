import pandas as pd
import numpy as np
from ucimlrepo import fetch_ucirepo 

datasets_dict = {"breast_cancer" :  17}
def understand_data(nom_dataset):
    df = fetch_ucirepo(id=datasets_dict[nom_dataset]) 
    
    # data (as pandas dataframes) 
    X = df.data.features 
    y = df.data.targets 
    
    # metadata 
    #print(df.metadata) 
    
    # variable information 
    print(df.variables)
    
    categorical_variables = [ name for (name,var_type) in zip(df.variables["name"],df.variables["type"]) if var_type=="Categorical" ]
    quantatatives_variables = [name for name in df.variables["name"] if name not in categorical_variables]
    return categorical_variables,quantatatives_variables
    
    