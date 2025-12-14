import os
import numpy as np
import pandas as pd

datasets_dict = {
    "breast_cancer": {
        "path": "breast_cancer.csv",
        "target": "diagnosis",
        "id_col": "id",
        "type": "Categorical",
    },
    "concrete": {
        "path": "concrete_data.csv",
        "target": "Strength",
        "type": "Continuous",
    },
    "parkinsons": {
        "path": "Parkinsson disease.csv",
        "target": "status",
        "id_col": "name",
        "type": "Categorical",
    },
    "sonar": {
        "path": "sonar.all-data-uci.csv",
        "target": "Label",
        "type": "Categorical",
    },
    "wine_red": {
        "path": "winequality-red.csv",
        "target": "quality",
        "type": "Categorical",
    },
    "wine_white": {
        "path": "winequality-white.csv",
        "target": "quality",
        "type": "Categorical",
    },
    "auto_mpg": {"path": "auto-mpg.csv", "target": "mpg", "type": "Continuous"},
    "housing": {"path": "housing.csv", "target": "MEDV", "type": "Continuous"},
    "eighthr": {
        "path": "eighthr.csv",
        "target": "Class",
        "type": "Categorical",
    },
}


class DatasetWrapper:
    def __init__(self, name):
        config = datasets_dict[name]
        path = config["path"]

        self.target = config["target"]
        self.id_col = config.get("id_col", None)

        self.type_target = config["type"]

        current_dir = os.path.dirname(os.path.abspath(_file_))
        project_root = os.path.dirname(current_dir)
        full_path = os.path.join(project_root, "datasets", path)

        self.df = pd.read_csv(full_path, na_values=["?", "nan", "NaN", ""])
        self.df = self.df.drop_duplicates()

        all_numerics = self.df.select_dtypes(include=[np.number]).columns.tolist()

        cols_to_exclude = [self.target]
        if self.id_col:
            cols_to_exclude.append(self.id_col)

        self.quantitatives_variables = [
            c for c in all_numerics if c not in cols_to_exclude
        ]

        all_columns = self.df.columns.tolist()
        self.categorical_variables = [
            c
            for c in all_columns
            if (c not in self.quantitatives_variables and c not in cols_to_exclude)
        ]

        self.clean_variables = []
        self.class_names = None