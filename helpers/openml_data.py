import openml
import numpy as np
import pickle
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

## adapted from Max's GBT vs MLP expt

dataset_id_list = [3, 6, 11, 12, 14, 16, 18, 22, 23, 28, 31, 32, 37, 44, 46, 50, 54, 151, 182, 307, 300, 458, 469, 554, 1049, 
                   1050, 1063, 1067, 1068, 4134, 1510, 1489, 1494, 1497, 1501, 1480, 1485, 1486, 1487, 1468, 1475, 1462, 1464, 4534, 
                1461, 4538, 1478, 40499, 40668, 40982, 40994, 40983, 40975, 40984, 40979, 40996, 41027, 23517, 40923, 40927, 40978, 40670, 40701, 41166]

# #dataset_id_list = [1486, 1487, 1468, 1475, 1462, 1464, 4534, 
#                 1461, 4538, 1478, 40499, 40668, 40982, 40994, 40983, 40975, 40984, 40979, 40996, 41027, 23517, 40923, 40927, 40978, 40670, 40701, 41166]

image_id_list = [6, 12, 14, 16, 18, 22, 28, 32, 182, 300, 307, 554, 1489, 1501, 40984, 40979, 40996, 40923, 40927]

tabular_id_list = [item for item in dataset_id_list if item not in image_id_list]
# print(len(dataset_id_list))


def get_data(dataset_id):
    dataset = openml.datasets.get_dataset(dataset_id)
    X, y, _, _ = dataset.get_data(dataset_format="dataframe")
    
    if 'Contraceptive_method_used' in list(X.columns):
        class_col = 'Contraceptive_method_used'
    elif 'Class' in list(X.columns):
        class_col = 'Class'
    elif 'Author' in list(X.columns):
        class_col = 'Author'
    elif 'Prevention' in list(X.columns):
        class_col = 'Prevention'
    elif 'c' in list(X.columns):
        class_col = 'c'
    elif 'problems' in list(X.columns):
        class_col = 'problems'
    elif 'defects' in list(X.columns):
        class_col = 'defects'
    elif 'target' in list(X.columns):
        class_col = 'target'
    elif 'Result' in list(X.columns):
        class_col = 'Result'
    elif 'Phase' in list(X.columns):
        class_col = 'Phase'
    elif 'outcome' in list(X.columns):
        class_col = 'outcome'
    elif 'attribute_21' in list(X.columns):
        class_col = 'attribute_21'
    elif 'character' in list(X.columns):
        class_col = 'character'
    else:
        class_col = 'class'
        
    # print("Dataset ID:", dataset_id, "| Dataset Name:", dataset.name, "| Unique Labels:", X[class_col].nunique())
    
    # Create a LabelEncoder object
    le = LabelEncoder()
    ohe = OneHotEncoder()
    df = X

    # Iterate over all columns
    cat_columns = []
    for col in df.columns:
        # Check if the column is of object type (categorical)
        if df[col].dtype == 'category' or df[col].dtype == 'object':
            if col == class_col:
                continue
            cat_columns.append(col)

    # Fit and transform the dataframe
    encoded_df = pd.DataFrame(ohe.fit_transform(df[cat_columns]).toarray())
    # Rename the columns in the encoded dataframe
    encoded_df.columns = ohe.get_feature_names_out(cat_columns)
    # Concatenate the encoded dataframe with the original dataframe
    final_df = pd.concat([df, encoded_df], axis=1)
    final_df = final_df.drop(columns=cat_columns)
            
    X = final_df.drop([class_col], axis=1).values
    y = le.fit_transform(final_df[class_col])

    return X, y

def _get_description(dataset_id):
    dataset = openml.datasets.get_dataset(dataset_id)
    X, y, _, _ = dataset.get_data(dataset_format="dataframe")
    
    if 'Contraceptive_method_used' in list(X.columns):
        class_col = 'Contraceptive_method_used'
    elif 'Class' in list(X.columns):
        class_col = 'Class'
    elif 'Author' in list(X.columns):
        class_col = 'Author'
    elif 'Prevention' in list(X.columns):
        class_col = 'Prevention'
    elif 'c' in list(X.columns):
        class_col = 'c'
    elif 'problems' in list(X.columns):
        class_col = 'problems'
    elif 'defects' in list(X.columns):
        class_col = 'defects'
    elif 'target' in list(X.columns):
        class_col = 'target'
    elif 'Result' in list(X.columns):
        class_col = 'Result'
    elif 'Phase' in list(X.columns):
        class_col = 'Phase'
    elif 'outcome' in list(X.columns):
        class_col = 'outcome'
    elif 'attribute_21' in list(X.columns):
        class_col = 'attribute_21'
    elif 'character' in list(X.columns):
        class_col = 'character'
    else:
        class_col = 'class'
        
    # print("Dataset ID:", dataset_id, "| Dataset Name:", dataset.name, "| Unique Labels:", X[class_col].nunique())
    
    # Create a LabelEncoder object
    le = LabelEncoder()
    ohe = OneHotEncoder()
    df = X
    
    categorical_cols = 0

    # Iterate over all columns
    cat_columns = []
    for col in df.columns:
        # Check if the column is of object type (categorical)
        if df[col].dtype == 'category' or df[col].dtype == 'object':
            if col == class_col:
                continue
            cat_columns.append(col)
            categorical_cols += 1
    

    return dataset_id, dataset.name, len(df), len(dataset.features) - 1, categorical_cols, X[class_col].nunique()


def get_description_table(id_list):
    columns = ['id','name','samples','features','categories', 'classes']
    rows = [_get_description(i) for i in id_list]
    
    return pd.DataFrame(rows, columns=columns)