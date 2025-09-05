import pandas as pd
from sklearn.preprocessing import StandardScaler
from helpers.tabular_data import data_loaders, get_label_idx
from sklearn.preprocessing import LabelEncoder

def standardize_dataset(train_df, df_list_to_standardize, y_column_name):

    y_train = train_df[y_column_name]
    X_train = train_df.drop([y_column_name], axis=1)

    # fit standardizer
    sc = StandardScaler()
    scale = sc.fit(X_train)
    
    scaled_df_list = []
    for df in df_list_to_standardize:
        # print(df.columns)
        y = df[y_column_name]
        X = df.drop([y_column_name], axis=1)
        
        x_transformed = scale.transform(X)
        df_standardized = pd.DataFrame(x_transformed, columns = X_train.columns)
        scaled_df = pd.concat([
            df_standardized.reset_index(drop=True), 
            y.reset_index(drop=True)
        ], axis=1)
        
        scaled_df_list.append(scaled_df)

    return scaled_df_list


def get_data_df(dataset_name):
    X, y, df = data_loaders[dataset_name]()
    le = LabelEncoder()
    y_clean = le.fit_transform(y)
    df['target'] = y_clean
    
    return df