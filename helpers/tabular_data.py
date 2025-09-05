# changes
# data_loader now returns X, y and df of X with actual column names
# this is needed for ppscore


# used to load all dataset available in 'Dataset' folder
import pandas as pd
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import load_breast_cancer

from functools import partial

# In[37]:


folder_path = '../datasets/tabular/'


# In[38]:

def get_label_idx(labels, debug=False):
    label_dict = {x:i for i,x in enumerate(np.unique(labels))}
    label_idx_finder = np.vectorize(lambda l: label_dict[l])
    
    y = label_idx_finder(labels)

    return y


def load_breast_cancer_data():
    dbName = 'breast_cancer'
    y_column_array = ['target']
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df_clean = df.dropna()
    y_actual = data.target
    
    return df_clean.to_numpy(), y_actual, df_clean


# In[39]:


def load_dermatology_data():
    dbName = 'dermatology' 
    y_column_name = 34
    
    df = pickle.load(open(folder_path + dbName,'rb'))
   
    df = df[(df.values != '?').all(axis=1)]      
    df = df[(df.values != '   ?').all(axis=1)]

    y_actual = df[y_column_name].to_numpy()
    df_clean = df.drop([y_column_name], axis=1)
    
    return df_clean.astype(np.float).to_numpy(), y_actual, df_clean.astype(np.float)


# In[40]:


def load_synthetic_data():
    dbName = 'synthetic'
    y_column_name = 'f'

    # a synthetic gaussian dataset with four clusters
    mean = [[-1, 2, 1, 4, 3], [5, 3, 5, 4, 5], [2,-2, 6, 6, 1], [-7,-3, -3, 7, 4]] 
    cov = [[1, 0, 0, 0, 0],[0,1, 1, 1, 1], [0,1, 0, 1, 0], [1, 0, 1, 0, 1], [1, 1, 0, 0, 1]]
    
    count = 0
    dataframes = []
    for mn in mean:
        a, b, c, d, e = np.random.multivariate_normal(mn, cov, 50).T
        f = np.full(50, count)
        count+=1
        temp_df = pd.DataFrame({'a':a, 'b':b, 'c':c, 'd':d, 'e':e, 'f':f})
        dataframes.append(temp_df)
        
    df = pd.concat(dataframes)
    y_actual = df[y_column_name].to_numpy()
    df_clean = df.drop([y_column_name], axis=1)
        
    
    return df_clean.to_numpy(), y_actual, df_clean


# In[41]:


def load_mice_data():
    dbName = 'mice_data'
    y_column_name = 'class'

    df = pickle.load(open(folder_path + dbName,'rb'))
    df = df.drop(['MouseID','Genotype', 'Treatment', 'Behavior'], axis=1)
    df = df.dropna()
    
    le = LabelEncoder()
    df[y_column_name] = le.fit_transform(df[y_column_name])

    y_actual = df[y_column_name].to_numpy()
    df_clean = df.drop([y_column_name], axis=1)
    
    return df_clean.to_numpy(), y_actual, df_clean


# In[42]:


def load_malware_data():
    dbName = 'malware'
    y_column_name = 'Label'

    df = pd.read_csv(open(folder_path+'TUANDROMD.csv','rb'))
    df = df.dropna()
    
    le = LabelEncoder()
    df[y_column_name] = le.fit_transform(df[y_column_name])

    y_actual = df[y_column_name].to_numpy()
    df_clean = df.drop([y_column_name], axis=1)
    
    return df_clean.to_numpy(), y_actual, df_clean


# In[44]:


def importData (dataset):
    
    df = pd.read_excel(folder_path + dataset + '.xlsx', header=None, engine='openpyxl')
    df_clean = df.dropna()
    
    return df_clean.to_numpy(), df_clean


def get_actual_y (dataset):
    # flat the list of lists to get one list of labels 
    y_actual = []
    for sublist in pd.read_excel(folder_path+ dataset + '_label.xlsx', header=None, engine='openpyxl').values.tolist():
        for item in sublist:     
            y_actual.append(item)
    y_actual = np.array(y_actual)
    
    return y_actual


# one function to load all dataset listed below
dataset_list = ['wine', 
                'ecoli', 
                'olive','vehicle','satellite', 'parkinson', 
                # 'arcene'
               ]

# dataset = 'wine'
# y_actual = get_actual_y(dataset)
# X = importData(dataset = dataset).to_numpy()

def get_feature_and_labels(k):
    X, df_clean = importData(k)
    return X, get_actual_y(k), df_clean


data_loaders = {k: partial(get_feature_and_labels, k) for k in dataset_list}
data_loaders['breast cancer'] = load_breast_cancer_data
data_loaders['dermatology'] = load_dermatology_data
data_loaders['mice'] = load_mice_data
data_loaders['malware'] = load_malware_data
data_loaders['synthetic'] = load_synthetic_data