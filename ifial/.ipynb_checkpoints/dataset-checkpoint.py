import os
import pdb
import math
import random
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, MinMaxScaler, StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold
from missingness.sampler import mar_sampling, mcar_sampling, mnar_sampling
from .impute_and_classify import impute_and_classify
import openml
from loguru import logger
from .corruptor_df import Corruptor
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
import sklearn.neighbors._base
import sys
sys.modules['sklearn.neighbors.base'] = sklearn.neighbors._base


# TODO
# organize teh dataset_config for the load_data API.
# dataset_config = {
# 'dataname': { 'cat':[],'bin':[], 'num':[], 
# 'cols':[]}
# }


OPENML_DATACONFIG = {
    'credit-g': {'bin': ['own_telephone', 'foreign_worker']},
}

X_test_gbt_missing_mask = 0
X_test_missing_mask = 0

def preprocess_dataset_for_missing_values(dataframe):
    # Convert integer columns to float to allow NaN
    for column in dataframe.columns:
        if pd.api.types.is_integer_dtype(dataframe[column]):
            dataframe[column] = dataframe[column].astype(float)
    return dataframe

    
def analyze_missingness(X, data_cut):
    """
    Analyzes missing values in a DataFrame and ranks features based on their missingness.

    Parameters:
    X (pd.DataFrame): The input DataFrame to analyze.
    data_cut (int): The number of features to include in each set.

    Returns:
    feature_sets (list of lists): Sets of features ranked by missingness.
    """

    # Get the rows (samples) that contain missing values
    samples_with_missing_values = X[X.isna().any(axis=1)]

    # Print the samples with missing values
    print("Number of samples with missing values:", len(samples_with_missing_values))

    # Calculate the missingness for each feature
    missing_counts = X.isnull().sum()

    # Check if there are any missing values in the dataset
    if X.isna().any().any():
        print("Dataset has missing values.")
    else:
        print("Dataset has no missing values.")

    # Rank the features based on missingness (ascending order)
    ranked_features = missing_counts.sort_values()

    # Determine the number of features (k) for each set
    ranked_feature_names = ranked_features.index.tolist()
    
    #print('data cut', data_cut)

    # Create sets with top k features
    feature_sets = [ranked_feature_names[i:i + data_cut] for i in range(0, len(ranked_feature_names), data_cut)]

    # Display the ranked features and created sets
    print("\nRanked Features Based on Missingness:")
    print(ranked_features)
    print("\nSets of Top k Features:")
    for i, fset in enumerate(feature_sets, start=1):
        print(f"Set {i}: {fset}")

    return feature_sets


def overlap(feature_sets):
    
    print(len(feature_sets[0]))
    k = len(feature_sets[0])
    print('subset_size: ',k)
    overlap_size = math.ceil(k*0.5)
    step_size = k - overlap_size 

    
    all_features = [feature for feature_set in feature_sets for feature in feature_set]

    overlapped_sets = []
    num_features = len(all_features)
    i = 0
    while i + k <= num_features:
        overlapped_set = all_features[i:i + k]
        overlapped_sets.append(overlapped_set)
        i += step_size

   
    if i < num_features:
        overlapped_set = all_features[-k:]
        if overlapped_set not in overlapped_sets:
            overlapped_sets.append(overlapped_set)

    return overlapped_sets

def load_data(dataname, data, seed, dataset_config=None, encode_cat=False, data_cut=None, missing=0.1, imputation_method=None, method=None):
    '''Load datasets from the local device or from openml.datasets.

    Parameters
    ----------
    dataname: str or int
        the dataset name/index intended to be loaded from openml. or the directory to the local dataset.
    
    dataset_config: dict
        the dataset configuration to specify for loading. Please note that this variable will
        override the configuration loaded from the local files or from the openml.dataset.
    
    encode_cat: bool
        whether encoder the categorical/binary columns to be discrete indices, keep False for TransTab models.
    
    data_cut: int
        how many to split the raw tables into partitions equally; set None will not execute partition.

    seed: int
        the random seed set to ensure the fixed train/val/test split.

    Returns
    -------
    all_list: list or tuple
        the complete dataset, be (x,y) or [(x1,y1),(x2,y2),...].

    train_list: list or tuple
        the train dataset, be (x,y) or [(x1,y1),(x2,y2),...].

    val_list: list or tuple
        the validation dataset, be (x,y) or [(x1,y1),(x2,y2),...].

    test_list: list
        the test dataset, be (x,y) or [(x1,y1),(x2,y2),...].

    cat_col_list: list
        the list of categorical column names.

    num_col_list: list
        the list of numerical column names.

    bin_col_list: list
        the list of binary column names.

    '''
    
    print(method)
    
    if dataset_config is None: dataset_config = OPENML_DATACONFIG
    if isinstance(dataname, str):
        # load a single tabular data
        return load_single_data(dataname=dataname, dataset_config=dataset_config, encode_cat=encode_cat, data_cut=data_cut, seed=seed)
    
    if isinstance(dataname, list):
        # load a list of datasets, combine together and outputs
        num_col_list, cat_col_list, bin_col_list = [], [], []
        all_list = []
        train_list, val_list, test_list = [], [], []
        score = 0
        for dataname_ in dataname:
            data_config = dataset_config.get(dataname_, None)
            if data_cut!=None:
                allset, trainset, valset, testset, cat_cols, num_cols, bin_cols, samples, chunks = \
                    load_single_data(dataname_, data, dataset_config=None, encode_cat=encode_cat, data_cut=data_cut, seed=seed, missing=missing, method=method)
                num_col_list.extend(num_cols)
                cat_col_list.extend(cat_cols)
                bin_col_list.extend(bin_cols)
                all_list.append(allset)
                train_list.append(trainset)
                val_list.append(valset)
                test_list.append(testset)
                
        return all_list, train_list, val_list, test_list, cat_col_list, num_col_list, bin_col_list, samples, chunks

def load_single_data(dataname, data, dataset_config=None, encode_cat=False, data_cut=None, seed=123, missing=0.1, method=None): 
    '''Load tabular dataset from local or from openml public database.
    args:
        dataname: Can either be the data directory on `./data/{dataname}` or the dataname which can be found from the openml database.
        dataset_config: 
            A dict like {'dataname':{'bin': [col1,col2,...]}} to indicate the binary columns for the data obtained from openml.
            Also can be used to {'dataname':{'cols':[col1,col2,..]}} to assign a new set of column names to the data
        encode_cat:  Set `False` if we are using transtab, otherwise we set it True to encode categorical values into indexes.
        data_cut: The number of cuts of the training set. Cut is performed on both rows and columns.
    outputs:
        allset: (X,y) that contains all samples of this dataset
        trainset, valset, testset: the train/val/test split
        num_cols, cat_cols, bin_cols: the list of numerical/categorical/binary column names
    '''
    print('####'*10)
    
    bin_cols = []
    full_train_methods = ['ftt-am']
    imputation_methods = ['ftt-median','ftt-mice']
    proposed_methods = ['ifial']
    if os.path.exists(dataname):
        print(f'load from local data dir {dataname}')
        filename = os.path.join(dataname, 'data_processed.csv')
        df = pd.read_csv(filename, index_col=0)
        y = df['target_label']
        X = df.drop(['target_label'],axis=1)
        all_cols = [col.lower() for col in X.columns.tolist()]

        X.columns = all_cols
        attribute_names = all_cols
        ftfile = os.path.join(dataname, 'numerical_feature.txt')
        if os.path.exists(ftfile):
            with open(ftfile,'r') as f: num_cols = [x.strip().lower() for x in f.readlines()]
        else:
            num_cols = []
        bnfile = os.path.join(dataname, 'binary_feature.txt')
        if os.path.exists(bnfile):
            with open(bnfile,'r') as f: bin_cols = [x.strip().lower() for x in f.readlines()]
        else:
            bin_cols = []
        cat_cols = [col for col in all_cols if col not in num_cols and col not in bin_cols]

        # update cols by loading dataset_config
        if dataset_config is not None:
            if 'columns' in dataset_config:
                new_cols = dataset_config['columns']
                X.columns = new_cols

            if 'bin' in dataset_config:
                bin_cols = dataset_config['bin']
            
            if 'cat' in dataset_config:
                cat_cols = dataset_config['cat']

            if 'num' in dataset_config:
                num_cols = dataset_config['num']
        
    else:
        #dataset = openml.datasets.get_dataset(dataname)
        X,y,categorical_indicator, attribute_names = data.get_data(dataset_format='dataframe', target=data.default_target_attribute)

        X = preprocess_dataset_for_missing_values(X)
        _, X, sampled_indices = mcar_sampling(X, missing, len(X))
        y = y.iloc[sampled_indices]
        
        '''
        if isinstance(dataname, int):
            openml_list = openml.datasets.list_datasets(output_format="dataframe")  # returns a dict
            dataname = openml_list.loc[openml_list.did == dataname].name.values[0]
        else:
            openml_list = openml.datasets.list_datasets(output_format="dataframe")  # returns a dict
            print(f'openml data index: {openml_list.loc[openml_list.name == dataname].index[0]}')
        '''
        print(f'load data from {dataname}')
        print(type(X))

        # drop cols which only have one unique value
        drop_cols = [col for col in attribute_names if X[col].nunique()<=1]

        all_cols = np.array(attribute_names)
        categorical_indicator = np.array(categorical_indicator)
        cat_cols = [col for col in all_cols[categorical_indicator] if col not in drop_cols]
        num_cols = [col for col in all_cols[~categorical_indicator] if col not in drop_cols]
        all_cols = [col for col in all_cols if col not in drop_cols]
        
        if dataset_config is not None:
            if 'bin' in dataset_config: bin_cols = [c for c in cat_cols if c in dataset_config['bin']]
        else: bin_cols = []
        cat_cols = [c for c in cat_cols if c not in bin_cols]

        # encode target label
        y = LabelEncoder().fit_transform(y.values)
        y = pd.Series(y,index=X.index)
        
    
    X = X[bin_cols + num_cols + cat_cols]

    # rename column names if is given
    if dataset_config is not None:
        data_config = dataset_config
        if 'columns' in data_config:
            new_cols = data_config['columns']
            X.columns = new_cols
            attribute_names = new_cols

        if 'bin' in data_config:
            bin_cols = data_config['bin']
        
        if 'cat' in data_config:
            cat_cols = data_config['cat']

        if 'num' in data_config:
            num_cols = data_config['num']


    # split train/val/test
    data_split_idx = None
    if dataset_config is not None:
        data_split_idx = dataset_config.get('data_split_idx', None)

    if data_split_idx is not None:
        train_idx = data_split_idx.get('train', None)
        val_idx = data_split_idx.get('val', None)
        test_idx = data_split_idx.get('test', None)

        if train_idx is None or test_idx is None:
            raise ValueError('train/test split indices must be provided together')
    
        else:
            train_dataset = X.iloc[train_idx]
            y_train = y[train_idx]
            test_dataset = X.iloc[test_idx]
            y_test = y[test_idx]
            if val_idx is not None:
                val_dataset = X.iloc[val_idx]
                y_val = y[val_idx]
            else:
                val_dataset = None
                y_val = None
    else:
        
        # split train/val/test
        print("No initial split")
        #train_dataset, test_dataset, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y, shuffle=True)
        splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        indices = np.arange(X.shape[0])
        print('For fold: ', seed)
        splits = list(splitter.split(indices, y))
        train_indices, test_indices = splits[seed]
        train_indices, valid_indices = train_test_split(train_indices, test_size=1/8, stratify=y[train_indices], random_state=42)
        
        train_dataset = X.iloc[train_indices]
        y_train = y.iloc[train_indices]
        val_dataset = X.iloc[valid_indices]
        y_val = y.iloc[valid_indices]
        test_dataset= X.iloc[test_indices]
        y_test = y.iloc[test_indices]
        
        
        if method in imputation_methods:
            if(len(num_cols)>0):
                if method=='ftt-median':  
                    print('Doing median imputation.')
                    num_imputer = SimpleImputer(strategy='median')
                elif method=='ftt-mice': 
                    print('Doing mice imputation.')
                    num_imputer = IterativeImputer(random_state=0)

                    
                
               
                num_imputer.fit(train_dataset[num_cols])

                train_dataset[num_cols] = num_imputer.transform(train_dataset[num_cols])
                val_dataset[num_cols] = num_imputer.transform(val_dataset[num_cols])
                test_dataset[num_cols] = num_imputer.transform(test_dataset[num_cols])

                scaler = MinMaxScaler().fit(train_dataset[num_cols])

                train_dataset[num_cols] = scaler.transform(train_dataset[num_cols])
                val_dataset[num_cols] = scaler.transform(val_dataset[num_cols])
                test_dataset[num_cols] = scaler.transform(test_dataset[num_cols])#

            if(len(cat_cols)>0):
                cat_imputer = SimpleImputer(strategy='most_frequent')
                cat_imputer.fit(train_dataset[cat_cols])

                train_dataset[cat_cols] = cat_imputer.transform(train_dataset[cat_cols])
                val_dataset[cat_cols] = cat_imputer.transform(val_dataset[cat_cols])
                test_dataset[cat_cols] = cat_imputer.transform(test_dataset[cat_cols]) 
            
            return (X,y), (train_dataset,y_train), (val_dataset,y_val), (test_dataset, y_test), cat_cols, num_cols, bin_cols, len(train_dataset), 0
                
        elif method in full_train_methods:
            print('Running on whole dataset without doing chunking.')
            return (X,y), (train_dataset,y_train), (val_dataset,y_val), (test_dataset, y_test), cat_cols, num_cols, bin_cols, len(train_dataset), 0
            
        
        elif method in proposed_methods:
            scaler = MinMaxScaler().fit(train_dataset[num_cols])

            train_dataset[num_cols] = scaler.transform(train_dataset[num_cols])
            val_dataset[num_cols] = scaler.transform(val_dataset[num_cols])
            test_dataset[num_cols] = scaler.transform(test_dataset[num_cols])
            
            
            tmp_y_val = y_val.copy()


            print('Generating feature sets for Feature Incremental Learning: ')


            train_subset_list = []
            val_subset_list = []
            test_subset_list = []

            # Initialize the lists to store information about each chunk
            chunk_numbers = []
            chunk_features_list = []
            overlap_features_list = []
            train_samples_list = []
            val_samples_list = []

            # Initialize the lists to store subsets for training and validation
            train_subset_list = []
            val_subset_list = []

            miss_rate = train_dataset.isna().sum().sum() / train_dataset.size * 100
            print(f"Missing rate in test samples: {miss_rate:.2f}%")
            miss_rate = test_dataset.isna().sum().sum() / test_dataset.size * 100
            print(f"Missing rate in test samples: {miss_rate:.2f}%")

            print("Number of samples with missing values in trainset:", train_dataset.isna().any(axis=1).sum())
            print("Number of samples with missing values in testset:", test_dataset.isna().any(axis=1).sum())

            # Initialize an empty list to keep track of the growing feature set
            cumulative_features = []
            overlapping_features = []
            samples = 0
            chunks = 0
            overlap_ratio = 0.5

            feature_sets = analyze_missingness(train_dataset,data_cut)
            cumulative_features = overlap(feature_sets)


            # Iterate over each feature set to create the growing subsets
            for i in range(len(cumulative_features)):

                tmp_y_train = y_train.copy()
                tmp_y_val = y_val.copy()


                # Create a subset of the dataset using the cumulative features
                print(f"Subset {i+1} includes features: ", cumulative_features[i])
                
                if method=='chunking+attention-mask':
                    print(f'Running {method} method.')
                    train_subset = train_dataset[cumulative_features[i]]#.dropna()
                    val_subset = val_dataset[cumulative_features[i]]#.dropna()
                else:
                    print(f'Running {method} method.')
                    train_subset = train_dataset[cumulative_features[i]].dropna()
                    val_subset = val_dataset[cumulative_features[i]].dropna()


                chunks+=1

                # Store chunk information
                chunk_numbers.append(i+1)
                chunk_features_list.append(cumulative_features[i])  # Features in the current chunk

                train_size = len(train_subset)
                val_size = len(val_subset)
                min_val_size = 2


                if train_subset.empty or len(train_subset)<=3:
                        chunks -= 1
                        print(f"Warning: Subset {i+1} contains no samples after dropping missing values. Skipping this subset.")
                        continue
                elif val_subset.empty or len(val_subset)<=1:

                        val_subset = train_subset.sample(n=min_val_size, random_state=42)
                        train_subset = train_subset.drop(index=val_subset.index)
                
                        tmp_y_val = pd.concat([tmp_y_val, tmp_y_train.loc[val_subset.index]])
                        tmp_y_train = tmp_y_train.drop(index=val_subset.index)
                        print(f"Validation subset {i+1} was empty. Moved {min_val_size} samples from the training set.")


                        train_subset = train_subset.reset_index(drop=True)
                        tmp_y_train = tmp_y_train.reset_index(drop=True)

                        val_subset = val_subset.reset_index(drop=True)
                        tmp_y_val = tmp_y_val.reset_index(drop=True)

                # Print the number of samples in each subset
                print(f"Number of samples in Train subset {i+1}: {len(train_subset)}")
                print(f"Number of samples in Validation subset {i+1}: {len(val_subset)}")

                # Store the number of samples
                train_samples_list.append(train_size)
                val_samples_list.append(val_size)

                samples+=len(train_subset)

                # Append the subset of features and corresponding target values to the lists
                train_subset_list.append(
                    (train_subset, tmp_y_train.loc[train_subset.index])
                )
                val_subset_list.append(
                    (val_subset, tmp_y_val.loc[val_subset.index])
                )

            # Print dataset information
            print('# data: {}, # feat: {}, # cate: {}, # bin: {}, # numerical: {}, pos rate: {:.2f}'.format(
                len(X), len(attribute_names), len(cat_cols), len(bin_cols), len(num_cols), (y == 1).sum() / len(y)))

            # Return the subsets and column information
            return (X, y), train_subset_list, val_subset_list, (test_dataset, y_test), cat_cols, num_cols, bin_cols, samples, chunks 
        
        else:
            print("Method name is invalid.")
        