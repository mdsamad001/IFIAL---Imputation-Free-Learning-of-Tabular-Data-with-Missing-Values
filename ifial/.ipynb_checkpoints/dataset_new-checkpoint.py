import os
import pdb
import random
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from missingness.sampler import mar_sampling, mcar_sampling, mnar_sampling
from .impute_and_classify import impute_and_classify
import openml
from loguru import logger
from .corruptor_df import Corruptor
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score

# TODO
# organize teh dataset_config for the load_data API.
# dataset_config = {
# 'dataname': { 'cat':[],'bin':[], 'num':[], 
# 'cols':[]}
# }


OPENML_DATACONFIG = {
    'credit-g': {'bin': ['own_telephone', 'foreign_worker'],}
}

EXAMPLE_DATACONFIG = {
    "example": {
        "bin": ["bin1", "bin2"],
        "cat": ["cat1", "cat2"],
        "num": ["num1", "num2"],
        "cols": ["bin1", "bin2", "cat1", "cat2", "num1", "num2"],
        "binary_indicator": ["1", "yes", "true", "positive", "t", "y"],
        "data_split_idx": {
            "train":[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            "val":[10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
            "test":[20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
        }
    }
}

X_test_gbt_missing_mask = 0
X_test_missing_mask = 0

def gbt_pipeline(dataname, missing, encode_cat=True, seed=123, dataset_config=None):
    
    print('gbt')
    # Step 1: Load the dataset
    dataset = openml.datasets.get_dataset(dataname)
    X, y, categorical_indicator, attribute_names = dataset.get_data(
        dataset_format='dataframe', target=dataset.default_target_attribute)
    
    # Step 2: Apply MCAR sampling
    _, X, sampled_indices = mcar_sampling(X, missing, len(X))
    y = y.iloc[sampled_indices]
    
    # Step 3: Process features
    # Drop columns with only one unique value
    drop_cols = [col for col in attribute_names if X[col].nunique() <= 1]
    
    all_cols = np.array(attribute_names)
    categorical_indicator = np.array(categorical_indicator)
    cat_cols = [col for col in all_cols[categorical_indicator] if col not in drop_cols]
    num_cols = [col for col in all_cols[~categorical_indicator] if col not in drop_cols]
    all_cols = [col for col in all_cols if col not in drop_cols]
    
    if dataset_config is not None:
        bin_cols = [c for c in cat_cols if c in dataset_config.get('bin', [])]
    else:
        bin_cols = []
    
    cat_cols = [c for c in cat_cols if c not in bin_cols]
    
    # Encode target label
    y = LabelEncoder().fit_transform(y.values)
    y = pd.Series(y, index=X.index)

    # Step 4: Process numerical and categorical columns
    if len(num_cols) > 0:
        X[num_cols] = MinMaxScaler().fit_transform(X[num_cols])

    if len(cat_cols) > 0:
        if encode_cat:
            X[cat_cols] = OrdinalEncoder().fit_transform(X[cat_cols])
        else:
            X[cat_cols] = X[cat_cols].astype(str).replace("nan", np.nan)

    if len(bin_cols) > 0:
        if 'binary_indicator' in dataset_config:
            X[bin_cols] = X[bin_cols].astype(str).replace("nan", np.nan).map(
                lambda x: 1 if x.lower() in dataset_config['binary_indicator'] else 0).values
        else:
            X[bin_cols] = X[bin_cols].astype(str).replace("nan", np.nan).map(
                lambda x: 1 if isinstance(x, str) and x.lower() in ['yes', 'true', '1', 't'] else 0).values

        # Check for non-binary values in binary columns
        if (~X[bin_cols].isin([0, 1])).any().any():
            raise ValueError(f'binary columns {bin_cols} contains values other than 0/1.')
    
    X = X[bin_cols + num_cols + cat_cols]

    # Rename column names if provided in dataset_config
    if dataset_config is not None:
        if 'columns' in dataset_config:
            X.columns = dataset_config['columns']
            attribute_names = dataset_config['columns']
        # Step 5: Perform train-test split
    train_dataset, test_dataset, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y, shuffle=True)
    
    #_, train_dataset, sampled_indices = mcar_sampling(train_dataset, missing, len(train_dataset))
    #y_train = y_train.iloc[sampled_indices]
    
    #_, test_dataset, sampled_indices = mcar_sampling(test_dataset, missing, len(test_dataset))
    #y_test = y_test.iloc[sampled_indices]
    
    #X_test_gbt_missing_mask = test_dataset
    #print('X_test_gbt_missing_mask in gbt', len(X_test_gbt_missing_mask))
    
    
    if len(num_cols) > 0:
        num_imputer = SimpleImputer(strategy='median')
        #num_imputer = IterativeImputer(max_iter=10,random_state=0)
        train_dataset[num_cols] = num_imputer.fit_transform(train_dataset[num_cols])
        #test_dataset[num_cols] = num_imputer.transform(test_dataset[num_cols])
        #for col in num_cols: test_dataset[col].fillna(0, inplace=True)
        # Compute medians from the training dataset
        test_dataset[num_cols] = num_imputer.transform(test_dataset[num_cols])
        #medians = test_dataset[num_cols].median()
        # Apply the same median values to the test dataset
        #test_dataset[num_cols] = test_dataset[num_cols].fillna(medians)


    # Impute categorical columns with mode
    if len(cat_cols) > 0:
        cat_imputer = SimpleImputer(strategy='median')
        #cat_imputer = IterativeImputer(max_iter=10,random_state=0) #SimpleImputer(strategy='most_frequent')
        train_dataset[cat_cols] = cat_imputer.fit_transform(train_dataset[cat_cols])
        test_dataset[cat_cols] = cat_imputer.transform(test_dataset[cat_cols])
        #for col in cat_cols: test_dataset[col].fillna(0, inplace=True)
        # Compute modes from the training dataset
        #modes = test_dataset[cat_cols].mode().iloc[0]

        # Apply the same mode values to the test dataset
        #test_dataset[cat_cols] = test_dataset[cat_cols].fillna(modes)

    # Impute binary columns with mode
    if len(bin_cols) > 0:
            bin_imputer = SimpleImputer(strategy='median')
            #bin_imputer = IterativeImputer(max_iter=10,random_state=0) #SimpleImputer(strategy='most_frequent')
            train_dataset[bin_cols] = bin_imputer.fit_transform(train_dataset[bin_cols])
            test_dataset[bin_cols] = bin_imputer.transform(test_dataset[bin_cols])
        #for col in bin_cols: test_dataset[col].fillna(0, inplace=True)
            # Compute modes from the training dataset
            #bin_modes = train_dataset[bin_cols].mode().iloc[0]
            # Apply the same mode values to the test dataset
            #test_dataset[bin_cols] = test_dataset[bin_cols].fillna(bin_modes)
        
    #print(test_dataset.isnull().sum())
    #print(test_dataset)


    # Step 6: Train Gradient Boosting Classifier
    model = GradientBoostingClassifier(random_state=seed)
    model.fit(train_dataset, y_train)

    # Step 7: Evaluate AUC score
    y_pred_prob = model.predict_proba(test_dataset)[:, 1]
    auc_score = roc_auc_score(y_test, y_pred_prob)
    #print(f"AUC score: {auc_score}")
    
    return model, auc_score

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

    # Create sets with top k features
    feature_sets = [ranked_feature_names[i:i + data_cut] for i in range(0, len(ranked_feature_names), data_cut)]

    # Display the ranked features and created sets
    print("\nRanked Features Based on Missingness:")
    print(ranked_features)
    print("\nSets of Top k Features:")
    for i, fset in enumerate(feature_sets, start=1):
        print(f"Set {i}: {fset}")

    return feature_sets

def load_data(dataname, data, dataset_config=None, encode_cat=False, data_cut=None, seed=123, missing=0.1):
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
            #_,score = gbt_pipeline(dataname_, missing, encode_cat=True, seed=123, dataset_config=None)
            #print('GBT score', score)
            if data_cut!=None:
                allset, trainset, valset, testset, cat_cols, num_cols, bin_cols, samples, chunks = \
                    load_single_data(dataname_, data, dataset_config=data_config, encode_cat=encode_cat, data_cut=data_cut, seed=seed,missing=missing)
                num_col_list.extend(num_cols)
                cat_col_list.extend(cat_cols)
                bin_col_list.extend(bin_cols)
                all_list.append(allset)
                train_list.append(trainset)
                val_list.append(valset)
                test_list.append(testset)
        if data_cut==None:
            _,score = gbt_pipeline(dataname_, missing, encode_cat=True, seed=123, dataset_config=None)
            return score
        return all_list, train_list, val_list, test_list, cat_col_list, num_col_list, bin_col_list, samples, chunks

def load_single_data(dataname, data, dataset_config=None, encode_cat=False, data_cut=None, seed=123,missing=0.1):
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
        
        
        #_,X = mcar_sampling(X,0.1,len(X))
        
        '''
        #New Change -
        # Get the rows (samples) that contain missing values
        samples = X[X.isna().any(axis=1)]

        # Print the samples with missing values
        #print("Samples with missing values:")
        #print(samples)
        # Step 1: Calculate the missingness for each feature
        missing_counts = X.isnull().sum()

        if X.isna().any().any():
            print(f"Dataset has missing values.")
        else:
            print(f"Dataset has no missing values.")

        # Step 2: Rank the features based on missingness (ascending order)
        ranked_features = missing_counts.sort_values()

        # Step 3: Determine the number of features (k) for each set
        k = data_cut
        ranked_feature_names = ranked_features.index.tolist()

        # Step 4: Create sets with top k features
        feature_sets = [ranked_feature_names[i:i + k] for i in range(0, len(ranked_feature_names), k)]

        # Display the ranked features and created sets
        print("Ranked Features Based on Missingness:")
        print(ranked_features)
        print("\nSets of Top k Features:")
        for i, fset in enumerate(feature_sets, start=1):
               print(f"Set {i}: {fset}")
        '''
    # start processing features
    # process num
    if len(num_cols) > 0:
        #for col in num_cols: X[col].fillna(X[col].mode()[0], inplace=True)
        X[num_cols] = MinMaxScaler().fit_transform(X[num_cols])

    if len(cat_cols) > 0:
        #for col in cat_cols: X[col].fillna(X[col].mode()[0], inplace=True)
        # process cate
        if encode_cat:
            X[cat_cols] = OrdinalEncoder().fit_transform(X[cat_cols])
        else:
            X[cat_cols] = X[cat_cols].astype(str).replace("nan", np.nan)  # Replace 'nan' strings with np.nan

    if len(bin_cols) > 0:
        #for col in bin_cols: X[col].fillna(X[col].mode()[0], inplace=True)
        if 'binary_indicator' in dataset_config:
            X[bin_cols] = X[bin_cols].astype(str).replace("nan", np.nan).map(lambda x: 1 if x.lower() in dataset_config['binary_indicator'] else 0).values
        else:
            X[bin_cols] = X[bin_cols].astype(str).replace("nan", np.nan).map(lambda x: 1 if isinstance(x, str) and x.lower() in ['yes', 'true', '1', 't'] 
                                                                             else 0 ).values       
        
        # if no dataset_config given, keep its original format
        # raise warning if there is not only 0/1 in the binary columns
        if (~X[bin_cols].isin([0,1])).any().any():
            raise ValueError(f'binary columns {bin_cols} contains values other than 0/1.')

    
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
        train_dataset, test_dataset, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y, shuffle=True)
        X_test_missing_mask = test_dataset
        '''
        # Use .equals() to compare two DataFrames
        if X_test_missing_mask.equals(X_test_gbt_missing_mask):
            print('X_test_missing_mask : ', len(X_test_missing_mask))
            print('X_test_gbt_missing_mask: ', len(X_test_gbt_missing_mask))
            print('Equal')
        else:
            print('X_test_missing_mask : ', len(X_test_missing_mask))
            print('X_test_gbt_missing_mask: ', len(X_test_gbt_missing_mask))
            print('Not equal')
        '''
        #_,train_dataset = mcar_sampling(train_dataset,0.1,len(train_dataset))
        # Apply MCAR sampling only to the training data
        # Apply MCAR sampling only to the training data
        #_, train_dataset, sampled_indices = mcar_sampling(train_dataset, missing, len(train_dataset))
        #_, test_dataset, test_sampled_indices = mcar_sampling(test_dataset, missing, len(test_dataset))

        # Align the y_train indices with the modified train_dataset
        #y_train = y_train.iloc[sampled_indices]
        #y_test = y_test.iloc[test_sampled_indices]

        # Create validation dataset from training dataset
        val_size = int(len(y_train) * 0.1)
        val_dataset = train_dataset.iloc[-val_size:]
        y_val = y_train.iloc[-val_size:]
        train_dataset = train_dataset.iloc[:-val_size]
        y_train = y_train.iloc[:-val_size]
        
        tmp_y_train = y_train.copy()
        tmp_y_val = y_val.copy()
        
        #_, test_auc = impute_and_classify(train_dataset, test_dataset, y_train, y_test, val_dataset, y_val)
        #print("GBT test_auc", test_auc)
        
    if data_cut == 0:
        print("All cols of the dataset : ", all_cols)
        np.random.shuffle(all_cols)
        sp_size=int(len(all_cols)/data_cut)
        col_splits = np.split(all_cols, range(0,len(all_cols),sp_size))[1:]
        new_col_splits = []
        for split in col_splits:
            print("Split : ", split)
            candidate_cols = np.random.choice(np.setdiff1d(all_cols, split), int(sp_size*overlap), replace=False)
            print("Candidate Cols : ", candidate_cols)
            new_col_splits.append(split.tolist() + candidate_cols.tolist())
            print("New splits after Overlapping: ", new_col_splits)
        if len(col_splits) > data_cut:
            for i in range(len(col_splits[-1])):
                new_col_splits[i] += [col_splits[-1][i]]
                new_col_splits[i] = np.unique(new_col_splits[i]).tolist()
            new_col_splits = new_col_splits[:-1]
        
        print("New splits after overlapping and adjusting partitions: ", new_col_splits)

        # cut subset
        trainset_splits = np.array_split(train_dataset, data_cut)
        testset_splits = np.array_split(test_dataset, data_cut)
        valset_splits = np.array_split(val_dataset, data_cut)
        train_subset_list = []
        for i in range(data_cut):
            train_subset_list.append(
                (trainset_splits[i][new_col_splits[i]], y_train.loc[trainset_splits[i].index])
            )
        val_subset_list = []
        for i in range(data_cut):
            val_subset_list.append(
                (valset_splits[i][new_col_splits[i]], y_val.loc[valset_splits[i].index])
            )
        test_subset_list = []
        for i in range(data_cut):
            test_subset_list.append(
                (testset_splits[i][new_col_splits[i]], y_test.loc[testset_splits[i].index])
            )
        print('# data: {}, # feat: {}, # cate: {},  # bin: {}, # numerical: {}, pos rate: {:.2f}'.format(len(X), len(attribute_names), len(cat_cols), len(bin_cols), len(num_cols), (y==1).sum()/len(y)))
        return (X, y), train_subset_list, val_subset_list, test_subset_list, cat_cols, num_cols, bin_cols

    #else:
    #    print('# data: {}, # feat: {}, # cate: {},  # bin: {}, # numerical: {}, pos rate: {:.2f}'.format(len(X), len(attribute_names), len(cat_cols), len(bin_cols), len(num_cols), (y==1).sum()/len(y)))
    #    return (X,y), (train_dataset,y_train), (val_dataset,y_val), (test_dataset, y_test), cat_cols, num_cols, bin_cols

    #Feature Incremental Learning 
    elif data_cut >= 2:
        print('Generating feature sets for Feature Incremental Learning: ')
        #v1, v2, v3 = [],[],[]
        #np.random.shuffle(all_cols)

        '''
        v1_size = len(all_cols) // 3
        v2_size = v1_size
        v3_size = len(all_cols) - v1_size - v2_size  # Remaining columns

        
        v1 = all_cols[:v1_size]
        v2 = all_cols[v1_size:v1_size + v2_size]
        v3 = all_cols[v1_size + v2_size:]

       
        set1_cols = v1
        set2_cols = np.concatenate([v1, v2])
        set3_cols = np.concatenate([v1, v2, v3])
        
        print('set 1: ', set1_cols)
        print('set 2: ', set2_cols)
        print('set 3: ', set3_cols)
        
        
        # Step 1: Calculate the missingness for each feature
        missing_counts = np.sum(train_dataset == b'?', axis=0)  # Assuming missing values are represented by '?' in ARFF files
        
        # Step 2: Rank the features based on missingness (ascending order)
        ranked_indices = np.argsort(missing_counts)
        
        # Step 3: Determine the number of features (k) for each set
        k = 4
        ranked_feature_names = np.array(all_cols)[ranked_indices]
        
        # Step 4: Create sets with top k features
        feature_sets = [ranked_feature_names[i:i + k].tolist() for i in range(0, len(ranked_feature_names), k)]
        
        # Display the ranked features and created sets
        print("Ranked Features Based on Missingness:")
        for feature, count in zip(ranked_feature_names, missing_counts[ranked_indices]):
            print(f"{feature}: {count}")
            
        print("\nSets of Top k Features:")
        for i, fset in enumerate(feature_sets, start=1):
            print(f"Set {i}: {fset}")
        '''
        #trainset_splits = np.array_split(train_dataset, data_cut)
        train_subset_list = []
        val_subset_list = []
        test_subset_list = []
        '''
        # Assign column subsets to the data splits
        for i in range(data_cut):
        #    if i % 3 == 0:
        #        subset_cols = set1_cols
        #    elif i % 3 == 1:
        #        subset_cols = set2_cols
        #    else:
        #        subset_cols = set3_cols
            
            #if i % 3 == 2:
            train_subset_list.append(
                    #(trainset_splits[i][subset_cols], y_train.loc[trainset_splits[i].index])
                    (train_dataset[feature_sets[i]], y_train.loc[train_dataset.index])
                )
        '''
        '''
        print("Number of samples with missing values in testset", test_dataset.isna().any(axis=1).sum())
        for i in range(len(feature_sets)):
            # Create a subset of the dataset with the selected feature columns
            #print(f"Set {i+1}: ", feature_sets[i])
            subset_cols = feature_sets[i]
            train_subset = train_dataset[subset_cols].dropna()
            val_subset = val_dataset[subset_cols].dropna()
            #test_subset = test_dataset[subset_cols]
            

            if train_subset.isna().any().any():
                print(f"Train subset {i+1} has missing values.")
            else:
                print(f"Train subset {i+1} has no missing values.")
            # Append the tuple (subset of features, corresponding target values) to the list
            train_subset_list.append(
                (train_subset, y_train.loc[train_subset.index])
            )
                    
            val_subset_list.append(
                (val_subset, y_val.loc[val_subset.index])
            )
       
            #test_subset_list.append(
            #    (test_subset, y_test.loc[test_subset.index])
            #)
        #print('train_subset_list: ', train_subset_list)
        print('# data: {}, # feat: {}, # cate: {},  # bin: {}, # numerical: {}, pos rate: {:.2f}'.format(
            len(X), len(attribute_names), len(cat_cols), len(bin_cols), len(num_cols), (y == 1).sum() / len(y)))

        return (X, y), train_subset_list, val_subset_list, (test_dataset, y_test), cat_cols, num_cols, bin_cols
        '''
        # Initialize the lists to store information about each chunk
        chunk_numbers = []
        chunk_features_list = []
        overlap_features_list = []
        train_samples_list = []
        val_samples_list = []
        
        # Initialize the lists to store subsets for training and validation
        train_subset_list = []
        val_subset_list = []

        # Print the number of samples with missing values in the test dataset
        #print(len(train_dataset))
        #print(len(test_dataset))
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
        
        # Iterate over each feature set to create the growing subsets
        for i in range(len(feature_sets)):
            tmp_y_train = y_train.copy()
            tmp_y_val = y_val.copy()
            # Combine the current set of features with all previous ones
            #cumulative_features.extend(feature_sets[i])  # Append current features to the cumulative list
            if i == 0:
                # For the first feature set, just use the entire set
                cumulative_features = feature_sets[i]
                overlap_features_list.append('None')
            else:
                # Calculate the number of overlapping features from the old cumulative set
                overlap_count = int(len(cumulative_features) * overlap_ratio)

                # Select the first 'overlap_count' features from the old cumulative set
                overlapping_features = random.sample(cumulative_features, overlap_count) #cumulative_features[:overlap_count]
                
                print('Now overlapping')
                print('Overlapped Features: ',overlapping_features) 

                # Combine overlapping features with the current new feature set
                cumulative_features = overlapping_features + feature_sets[i]
            # Create a subset of the dataset using the cumulative features
            print(f"Subset {i+1} includes features: ", cumulative_features)
            train_subset = train_dataset[cumulative_features].dropna()
            val_subset = val_dataset[cumulative_features].dropna()

            #print("Number of samples with missing values in train subset: ", (train_subset == 'NaN').any(axis=1).sum())
            #print("Number of samples with missing values in test subset:", (train_subset == 'NaN').any(axis=1).sum())
            chunks+=1

            # Store chunk information
            chunk_numbers.append(i+1)
            chunk_features_list.append(cumulative_features)  # Features in the current chunk
            if len(overlapping_features)!=0:
                overlap_features_list.append(overlapping_features)  # Overlapping features
            # Print a message indicating if there are missing values in the training subset
            #if train_subset.isna().any().any():
            #    print(f"Train subset {i+1} has missing values.")
            #else:
            #    print(f"Train subset {i+1} has no missing values.")
            
                    # Ensure validation subset has at least 10% of training subset samples
            train_size = len(train_subset)
            val_size = len(val_subset)
            min_val_size = 2
            
            # Check if there are any samples left after dropping missing values
            #if train_subset.empty or val_subset.empty: #or val_size < min_val_size:
            #    chunks-=1
            #    print(f"Warning: Subset {i+1} contains no samples after dropping missing values. Skipping this subset.")
            #    continue
            #if train_subset.empty or val_subset.empty or len(val_subset)<=1:
            if train_subset.empty or len(train_subset)<=3:
                    chunks -= 1
                    print(f"Warning: Subset {i+1} contains no samples after dropping missing values. Skipping this subset.")
                    continue
            elif val_subset.empty or len(val_subset)<=1:
                    # If the validation set is empty, move some samples from the training set
                    val_subset = train_subset.sample(n=min_val_size, random_state=42)
                    train_subset = train_subset.drop(index=val_subset.index)
                    # Move the corresponding labels from y_train to y_val
                    tmp_y_val = pd.concat([tmp_y_val, tmp_y_train.loc[val_subset.index]])
                    tmp_y_train = tmp_y_train.drop(index=val_subset.index)
                    print(f"Validation subset {i+1} was empty. Moved {min_val_size} samples from the training set.")
            #if val_size < min_val_size:
            #    continue
            #    print(f"Validation subset {i+1} has fewer samples than 10% of training subset.")
        
                # Resample validation set from the training set if needed
                #val_subset = train_subset.sample(n=min_val_size, random_state=seed)
                # Align labels with the current train and validation subsets
            # After modifying train_subset and y_train
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
        '''
        chunk_info_df = pd.DataFrame({
        "Chunks": chunk_numbers,
        "Features": chunk_features_list,
        "Overlapped Features": overlap_features_list,
        "Samples(Train Subset)": train_samples_list,
        "Samples(Val Subset)": val_samples_list
        })
        print("Number of samples: ", samples)
        print("Number of chunks: ", chunks)
        '''
        # Print dataset information
        print('# data: {}, # feat: {}, # cate: {}, # bin: {}, # numerical: {}, pos rate: {:.2f}'.format(
            len(X), len(attribute_names), len(cat_cols), len(bin_cols), len(num_cols), (y == 1).sum() / len(y)))

        # Return the subsets and column information
        return (X, y), train_subset_list, val_subset_list, (test_dataset, y_test), cat_cols, num_cols, bin_cols, samples, chunks 


    else:
        print('# data: {}, # feat: {}, # cate: {},  # bin: {}, # numerical: {}, pos rate: {:.2f}'.format(
            len(X), len(attribute_names), len(cat_cols), len(bin_cols), len(num_cols), (y == 1).sum() / len(y)))
        return (X, y), (train_dataset, y_train), (val_dataset, y_val), (test_dataset, y_test), cat_cols, num_cols, bin_cols
