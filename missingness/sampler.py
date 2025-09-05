import numpy as np
import pandas as pd
import math
import random
import torch


from missingness.utils import sample_batch_index, binary_sampler
from scipy import optimize

def random_sampling(dataframe, no_of_samples):
    no, dim = dataframe.shape

    if no < no_of_samples:
        no_of_samples = no

    data_x = dataframe.values.astype(np.float32)
    sample_idx = sample_batch_index(no, no_of_samples)
    data_x_i = data_x[sample_idx, :]

    actual_dataframe = pd.DataFrame(
        data=data_x_i[0:,0:],
        index=[i for i in range(data_x_i.shape[0])],
        columns=dataframe.columns
        )
    return actual_dataframe



import numpy as np
import pandas as pd
import torch
from scipy import optimize

# def pick_coeffs(X, idxs_obs=None, idxs_nas=None, self_mask=False):
#     n, d = X.shape
#     if self_mask:
#         coeffs = torch.randn(d)
#         Wx = X * coeffs
#         coeffs /= torch.std(Wx, 0)
#     else:
#         d_obs = len(idxs_obs)
#         d_na = len(idxs_nas)
#         coeffs = torch.randn(d_obs, d_na, dtype=X.dtype)
#         Wx = X[:, idxs_obs].mm(coeffs)
#         coeffs /= torch.std(Wx, 0, keepdim=True)
#     return coeffs

# def fit_intercepts(X, coeffs, p, self_mask=False):
#     if self_mask:
#         d = len(coeffs)
#         intercepts = torch.zeros(d)
#         for j in range(d):
#             def f(x):
#                 return torch.sigmoid(X * coeffs[j] + x).mean().item() - p
#             intercepts[j] = optimize.bisect(f, -50, 50)
#     else:
#         d_obs, d_na = coeffs.shape
#         intercepts = torch.zeros(d_na)
#         for j in range(d_na):
#             def f(x):
#                 return torch.sigmoid(X.mv(coeffs[:, j]) + x).mean().item() - p
#             intercepts[j] = optimize.bisect(f, -50, 50)
#     return intercepts

def mnar_sampling_logistic(dataframe, miss_rate, no_of_samples=10, p_params=0.3, seed=0, x_len = None):
    """
    MNAR sampling using a logistic model.
    
    Args:
        dataframe (pd.DataFrame): Input data (all numeric).
        miss_rate (float): Target missing rate.
        no_of_samples (int or None): Number of rows to sample (if None, use all rows).
        p_params (float): Proportion of features used as inputs.
        seed (int): Random seed.
        
    Returns:
        actual_dataframe (pd.DataFrame): The original (sampled) data.
        missing_dataframe (pd.DataFrame): Data with MNAR missing values induced.
        sample_idx (array-like): The indices of the sampled rows.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Optionally adjust p_params based on miss_rate
    if miss_rate > 0.3:
        p_params = 0.1
    else:
        p_params = 0.3

    # Sample rows if no_of_samples is provided
    if no_of_samples is not None:
        no, d = dataframe.shape
        sample_idx = np.random.choice(no, min(no_of_samples, no), replace=False)
        df_sampled = dataframe.iloc[sample_idx].reset_index(drop=True)
    else:
        df_sampled = dataframe.copy()
        sample_idx = dataframe.index

    # Convert sampled data to a torch tensor (assumes data is float32)
    X = torch.tensor(df_sampled.values, dtype=torch.float32)
    n, d = X.shape

    # Select p_params fraction of columns to be used as logistic inputs
    d_params = max(int(p_params * d), 1)
    idxs_params = np.random.choice(d, d_params, replace=False)
    idxs_nas = np.array([i for i in range(d) if i not in idxs_params])
    
    # Generate logistic coefficients and intercepts to achieve target missing rate
    coeffs = pick_coeffs(X, idxs_obs=idxs_params, idxs_nas=idxs_nas)
    intercepts = fit_intercepts(X[:, idxs_params], coeffs, p=miss_rate)
    
    # Compute missing probabilities for the non-input features
    ps = torch.sigmoid(X[:, idxs_params].mm(coeffs) + intercepts)
    ber = torch.rand(n, len(idxs_nas))
    mask = torch.zeros(n, d).bool()
    mask[:, idxs_nas] = ber < ps
    
    # Optionally, also mask the logistic input features with MCAR missingness at rate miss_rate
    mask[:, idxs_params] = torch.rand(n, d_params) < miss_rate

    # Apply mask to data (set missing entries to NaN)
    X_missing = X.clone()
    X_missing[mask] = float('nan')
    
    # Convert back to DataFrame
    missing_dataframe = pd.DataFrame(X_missing.numpy(), columns=df_sampled.columns)
    
    return df_sampled, missing_dataframe, sample_idx
# def mnar_sampling_cat(df_cat, overall_rate, seed=0):
#     """
#     Create an MNAR missingness mask for a categorical DataFrame.
    
#     For each categorical column (assumed to be label encoded), the missing probability
#     for each value is overall_rate * (prevalence of that value).
    
#     Parameters:
#         df_cat (pd.DataFrame): Categorical data with integer labels.
#         overall_rate (float): Overall missing rate (e.g. 0.3).
#         seed (int): Random seed for reproducibility.
        
#     Returns:
#         df_mask (pd.DataFrame): Boolean DataFrame (same shape as df_cat) where True indicates a missing value.
#     """
#     np.random.seed(seed)
#     df_mask = pd.DataFrame(index=df_cat.index, columns=df_cat.columns)
#     for col in df_cat.columns:
#         # Compute the normalized frequency (prevalence) for each category in this column.
#         freqs = df_cat[col].value_counts(normalize=True)
#         # Map each row's category to its prevalence
#         prob = df_cat[col].map(freqs)
#         # The missing probability for this row is overall_rate * (prevalence of its value)
#         mask_col = np.random.rand(len(df_cat)) < (overall_rate * prob)
#         df_mask[col] = mask_col
#     return df_mask.astype(bool)


# def mnar_sampling_cat_ohe(df_ohe, overall_rate, original_columns, seed=0):
#     """
#     Create an MNAR missingness mask for one-hot encoded categorical data.

#     For each original categorical column, the missing probability for each row is 
#     overall_rate * (prevalence of that category), where the prevalence is calculated
#     from the one-hot encoded columns.

#     Parameters:
#         df_ohe (pd.DataFrame): One-hot encoded categorical data.
#         overall_rate (float): Overall missing rate (e.g. 0.3).
#         original_columns (list): List of original categorical column names (before encoding).
#         seed (int): Random seed for reproducibility.

#     Returns:
#         df_mask (pd.DataFrame): Boolean DataFrame (same shape as df_ohe) where True indicates a missing value.
#     """
#     np.random.seed(seed)
#     df_mask = pd.DataFrame(index=df_ohe.index, columns=df_ohe.columns)

#     # Loop over each original categorical column
#     for orig_col in original_columns:
#         # Identify the corresponding one-hot encoded columns for the original column
#         one_hot_cols = [col for col in df_ohe.columns if col.startswith(orig_col)]

#         # Calculate the prevalence of each category (sum of the one-hot columns / total samples)
#         category_prevalence = df_ohe[one_hot_cols].mean(axis=0)

#         # For each row, compute the missingness mask based on the category prevalence
#         for i, row in df_ohe.iterrows():
#             # Get the category index (i.e., the column with 1 in the one-hot encoding)
#             category_idx = row[one_hot_cols].idxmax()

#             # Calculate the missing probability based on the prevalence of the chosen category
#             prob = overall_rate * category_prevalence[category_idx]
#             # Assign True for missingness based on the probability
#             df_mask.loc[i, one_hot_cols] = np.random.rand(len(one_hot_cols)) < prob

#     return df_mask.astype(bool)

# def mnar_sampling_cat_ohe(df_ohe, overall_rate, original_columns, seed=0):
#     """
#     Vectorized MNAR missingness mask for one-hot encoded categorical data.
#     """
#     np.random.seed(seed)
#     df_mask = pd.DataFrame(False, index=df_ohe.index, columns=df_ohe.columns)
#     prevalences = df_ohe.mean()
    
#     for orig_col in original_columns:
#         # Get one-hot columns for this original feature
#         ohe_cols = [c for c in df_ohe.columns if c.startswith(f"{orig_col}_")]
#         if not ohe_cols:
#             continue
            
#         # Vectorized operations for entire column group
#         active_cats = df_ohe[ohe_cols].idxmax(axis=1)
#         prob = active_cats.map(prevalences) * overall_rate
        
#         # Create mask for all columns in this group simultaneously
#         mask = np.random.rand(len(df_ohe), len(ohe_cols)) < prob.values[:, None]
        
#         df_mask[ohe_cols] = mask
    
#     return df_mask

def mnar_sampling_cat_ohe(df_ohe, overall_rate, original_columns, seed=0):
    """
    Generate a missingness mask for one-hot encoded categorical data such that,
    for each original categorical feature, exactly a target proportion (overall_rate)
    of rows are missing. For each feature, the probability of a row being missing
    is weighted by the prevalence of its active (1-valued) category, thereby following
    an MNAR mechanism.
    """
    np.random.seed(seed)
    df_mask = pd.DataFrame(False, index=df_ohe.index, columns=df_ohe.columns)
    # Compute prevalence for each one-hot column
    prevalences = df_ohe.mean()
    n_rows = len(df_ohe)
    
    for orig_col in original_columns:
        # Get one-hot columns for this original feature
        ohe_cols = [c for c in df_ohe.columns if c.startswith(f"{orig_col}_")]
        if not ohe_cols:
            continue

        # Determine, for each row, the active (nonzero) category
        active_cats = df_ohe[ohe_cols].idxmax(axis=1)
        # Compute weights: use the prevalence of the active category as the MNAR factor
        weights = active_cats.map(prevalences)
        # Normalize weights so they sum to 1 (required for weighted sampling)
        weights = weights / weights.sum()
        
        # Calculate the exact number of rows to be missing for this feature
        n_missing = int(round(overall_rate * n_rows))
        # Weighted random sampling of row indices, without replacement
        missing_indices = np.random.choice(df_ohe.index, size=n_missing, replace=False, p=weights.values)
        
        # For the selected rows, mark all one-hot columns corresponding to this feature as missing
        df_mask.loc[missing_indices, ohe_cols] = True
        
    return df_mask


def mcar_sampling(dataframe, miss_rate, no_of_samples):
    """
    Simulate missing data (MCAR - Missing Completely At Random) in the given dataframe.

    Parameters:
    - dataframe (pd.DataFrame): The input dataframe to be sampled.
    - miss_rate (float): The proportion of values that will be missing.
    - no_of_samples (int): The number of samples to draw from the dataframe.
    - seed (int, optional): Random seed for reproducibility.

    Returns:
    - actual_dataframe (pd.DataFrame): The dataframe with the original data (if sampling was applied).
    - missing_dataframe (pd.DataFrame): The dataframe with missing data introduced.
    - sample_idx (array-like): The indices of the sampled rows.
    """

    if no_of_samples is not None:
        no, dim = dataframe.shape

        if no < no_of_samples:
            no_of_samples = no

        data_x = dataframe.values

        # Sample the row indices
        sample_idx = np.random.choice(no, no_of_samples, replace=False)
        data_x_i = data_x[sample_idx, :]
    else:
        data_x_i = dataframe.values
        sample_idx = dataframe.index  # Use the original indices if no sampling is required

    no_i, dim_i = data_x_i.shape
    #data_x_i = data_x_i.astype(float)


    actual_dataframe = pd.DataFrame(
        data=data_x_i,
        index=[dataframe.index[i] for i in sample_idx],
        columns=dataframe.columns
    )

    # Introduce missing data
    data_m = np.random.binomial(1, 1 - miss_rate, size=(no_i, dim_i))
    miss_data_x = data_x_i.copy()
    miss_data_x[data_m == 0] = np.nan

    missing_dataframe = pd.DataFrame(
        data=miss_data_x,
        index=[dataframe.index[i] for i in sample_idx],
        columns=dataframe.columns
    )

    return actual_dataframe, missing_dataframe, sample_idx

'''
def mcar_sampling(dataframe, miss_rate, no_of_samples):

    if no_of_samples is not None:
        no, dim = dataframe.shape

        if no < no_of_samples:
            no_of_samples = no

        data_x = dataframe.values

        # Sample the row indices
        sample_idx = sample_batch_index(no, no_of_samples)
        data_x_i = data_x[sample_idx, :]
    else:
        data_x_i = dataframe.values
        sample_idx = dataframe.index  # Use the original indices if no sampling is required

    no_i, dim_i = data_x_i.shape

    actual_dataframe = pd.DataFrame(
        data=data_x_i[0:, 0:],
        index=[dataframe.index[i] for i in sample_idx],
        columns=dataframe.columns
    )

    # **Modification starts here:**
    # Determine the number of rows to introduce missingness
    num_rows_with_missingness = int(np.ceil(no_i * miss_rate))

    # Randomly select rows to introduce missingness from the sampled indices
    rows_to_modify = np.random.choice(range(no_i), size=num_rows_with_missingness, replace=False)

    # Introduce missing data only in the selected rows
    data_m = binary_sampler(1 - miss_rate, num_rows_with_missingness, dim_i)
    miss_data_x = data_x_i.copy()
    miss_data_x[rows_to_modify, :] = np.where(data_m == 0, np.nan, miss_data_x[rows_to_modify, :])

    missing_dataframe = pd.DataFrame(
        data=miss_data_x[0:, 0:],
        index=[dataframe.index[i] for i in sample_idx],
        columns=dataframe.columns
    )

    return actual_dataframe, missing_dataframe, sample_idx
'''


# def mar_sampling(dataframe, miss_rate, no_of_samples):
#     '''introduce miss_rate percentage of missing data in a dataset in randomly
#     Args:
#     - data: original data
#     - missing_rate: percentage of data missing (50% should be sent as .5)
#     - no_of_samples: no of rows to be samples
#     Returns:
#     - miss_data_x: dataset with missing data
#     '''

#     if no_of_samples != None:
#         no, dim = dataframe.shape

#         if no < no_of_samples:
#             no_of_samples = no

#         data_x = dataframe.values#.astype(np.float32)

#         sample_idx = sample_batch_index(no, no_of_samples)
#         data_x_i = data_x[sample_idx, :]
#     else:
#         data_x_i = dataframe.values.astype(np.float32)
#         sample_idx = dataframe.index  # Use the original indices if no sampling is required
#     no_i, dim_i = data_x_i.shape

#     actual_dataframe = pd.DataFrame(
#         data=data_x_i[0:,0:],
#         index=[i for i in range(data_x_i.shape[0])],
#         columns=dataframe.columns
#         )

#     missing=0
#     j_size = len(data_x_i)
#     max_missing = j_size * len(data_x_i[0]) * miss_rate
    
#     if dim_i < 5:
#         raise ValueError("There should be more than five features")
#     if miss_rate>.85:
#         raise ValueError("Miss rate can not be more than 85 percent")

#     quantile_low = miss_rate / 2
#     quantile_high = 1 - miss_rate / 2

#     for i in range(0, dim_i):
#         np.random.seed(i)
#         sc1 = np.random.choice([x for x in range(0,dim_i-1) if x not in [i]])
#         sc2 = np.random.choice([x for x in range(0,dim_i-1) if x not in [i,sc1]])
#         sc3 = np.random.choice([x for x in range(0,dim_i-1) if x not in [i,sc1,sc2]])
#         df_1 = actual_dataframe[actual_dataframe[dataframe.columns[sc1]] >= actual_dataframe[dataframe.columns[sc1]].quantile(quantile_high)]
#         df_2 = actual_dataframe[actual_dataframe[dataframe.columns[sc1]] >= actual_dataframe[dataframe.columns[sc1]].quantile(quantile_low)]
#         df_3 = actual_dataframe[actual_dataframe[dataframe.columns[sc2]] >= actual_dataframe[dataframe.columns[sc2]].quantile(quantile_high)]
#         df_4 = actual_dataframe[actual_dataframe[dataframe.columns[sc2]] >= actual_dataframe[dataframe.columns[sc2]].quantile(quantile_low)]
#         df_5 = actual_dataframe[actual_dataframe[dataframe.columns[sc3]] >= actual_dataframe[dataframe.columns[sc3]].quantile(quantile_high)]
#         df_6 = actual_dataframe[actual_dataframe[dataframe.columns[sc3]] >= actual_dataframe[dataframe.columns[sc3]].quantile(quantile_low)]
#         result_indexes = list(set(df_1.index)|set(df_2.index)|set(df_3.index)|set(df_4.index)|set(df_5.index)|set(df_6.index))
#         random.shuffle(result_indexes)
        
#         data_m_bin = binary_sampler(1, no_i, 1)
#         column_limit = math.ceil(no_i*miss_rate)
#         column_missing = 0
        
#         for j in result_indexes:
#             if missing<max_missing and column_missing<column_limit:
#                 data_m_bin[j] = 0
#                 column_missing+=1
#                 missing+=1
            
#         if 'data_m' in vars():
#             data_m = np.append(data_m, data_m_bin, 1)
#         else:
#             data_m = data_m_bin


    
#     # print("max missing: "+str(max_missing)+":::   total removed:"+str(missing))
#     miss_data_x = data_x_i.copy()
#     miss_data_x[data_m == 0] = np.nan


#     missing_dataframe = pd.DataFrame(
#         data=miss_data_x[0:,0:],
#         index=[i for i in range(miss_data_x.shape[0])],
#         columns=dataframe.columns
#         )

#     return actual_dataframe, missing_dataframe, sample_idx
import numpy as np
import pandas as pd
import math
import random
import torch


from missingness.utils import sample_batch_index, binary_sampler
from scipy import optimize

def random_sampling(dataframe, no_of_samples):
    no, dim = dataframe.shape

    if no < no_of_samples:
        no_of_samples = no

    data_x = dataframe.values.astype(np.float32)
    sample_idx = sample_batch_index(no, no_of_samples)
    data_x_i = data_x[sample_idx, :]

    actual_dataframe = pd.DataFrame(
        data=data_x_i[0:,0:],
        index=[i for i in range(data_x_i.shape[0])],
        columns=dataframe.columns
        )
    return actual_dataframe

# def mcar_sampling(dataframe, miss_rate, no_of_samples):
#     '''introduce miss_rate percentage of missing data in a dataset in completely randomly
#     Args:
#     - data: original data
#     - missing_rate: percentage of data missing
#     - no_of_samples: no of rows to be samples
#     Returns:
#     - miss_data_x: dataset with missing data
#     '''
#     if no_of_samples != None:
#         no, dim = dataframe.shape

#         if no < no_of_samples:
#             no_of_samples = no

#         data_x = dataframe.values.astype(np.float32)

#         sample_idx = sample_batch_index(no, no_of_samples)
#         data_x_i = data_x[sample_idx, :]
#     else:
#         data_x_i = dataframe.values.astype(np.float32)

#     no_i, dim_i = data_x_i.shape

#     actual_dataframe = pd.DataFrame(
#         data=data_x_i[0:,0:],
#         index=[i for i in range(data_x_i.shape[0])],
#         columns=dataframe.columns
#         )

#     # Introduce missing data
#     data_m = binary_sampler(1 - miss_rate, no_i, dim_i)
#     miss_data_x = data_x_i.copy()
#     miss_data_x[data_m == 0] = np.nan


#     missing_dataframe = pd.DataFrame(
#         data=miss_data_x[0:,0:],
#         index=[i for i in range(miss_data_x.shape[0])],
#         columns=dataframe.columns
#         )

#     return actual_dataframe, missing_dataframe



def pick_coeffs(X, idxs_obs=None, idxs_nas=None, self_mask=False):
    n, d = X.shape
    if self_mask:
        coeffs = torch.randn(d)
        Wx = X * coeffs
        coeffs /= torch.std(Wx, 0)
    else:
        d_obs = len(idxs_obs)
        d_na = len(idxs_nas)
        coeffs = torch.randn(d_obs, d_na, dtype=X.dtype)
        Wx = X[:, idxs_obs].mm(coeffs)
        coeffs /= torch.std(Wx, 0, keepdim=True)
    return coeffs

# def fit_intercepts(X, coeffs, p, self_mask=False):
#     if self_mask:
#         d = len(coeffs)
#         intercepts = torch.zeros(d)
#         for j in range(d):
#             def f(x):
#                 return torch.sigmoid(X * coeffs[j] + x).mean().item() - p
#             intercepts[j] = optimize.bisect(f, -50, 50)
#     else:
#         d_obs, d_na = coeffs.shape
#         intercepts = torch.zeros(d_na)
#         for j in range(d_na):
#             def f(x):
#                 return torch.sigmoid(X.mv(coeffs[:, j]) + x).mean().item() - p
#             intercepts[j] = optimize.bisect(f, -50, 50)
#     return intercepts

def fit_intercepts(X, coeffs, p, self_mask=False):
    if self_mask:
        d = len(coeffs)
        intercepts = torch.zeros(d)
        for j in range(d):
            def f(x):
                return torch.sigmoid(X * coeffs[j] + x).mean().item() - p

            f_left = f(-100)
            f_right = f(100)

            if f_left * f_right < 0:
                intercepts[j] = optimize.bisect(f, -100, 100)
            else:
                intercepts[j] = -100 if abs(f_left) < abs(f_right) else 100

    else:
        d_obs, d_na = coeffs.shape
        intercepts = torch.zeros(d_na)
        for j in range(d_na):
            def f(x):
                return torch.sigmoid(X.mv(coeffs[:, j]) + x).mean().item() - p

            f_left = f(-100)
            f_right = f(100)

            if f_left * f_right < 0:
                intercepts[j] = optimize.bisect(f, -100, 100)
            else:
                intercepts[j] = -100 if abs(f_left) < abs(f_right) else 100

    return intercepts

def mar_sampling(dataframe, miss_rate, no_of_samples=None, p_obs=0.3, seed=0):
    """
    MAR sampling using a logistic model.

    Args:
        dataframe: pd.DataFrame (must contain only numeric columns)
        miss_rate: float (0 to 1) - desired missingness proportion
        no_of_samples: int or None - number of rows to sample (or use all)
        p_obs: float - fraction of features to be fully observed
        seed: int - random seed

    Returns:
        actual_dataframe: DataFrame with original (or sampled) data
        missing_dataframe: DataFrame with MAR missing values
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Use numeric-only features
    dataframe = dataframe.select_dtypes(include=[np.number])
    if dataframe.shape[1] < 2:
        raise ValueError("MAR requires at least two numeric features.")

    # Optionally sample rows
    if no_of_samples is not None:
        dataframe = dataframe.sample(n=min(no_of_samples, len(dataframe)), random_state=seed).reset_index(drop=True)

    X = torch.tensor(dataframe.values, dtype=torch.double)
    n, d = X.shape

    # Determine observed vs to-be-masked features
    d_obs = max(int(p_obs * d), 1)
    idxs_obs = np.random.choice(d, d_obs, replace=False)
    idxs_nas = [i for i in range(d) if i not in idxs_obs]

    # Generate logistic coefficients and intercepts
    coeffs = pick_coeffs(X, idxs_obs, idxs_nas, self_mask=False)
    intercepts = fit_intercepts(X[:, idxs_obs], coeffs, miss_rate, self_mask=False)

    # Compute probabilities and apply masking
    ps = torch.sigmoid(X[:, idxs_obs].mm(coeffs) + intercepts)
    mask = torch.rand(n, len(idxs_nas)) < ps

    # Apply mask
    X_missing = X.clone()
    for i, col in enumerate(idxs_nas):
        X_missing[mask[:, i], col] = float('nan')

    # Wrap up in pandas DataFrame
    actual_dataframe = pd.DataFrame(X.numpy(), columns=dataframe.columns)
    missing_dataframe = pd.DataFrame(X_missing.numpy(), columns=dataframe.columns)

    return actual_dataframe, missing_dataframe

# def mar_sampling(dataframe, miss_rate, no_of_samples):
#     '''introduce miss_rate percentage of missing data in a dataset in randomly
#     Args:
#     - data: original data
#     - missing_rate: percentage of data missing (50% should be sent as .5)
#     - no_of_samples: no of rows to be samples
#     Returns:
#     - miss_data_x: dataset with missing data
#     '''

#     if no_of_samples != None:
#         no, dim = dataframe.shape

#         if no < no_of_samples:
#             no_of_samples = no

#         data_x = dataframe.values.astype(np.float32)

#         sample_idx = sample_batch_index(no, no_of_samples)
#         data_x_i = data_x[sample_idx, :]
#     else:
#         data_x_i = dataframe.values.astype(np.float32)
#     no_i, dim_i = data_x_i.shape

#     actual_dataframe = pd.DataFrame(
#         data=data_x_i[0:,0:],
#         index=[i for i in range(data_x_i.shape[0])],
#         columns=dataframe.columns
#         )

#     missing=0
#     j_size = len(data_x_i)
#     max_missing = j_size * len(data_x_i[0]) * miss_rate
    
#     if dim_i < 5:
#         raise ValueError("There should be more than five features")
#     if miss_rate>.85:
#         raise ValueError("Miss rate can not be more than 85 percent")

#     quantile_low = miss_rate / 2
#     quantile_high = 1 - miss_rate / 2

#     for i in range(0, dim_i):
#         np.random.seed(i)
#         sc1 = np.random.choice([x for x in range(0,dim_i-1) if x not in [i]])
#         sc2 = np.random.choice([x for x in range(0,dim_i-1) if x not in [i,sc1]])
#         sc3 = np.random.choice([x for x in range(0,dim_i-1) if x not in [i,sc1,sc2]])
#         df_1 = actual_dataframe[actual_dataframe[dataframe.columns[sc1]] >= actual_dataframe[dataframe.columns[sc1]].quantile(quantile_high)]
#         df_2 = actual_dataframe[actual_dataframe[dataframe.columns[sc1]] >= actual_dataframe[dataframe.columns[sc1]].quantile(quantile_low)]
#         df_3 = actual_dataframe[actual_dataframe[dataframe.columns[sc2]] >= actual_dataframe[dataframe.columns[sc2]].quantile(quantile_high)]
#         df_4 = actual_dataframe[actual_dataframe[dataframe.columns[sc2]] >= actual_dataframe[dataframe.columns[sc2]].quantile(quantile_low)]
#         df_5 = actual_dataframe[actual_dataframe[dataframe.columns[sc3]] >= actual_dataframe[dataframe.columns[sc3]].quantile(quantile_high)]
#         df_6 = actual_dataframe[actual_dataframe[dataframe.columns[sc3]] >= actual_dataframe[dataframe.columns[sc3]].quantile(quantile_low)]
#         result_indexes = list(set(df_1.index)|set(df_2.index)|set(df_3.index)|set(df_4.index)|set(df_5.index)|set(df_6.index))
#         random.shuffle(result_indexes)
        
#         data_m_bin = binary_sampler(1, no_i, 1)
#         column_limit = math.ceil(no_i*miss_rate)
#         column_missing = 0
        
#         for j in result_indexes:
#             if missing<max_missing and column_missing<column_limit:
#                 data_m_bin[j] = 0
#                 column_missing+=1
#                 missing+=1
            
#         if 'data_m' in vars():
#             data_m = np.append(data_m, data_m_bin, 1)
#         else:
#             data_m = data_m_bin


    
#     # print("max missing: "+str(max_missing)+":::   total removed:"+str(missing))
#     miss_data_x = data_x_i.copy()
#     miss_data_x[data_m == 0] = np.nan


#     missing_dataframe = pd.DataFrame(
#         data=miss_data_x[0:,0:],
#         index=[i for i in range(miss_data_x.shape[0])],
#         columns=dataframe.columns
#         )

#     return actual_dataframe, missing_dataframe


def mnar_sampling(dataframe, miss_rate, no_of_samples):
    '''introduce miss_rate percentage of missing data in a dataset in randomly
    Args:
    - data: original data
    - missing_rate: percentage of data missing (50% should be sent as .5)
    - no_of_samples: no of rows to be samples
    Returns:
    - miss_data_x: dataset with missing data
    '''

    if no_of_samples != None:
        no, dim = dataframe.shape

        if no < no_of_samples:
            no_of_samples = no

        data_x = dataframe.values.astype(np.float32)

        sample_idx = sample_batch_index(no, no_of_samples)
        data_x_i = data_x[sample_idx, :]
    else:
        data_x_i = dataframe.values.astype(np.float32)

    no_i, dim_i = data_x_i.shape

    actual_dataframe = pd.DataFrame(
        data=data_x_i[0:,0:],
        index=[i for i in range(data_x_i.shape[0])],
        columns=dataframe.columns
        )

    missing=0
    j_size = len(data_x_i)
    max_missing = j_size * len(data_x_i[0]) * miss_rate
    maxReached = False

    high = True
    low = True

    if dim_i < 2:
        raise ValueError("There should be more than one feature")
    if miss_rate>.85:
        raise ValueError("Miss rate can not be more than 85 percent")

    quantile_low = miss_rate / 2
    quantile_high = 1 - (miss_rate / 2)

    column_limit = math.ceil(no_i*miss_rate)
    for i in range(0, dim_i):
        column_missing = 0
        percentile_high = actual_dataframe[dataframe.columns[i]].quantile(quantile_high)
        percentile_low = actual_dataframe[dataframe.columns[i]].quantile(quantile_low)
        data_m_bin = binary_sampler(1, no_i, 1)
        for j in range (0, no_i):
            if high and percentile_high <= data_x_i[j][i] and not maxReached and column_missing<column_limit:
                data_m_bin[j] = 0
                missing+=1
                column_missing+=1
                if missing >= max_missing:
                    maxReached = True
            elif low and percentile_low >= data_x_i[j][i] and not maxReached and column_missing<column_limit:
                data_m_bin[j] = 0
                missing+=1
                column_missing+=1
                if missing >= max_missing:
                    maxReached = True
            
        if 'data_m' in vars():
            data_m = np.append(data_m, data_m_bin, 1)
        else:
            data_m = data_m_bin

    
    # print("max missing: "+str(max_missing)+":::   total removed:"+str(missing))
    miss_data_x = data_x_i.copy()
    miss_data_x[data_m == 0] = np.nan


    missing_dataframe = pd.DataFrame(
        data=miss_data_x[0:,0:],
        index=[i for i in range(miss_data_x.shape[0])],
        columns=dataframe.columns
        )

    return actual_dataframe, missing_dataframe

'''
def mar_sampling(dataframe, miss_rate, no_of_samples):

    if no_of_samples != None:
        no, dim = dataframe.shape

        if no < no_of_samples:
            no_of_samples = no

        data_x = dataframe.values#.astype(np.float32)

        sample_idx = sample_batch_index(no, no_of_samples)
        data_x_i = data_x[sample_idx, :]
    else:
        data_x_i = dataframe.values.astype(np.float32)
    no_i, dim_i = data_x_i.shape

    actual_dataframe = pd.DataFrame(
        data=data_x_i[0:,0:],
        index=[i for i in range(data_x_i.shape[0])],
        columns=dataframe.columns
        )

    missing=0
    j_size = len(data_x_i)
    max_missing = j_size * len(data_x_i[0]) * miss_rate
    
    if dim_i < 5:
        raise ValueError("There should be more than five features")
    if miss_rate>.85:
        raise ValueError("Miss rate can not be more than 85 percent")

    quantile_low = miss_rate / 2
    quantile_high = 1 - miss_rate / 2

    for i in range(0, dim_i):
        np.random.seed(i)
        sc1 = np.random.choice([x for x in range(0,dim_i-1) if x not in [i]])
        sc2 = np.random.choice([x for x in range(0,dim_i-1) if x not in [i,sc1]])
        sc3 = np.random.choice([x for x in range(0,dim_i-1) if x not in [i,sc1,sc2]])
        df_1 = actual_dataframe[actual_dataframe[dataframe.columns[sc1]] >= actual_dataframe[dataframe.columns[sc1]].quantile(quantile_high)]
        df_2 = actual_dataframe[actual_dataframe[dataframe.columns[sc1]] >= actual_dataframe[dataframe.columns[sc1]].quantile(quantile_low)]
        df_3 = actual_dataframe[actual_dataframe[dataframe.columns[sc2]] >= actual_dataframe[dataframe.columns[sc2]].quantile(quantile_high)]
        df_4 = actual_dataframe[actual_dataframe[dataframe.columns[sc2]] >= actual_dataframe[dataframe.columns[sc2]].quantile(quantile_low)]
        df_5 = actual_dataframe[actual_dataframe[dataframe.columns[sc3]] >= actual_dataframe[dataframe.columns[sc3]].quantile(quantile_high)]
        df_6 = actual_dataframe[actual_dataframe[dataframe.columns[sc3]] >= actual_dataframe[dataframe.columns[sc3]].quantile(quantile_low)]
        result_indexes = list(set(df_1.index)|set(df_2.index)|set(df_3.index)|set(df_4.index)|set(df_5.index)|set(df_6.index))
        random.shuffle(result_indexes)
        
        data_m_bin = binary_sampler(1, no_i, 1)
        column_limit = math.ceil(no_i*miss_rate)
        column_missing = 0
        
        for j in result_indexes:
            if missing<max_missing and column_missing<column_limit:
                data_m_bin[j] = 0
                column_missing+=1
                missing+=1
            
        if 'data_m' in vars():
            data_m = np.append(data_m, data_m_bin, 1)
        else:
            data_m = data_m_bin


    
    # print("max missing: "+str(max_missing)+":::   total removed:"+str(missing))
    miss_data_x = data_x_i.copy()
    miss_data_x[data_m == 0] = np.nan


    missing_dataframe = pd.DataFrame(
        data=miss_data_x[0:,0:],
        index=[i for i in range(miss_data_x.shape[0])],
        columns=dataframe.columns
        )

    return actual_dataframe, missing_dataframe

'''

def mnar_sampling(dataframe, miss_rate, no_of_samples):
    '''introduce miss_rate percentage of missing data in a dataset in randomly
    Args:
    - data: original data
    - missing_rate: percentage of data missing (50% should be sent as .5)
    - no_of_samples: no of rows to be samples
    Returns:
    - miss_data_x: dataset with missing data
    '''

    if no_of_samples != None:
        no, dim = dataframe.shape

        if no < no_of_samples:
            no_of_samples = no

        data_x = dataframe.values.astype(np.float32)

        sample_idx = sample_batch_index(no, no_of_samples)
        data_x_i = data_x[sample_idx, :]
    else:
        data_x_i = dataframe.values.astype(np.float32)
        sample_idx = dataframe.index  # Use the original indices if no sampling is required

    no_i, dim_i = data_x_i.shape

    actual_dataframe = pd.DataFrame(
        data=data_x_i[0:,0:],
        index=[i for i in range(data_x_i.shape[0])],
        columns=dataframe.columns
        )

    missing=0
    j_size = len(data_x_i)
    max_missing = j_size * len(data_x_i[0]) * miss_rate
    maxReached = False

    high = True
    low = True

    if dim_i < 1:
        raise ValueError("There should be more than one feature")
    if miss_rate>.85:
        raise ValueError("Miss rate can not be more than 85 percent")

    quantile_low = miss_rate / 2
    quantile_high = 1 - (miss_rate / 2)

    column_limit = math.ceil(no_i*miss_rate)
    for i in range(0, dim_i):
        column_missing = 0
        percentile_high = actual_dataframe[dataframe.columns[i]].quantile(quantile_high)
        percentile_low = actual_dataframe[dataframe.columns[i]].quantile(quantile_low)
        data_m_bin = binary_sampler(1, no_i, 1)
        for j in range (0, no_i):
            if high and percentile_high <= data_x_i[j][i] and not maxReached and column_missing<column_limit:
                data_m_bin[j] = 0
                missing+=1
                column_missing+=1
                if missing >= max_missing:
                    maxReached = True
            elif low and percentile_low >= data_x_i[j][i] and not maxReached and column_missing<column_limit:
                data_m_bin[j] = 0
                missing+=1
                column_missing+=1
                if missing >= max_missing:
                    maxReached = True
            
        if 'data_m' in vars():
            data_m = np.append(data_m, data_m_bin, 1)
        else:
            data_m = data_m_bin

    
    # print("max missing: "+str(max_missing)+":::   total removed:"+str(missing))
    miss_data_x = data_x_i.copy()
    miss_data_x[data_m == 0] = np.nan


    missing_dataframe = pd.DataFrame(
        data=miss_data_x[0:,0:],
        index=[i for i in range(miss_data_x.shape[0])],
        columns=dataframe.columns
        )

    return actual_dataframe, missing_dataframe, sample_idx

'''
def mnar_sampling(dataframe, miss_rate, no_of_samples):
    
    if no_of_samples != None:
        no, dim = dataframe.shape

        if no < no_of_samples:
            no_of_samples = no

        data_x = dataframe.values.astype(np.float32)

        sample_idx = sample_batch_index(no, no_of_samples)
        data_x_i = data_x[sample_idx, :]
    else:
        data_x_i = dataframe.values.astype(np.float32)

    no_i, dim_i = data_x_i.shape

    actual_dataframe = pd.DataFrame(
        data=data_x_i[0:,0:],
        index=[i for i in range(data_x_i.shape[0])],
        columns=dataframe.columns
        )

    missing=0
    j_size = len(data_x_i)
    max_missing = j_size * len(data_x_i[0]) * miss_rate
    maxReached = False

    high = True
    low = True

    if dim_i < 2:
        raise ValueError("There should be more than one feature")
    if miss_rate>.85:
        raise ValueError("Miss rate can not be more than 85 percent")

    quantile_low = miss_rate / 2
    quantile_high = 1 - (miss_rate / 2)

    column_limit = math.ceil(no_i*miss_rate)
    for i in range(0, dim_i):
        column_missing = 0
        percentile_high = actual_dataframe[dataframe.columns[i]].quantile(quantile_high)
        percentile_low = actual_dataframe[dataframe.columns[i]].quantile(quantile_low)
        data_m_bin = binary_sampler(1, no_i, 1)
        for j in range (0, no_i):
            if high and percentile_high <= data_x_i[j][i] and not maxReached and column_missing<column_limit:
                data_m_bin[j] = 0
                missing+=1
                column_missing+=1
                if missing >= max_missing:
                    maxReached = True
            elif low and percentile_low >= data_x_i[j][i] and not maxReached and column_missing<column_limit:
                data_m_bin[j] = 0
                missing+=1
                column_missing+=1
                if missing >= max_missing:
                    maxReached = True
            
        if 'data_m' in vars():
            data_m = np.append(data_m, data_m_bin, 1)
        else:
            data_m = data_m_bin

    
    # print("max missing: "+str(max_missing)+":::   total removed:"+str(missing))
    miss_data_x = data_x_i.copy()
    miss_data_x[data_m == 0] = np.nan


    missing_dataframe = pd.DataFrame(
        data=miss_data_x[0:,0:],
        index=[i for i in range(miss_data_x.shape[0])],
        columns=dataframe.columns
        )

    return actual_dataframe, missing_dataframe
'''



# def mnar_sampling(dataframe, miss_rate, no_of_samples=None):
#     """
#     Introduce missing values in a dataset following the MNAR (Missing Not At Random) mechanism.

#     Args:
#     - dataframe (pd.DataFrame): Original dataset.
#     - miss_rate (float): Percentage of missing data (e.g., 50% should be sent as 0.5).
#     - no_of_samples (int, optional): Number of rows to sample. Defaults to using the entire dataset.

#     Returns:
#     - actual_dataframe (pd.DataFrame): Dataset before missing values were introduced.
#     - missing_dataframe (pd.DataFrame): Dataset with missing values.
#     """

#     if no_of_samples is not None:
#         no, dim = dataframe.shape
#         if no < no_of_samples:
#             no_of_samples = no
#         data_x = dataframe.values.astype(np.float32)
#         sample_idx = sample_batch_index(no, no_of_samples)
#         data_x_i = data_x[sample_idx, :]
#     else:
#         data_x_i = dataframe.values.astype(np.float32)

#     no_i, dim_i = data_x_i.shape

#     # Convert back to DataFrame
#     actual_dataframe = pd.DataFrame(data=data_x_i, columns=dataframe.columns)

#     if dim_i < 2:
#         raise ValueError("There should be more than one feature.")
#     if miss_rate >= 1.0:
#         raise ValueError("Miss rate cannot be 100% or more, as it would remove all data.")

#     # Initialize missing data mask
#     data_m = np.ones((no_i, dim_i))  

#     # Set missing rate thresholds
#     quantile_low = miss_rate / 2
#     quantile_high = 1 - (miss_rate / 2)

#     max_missing = int(no_i * dim_i * miss_rate)
#     missing = 0
#     maxReached = False

#     column_limit = math.ceil(no_i * miss_rate)  # Ensure at least some values remain

#     for i in range(dim_i):
#         column_missing = 0
#         percentile_high = actual_dataframe.iloc[:, i].quantile(quantile_high)
#         percentile_low = actual_dataframe.iloc[:, i].quantile(quantile_low)

#         for j in range(no_i):
#             if (data_x_i[j][i] >= percentile_high or data_x_i[j][i] <= percentile_low) and not maxReached:
#                 if column_missing < column_limit - 1:  # Ensure at least 1 value remains
#                     data_m[j, i] = 0  # Mark as missing
#                     missing += 1
#                     column_missing += 1
#                     if missing >= max_missing:
#                         maxReached = True

#     # Create missing data
#     miss_data_x = data_x_i.copy()
#     miss_data_x[data_m == 0] = np.nan

#     # Convert back to DataFrame
#     missing_dataframe = pd.DataFrame(data=miss_data_x, columns=dataframe.columns)

#     return actual_dataframe, missing_dataframe


