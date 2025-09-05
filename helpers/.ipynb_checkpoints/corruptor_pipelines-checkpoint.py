import torch
import numpy as np
import pandas as pd
from helpers.persistence import *

# import for corruption
from helpers.noises import add_noise
from missingness.sampler import mar_sampling, mcar_sampling, mnar_sampling
import MICE.micegradient.micegradient as mg

from sklearn.impute import KNNImputer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler, LabelBinarizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin

from helpers.corruptor_df import Corruptor
from copy import deepcopy

debug_mode = False
debug_mode = True


def get_valid_pipeline(X_valid, preprocessor):
    preprocessor = deepcopy(preprocessor)
    current = X_valid
    for k, v in preprocessor.named_steps.items():
        if k == 'corruptor':
            break
        current = v.transform(current)
        # print(k, current)

    v.corruptor.X_original = current
    return preprocessor


def get_col_preprocessor(cols, cat_mask, corrupt_before_ohe):
    num_cols = cols[~cat_mask]
    cat_cols = cols[cat_mask]

    categorical_transformer = Pipeline(
        steps=[
            ("encoder", OneHotEncoder(handle_unknown="error", sparse=False)),
        ]
    )

    col_preprocessor = MyColumnTransformer(
        transformers=[
            ("cat", categorical_transformer, cat_cols),
        ],
        remainder='passthrough',
    )

    return col_preprocessor


def get_fitted_pipeline(X, cols, cat_mask, col_preprocessor,
                        settings,
                        corrupt_before_ohe=False,  # use only for draw method
                        scale_cat=True,
                        ):

    df = pd.DataFrame(X.cpu().numpy(), columns=cols)

    t2df = Tensor2DataFrame(cols)
    df2t = DataFrame2Tensor()
    # num_cols = cols[~cat_mask]
    # cat_cols = cols[cat_mask]
    
    df = col_preprocessor.transform(df)
    num_cols = [x for x in df.columns if 'remainder__' in x]
    cat_cols = [x for x in df.columns if 'cat__' in x]

    # we do not scale numerical columns separate
    # we scale all columns together
    numeric_transformer = Pipeline(
        steps=[
            ('num_scaler', StandardScaler()),
        ]
    )

    cat_transformer = Pipeline(
        steps=[
            ('cat_scaler', StandardScaler()),
        ]
    )

    transformers = [
        ("num", numeric_transformer, num_cols),
    ]
    if scale_cat:
        transformers.insert(0, ("cat", cat_transformer, cat_cols))

    col_scaler = MyColumnTransformer(
        transformers=transformers,
        remainder='passthrough',
    )

    steps = [
        ('t2df', t2df),
        ('preprocessor', col_preprocessor),  # df->df
        ('col_scaler', col_scaler),
        ('df2t', df2t),
    ]

    # t2df->(corrupt)->col_prep(ohe)->col_scaler->df2t
    if corrupt_before_ohe:
        Z = t2df.fit_transform(X)
        c = CorruptorTransformer(settings).fit(Z)
        steps.insert(1, ('corruptor', c))
        
        Z = col_preprocessor.transform(Z)
        Z = col_scaler.fit_transform(Z)
        

    # t2df->col_prep(ohe)->col_scaler->(corrupt)->rescale->df2t
    else:
        Z = t2df.fit_transform(X)
        Z = col_preprocessor.transform(Z)
        Z = col_scaler.fit_transform(Z)
        Z1 = Z

        # fit c and add to pipeline after np2df
        c = CorruptorTransformer(settings).fit(Z)
        Z = c.corruptor._pass(Z)
        steps.insert(3, ('corruptor', c))
        rescale_cat_cols = [x for x in Z.columns if 'cat__cat__' in x]

        if scale_cat and len(rescale_cat_cols) > 0:
            tmp = [('cat_rescaler',
                    Pipeline(
                        steps=[
                            ('cat_scaler', StandardScaler()),
                        ]
                    ),
                    rescale_cat_cols)]
            cat_rescaler = MyColumnTransformer(
                transformers=tmp, remainder='passthrough')
            
            Z = cat_rescaler.fit_transform(Z)

            # assert np.all(np.abs(Z1.values - Z.values) < 1e-7), 'rescaling not same'

            steps.insert(4, ('cat_rescaler', cat_rescaler))

    preprocessor = Pipeline(
        steps=steps
    )

    return preprocessor


class CorruptorTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, settings):
        self.settings = settings

    def fit(self, X, y=None):
        # debug_mode and print('original', X.shape, X)
        self.corruptor = Corruptor(X, self.settings)
        return self

    def transform(self, X):
        X_new = self.corruptor(X)
        return X_new


class MyColumnTransformer(ColumnTransformer):

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(super().transform(X), columns=super().get_feature_names_out(), index=X.index)

    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        return pd.DataFrame(super().fit_transform(X), columns=super().get_feature_names_out(), index=X.index)


class MyStandardScaler(StandardScaler):

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(super().transform(X), columns=X.columns, index=X.index)

    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        return pd.DataFrame(super().fit_transform(X), columns=X.columns, index=X.index)


class Numpy2DataFrame(BaseEstimator, TransformerMixin):
    def __init__(self, cols):
        self.cols = cols

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = pd.DataFrame(X, columns=self.cols)
        return df


class Tensor2DataFrame(BaseEstimator, TransformerMixin):
    def __init__(self, cols):
        self.cols = cols

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = pd.DataFrame(X.cpu().numpy(), columns=self.cols)
        return df


class Numpy2Tensor(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return torch.from_numpy(X)


class DataFrame2Tensor(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # print('final pipeline step')
        # display(X)
        return torch.from_numpy(X.values)
