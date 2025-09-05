import torch

import numpy as np
import pandas as pd


from helpers.persistence import *


# import for corruption
from helpers.noises import add_noise
from missingness.sampler import mar_sampling, mcar_sampling, mnar_sampling
import MICE.micegradient.micegradient as mg
from sklearn.impute import KNNImputer

# need to remove unwanted imports

default_settings = {
    'method': 'pass',          # 'pass', 'noise' | 'draw' | 'sample' | 'knn' | 'mice'
    'corruption_rate': .6,      # 0.6 or between 0-1; fraction of features to corrupt (not used for mice/knn)
    'missing': .2,              # 0.2 between 0-1 float;  % of missingness
    'missing_type': 'mcar',     # 'mcar' | 'mnar' | 'mar'
    'mice': 'LinearRegression', # 'LinearRegression' | 'DecisionTree' | others...
}


class Corruptor:
    def __init__(self, X_original, settings, cat_mask=[]):
        '''
        X_orginal = Full (train/valid) features (needed for sampling/drawing)
        settings = dictionary of settings (see default settings)

        '''
        # overwrite keys provided on default settings
        settings = {**default_settings, **settings}
        # print(settings)
        self.method = settings['method']
        self.corruption_rate = settings['corruption_rate']
        self.X_original = torch.clone(X_original)
        self.missing = settings['missing']
        self.cat_mask = cat_mask
        
        sampler_map = {
            'mnar': mnar_sampling,
            'mcar': mcar_sampling,
            'mar': mar_sampling,
        }
        self.missing_type = settings['missing_type']
        self.missing_sampler = sampler_map[self.missing_type]
        self.mice = settings['mice']
        
    def _get_mask(self, X):
        '''
        TODO: implement without for-loop
        '''
        n,d = X.shape
        # debug_mode and print(X.shape)
        d_corrupt = int(self.corruption_rate * d)
        x = np.zeros((n,d))

        for i in range(n):
            a = np.arange(1,d+1)
            a1 = np.random.permutation(a)
            x[i,:] = a1

        mask = np.where(x<=d_corrupt, 1, 0)

        device = X.device
        mask = torch.from_numpy(mask)
        
        # mask = mask if device<0 else mask.to(device)
        mask = mask.to(device)
        # debug_mode and print('mask shape', mask.shape)
        
        return mask
        
    def _knn(self, X):
        device = X.device
            
        _, X_missing = self.missing_sampler(pd.DataFrame(X), self.missing, None)
        
        # KNNImputer removes features that has all missing
        # KNNImputer(keep_empty_features=True) in sklearn 1.2 added a parameter
        # However, we are using sklearn 1.0.x and cant upgrade due to depencency contraints
        # so we have to do it ourselves
        empty_cols = X_missing.columns[X_missing.isna().all(axis=0)].values
        X_missing.loc[:, empty_cols] = 0
        
        knn_imputer = KNNImputer()
        X_imputed = knn_imputer.fit_transform(X_missing)
        X1 = torch.from_numpy(X_imputed).to(device)
        
        # if X.shape[1] != X1.shape[1]:
        #     print(X)
        #     print(X_missing)
        #     print(X1)
        
        return X1
    
    def _mice(self, X):
        device = X.device
        
        _, X_missing = self.missing_sampler(pd.DataFrame(X), self.missing, None)
        empty_cols = X_missing.columns[X_missing.isna().all(axis=0)].values
        X_missing.loc[:, empty_cols] = 0
        
        kernel = mg.MultipleImputedKernel(
                X_missing,
                datasets=1,
                save_all_iterations=False,
                mean_match_candidates=0,
                initialization='median'
        )
        kernel.mice(self.mice, 1, n_estimators=1, n_jobs=4)
        X_imputed = kernel.complete_data(0)
        X1 = torch.from_numpy(X_imputed.values).to(device)
        return X1
    
    def _draw(self, X0):
        ''' 
        replace c*d random select columns for with another random row
        do this for each rows in X0
        where c=corruption_rate and d=number of features
        and X0 is assumed to be unnormalized
        '''
        X = torch.clone(X0)
        device = X.device
        mask = self._get_mask(X)
        
        # select random rows for each row (can have same row idx)
        r = torch.randint(self.X_original.shape[0],(X.shape[0],))
        noise_values = self.X_original[r,:]

        # return (1-mask)*X + mask*imputted
        real = X.mul(1-mask)
        draws = noise_values.mul(mask)

        return real + draws
    
    def _noise(self, X0, mean=0, std=1):
        ''' 
        add gaussian noise, N(mu, std) to c*d random columns for all rows
        where c=corruption_rate and d=number of features    
        '''
        X = torch.clone(X0)

        if mean==0 and std==0: return X

        device = X.device
        mask = self._get_mask(X)

        noise_values = torch.empty_like(X).normal_(mean, std)
        # noise_values = noise_values if device<0 else noise_values.to(device)
        noise_values = noise_values.to(device)

        # debug_mode and print(noise_values.shape)
        noise = noise_values.mul(mask)

        return X+noise
    
    def _sample(self, X0):
        ''' 
        replace c*d random columns for all rows using original feature distribution
        where c=corruption_rate and d=number of features
        and X0 is assumed to be unnormalized
        '''
        X = torch.clone(X0)
        device = X.device
        mask = self._get_mask(X)

        means = torch.mean(self.X_original, dim=0)
        stdevs = torch.std(self.X_original, dim=0)
        
        noise_values = torch.cat([
            torch.empty_like(X[:,i]).normal_(m.item(), s.item()) 
            for i,(m,s) in enumerate(zip(means, stdevs))
        ], dim=-1)

        noise_values = noise_values.reshape(X.shape).contiguous()

        # noise_values = noise_values if device<0 else noise_values.to(device)
        noise_values = noise_values.to(device)

        # return (1-mask)*X + mask*imputted
        real = X.mul(1-mask)
        imputed = noise_values.mul(mask)

        return real + imputed
    
    def __call__(self, X):
        
        method_map = {
            'pass': lambda x: x,
            'noise': self._noise,
            'sample': self._sample,
            'draw': self._draw,
            'knn': self._knn,
            'mice': self._mice,
        }
        
        return method_map[self.method](X)