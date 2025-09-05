import torch
import numpy as np
import pandas as pd


from helpers.persistence import *


# import for corruption
from helpers.noises import add_noise
from missingness.sampler import mar_sampling, mcar_sampling, mnar_sampling
#import MICE.micegradient.micegradient as mg
from sklearn.impute import KNNImputer

# need to remove unwanted imports

default_settings = {
    'method': 'pass',          # 'pass', 'noise' | 'draw' | 'sample' | 'knn' | 'mice'
    'corruption_rate': .6,      # 0.6 or between 0-1; fraction of features to corrupt (not used for mice/knn)
    'missing': .2,              # 0.2 between 0-1 float;  % of missingness
    'missing_type': 'mcar',     # 'mcar' | 'mnar' | 'mar'
    'mice': 'LinearRegression', # 'LinearRegression' | 'DecisionTree' | others...
    'cluster_labels': torch.Tensor([]), # becomes 1d tensor after kmeans
}





class Corruptor:
    def __init__(self, X_original, settings):
        '''
        X_orginal = Full (train/valid) features (needed for sampling/drawing)
        settings = dictionary of settings (see default settings)
        NOTE: X_original is a dataframe in this version of the code
        '''
        # overwrite keys provided on default settings
        settings = {**default_settings, **settings}
        # print(settings)
        self.method = settings['method']
        self.corruption_rate = settings['corruption_rate']
        self.X_original = X_original
        self.cluster_labels = settings['cluster_labels']
        self.clusters = torch.unique(self.cluster_labels)
        self.missing = settings['missing']
        
        sampler_map = {
            'mnar': mnar_sampling,
            'mcar': mcar_sampling,
            'mar': mar_sampling,
        }
        self.missing_type = settings['missing_type']
        self.missing_sampler = sampler_map[self.missing_type]
        self.mice = settings['mice']
        
        self.in_sample_knn_imputer = KNNImputer().fit(X_original)
        
    def _get_mask(self, X):
        '''
        TODO: implement without for-loop
        '''
        n,d = X.shape
        # debug_mode and print(X.shape)
        d_corrupt = int(self.corruption_rate * d)
        x = np.zeros((n,d))
        
        a = np.arange(1,d+1)
        for i in range(n):
            a1 = np.random.permutation(a)
            x[i,:] = a1

        mask = np.where(x<=d_corrupt, 1, 0)

        mask = torch.from_numpy(mask)
        
        mask = mask.to(X.device)
        # debug_mode and print('mask shape', mask.shape)
        
        return mask
    
    def _modify(func):
        def modified(instance, X):
            # print(X)s
            # note all categories are converted to ordinal (long) for working with tensors
            # if df without OHE is passed it will converted to float
            # however non-OHE df is done for SCARF draw
            # other methods use OHEncoded df
            # We need float to work with some tensor functions
            Z = torch.from_numpy(X.values).float()
            Z = func(instance, Z).numpy()

            df1 = pd.DataFrame(Z, columns=X.columns)
            df = df1.copy(deep=0)
            # display(df1)

            # 'cat__' is hard-coded in corruptor pipelines
            categoricals = np.unique([
                '_'.join(x.split('__')[-1].split('_')[:-1]) 
                for x in df.columns if 'cat__' in x
                ])

            def arg_max_cat(row):
                # print(row)
                x = torch.from_numpy(row.values)
                result = torch.zeros_like(x)
                # print('row.argmax()', row.argmax())
                result[row.argmax()] = 1
                s = pd.Series(result.numpy())
                # print(row.values, '->', s.values)
                return s

            drop_list = []

            for cat in categoricals:
                mask = df.columns.str.contains(f'cat__{cat}_')
                cat_column = df.columns[mask]
                drop_list.append(cat_column[0])
                df[cat_column] = df[cat_column].apply(arg_max_cat, axis=1)

            # display(df)
            # df.to_csv(f'./exports/debug_{instance.method}.csv')

            df_argmaxed = df#.drop(columns=drop_list) # dropping first column is dummy encoding not OHE
            return df_argmaxed
        
        return modified
    
    @_modify
    def _knn2(self, X):
        _, X_missing = self.missing_sampler(pd.DataFrame(X, columns=self.X_original.columns), self.missing, None)
        
        # KNNImputer removes features that has all missing
        # KNNImputer(keep_empty_features=True) in sklearn 1.2 added a parameter
        # However, we are using sklearn 1.0.x and cant upgrade due to depencency contraints
        # so we have to do it ourselves
        empty_cols = X_missing.columns[X_missing.isna().all(axis=0)].values
        X_missing.loc[:, empty_cols] = 0

        X_imputed = self.in_sample_knn_imputer.transform(X_missing)
        X1 = torch.from_numpy(X_imputed).to(X.device)
        
        # if X.shape[1] != X1.shape[1]:
        #     print(X)
        #     print(X_missing)
        #     print(X1)
        
        return X1
    
    @_modify
    def _knn1(self, X):
        _, X_missing = self.missing_sampler(pd.DataFrame(X), self.missing, None)
        
        # KNNImputer removes features that has all missing
        # KNNImputer(keep_empty_features=True) in sklearn 1.2 added a parameter
        # However, we are using sklearn 1.0.x and cant upgrade due to depencency contraints
        # so we have to do it ourselves
        empty_cols = X_missing.columns[X_missing.isna().all(axis=0)].values
        X_missing.loc[:, empty_cols] = 0
        
        knn_imputer = KNNImputer()
        knn_imputer.fit(pd.DataFrame(X))
        X_imputed = knn_imputer.transform(X_missing)
        X1 = torch.from_numpy(X_imputed).to(X.device)
        
        # if X.shape[1] != X1.shape[1]:
        #     print(X)
        #     print(X_missing)
        #     print(X1)
        
        return X1
        
    @_modify
    def _knn(self, X):
        _, X_missing = self.missing_sampler(X, self.missing, None)
        
        # KNNImputer removes features that has all missing
        # KNNImputer(keep_empty_features=True) in sklearn 1.2 added a parameter
        # However, we are using sklearn 1.0.x and cant upgrade due to depencency contraints
        # so we have to do it ourselves
        '''
        empty_cols = X_missing.columns[X_missing.isna().all(axis=0)].values
        X_missing.loc[:, empty_cols] = 0
        
        knn_imputer = KNNImputer()
        X_imputed = knn_imputer.fit_transform(X_missing)
        X1 = torch.from_numpy(X_imputed).to(X.device)
        '''
        # if X.shape[1] != X1.shape[1]:
        #     print(X)
        #     print(X_missing)
        #     print(X1)
        
        return X_missing
    
    @_modify
    def _mice(self, X):
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
        X1 = torch.from_numpy(X_imputed.values).to(X.device)
        return X1
    
    @_modify
    def _knn_test(self, X0):
        ''' 
        replace c*d random select columns for with another random row
        do this for each rows in X0
        where c=corruption_rate and d=number of features
        and X0 is assumed to be unnormalized
        '''
        X = torch.clone(X0)
        mask = self._get_mask(X)
        row_idx, col_idx = torch.where(mask==1)
        X[row_idx, col_idx] = torch.nan

        _, X_missing = self.missing_sampler(pd.DataFrame(X), self.missing, None)

        # KNNImputer removes features that has all missing
        # KNNImputer(keep_empty_features=True) in sklearn 1.2 added a parameter
        # However, we are using sklearn 1.0.x and cant upgrade due to depencency contraints
        # so we have to do it ourselves
        empty_cols = X_missing.columns[X_missing.isna().all(axis=0)].values
        X_missing.loc[:, empty_cols] = 0
        
        knn_imputer = KNNImputer()
        X_imputed = knn_imputer.fit_transform(X_missing)
        X1 = torch.from_numpy(X_imputed).to(X.device)
        
        return X1
    
    @_modify
    def _mice_test(self, X0):
        ''' 
        replace c*d random select columns for with another random row
        do this for each rows in X0
        where c=corruption_rate and d=number of features
        and X0 is assumed to be unnormalized
        '''
        X = torch.clone(X0)
        mask = self._get_mask(X)
        row_idx, col_idx = torch.where(mask==1)
        X[row_idx, col_idx] = torch.nan

        _, X_missing = self.missing_sampler(pd.DataFrame(X), self.missing, None)

        # KNNImputer removes features that has all missing
        # KNNImputer(keep_empty_features=True) in sklearn 1.2 added a parameter
        # However, we are using sklearn 1.0.x and cant upgrade due to depencency contraints
        # so we have to do it ourselves
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
        X1 = torch.from_numpy(X_imputed.values).to(X.device)
        return X1
    
    @_modify
    def _draw_feature(self, X0):
        ''' 
        sample from feature's marginal distribution
        select c*d random columns and replace their values from feature's value pool
        do this for each rows in X0
        where c=corruption_rate and d=number of features
        and X0 is assumed to be unnormalized
        '''
        X = torch.clone(X0)
        mask = self._get_mask(X)
        
        # select random rows for each row (can have same row idx)
        original = torch.from_numpy(self.X_original.values)
        noise_values = torch.zeros_like(X).to(X.device)

        for j in torch.arange(X.shape[1]):
            r = torch.randint(original.shape[0],(X.shape[0],))
            noise_values[:, j] = original[r, j]

        # return (1-mask)*X + mask*imputted
        real = X.mul(1-mask)
        draws = noise_values.mul(mask)

        return real + draws
    
    @_modify
    def _mixup_draw_features(self, X0):
        ''' 
        sample from feature's marginal distribution
        select c*d random columns and replace their values from feature's value pool
        do this for each rows in X0
        where c=corruption_rate and d=number of features
        and X0 is assumed to be unnormalized
        '''
        X = torch.clone(X0)
        mask = self._get_mask(X)
        
        # select random rows for each row (can have same row idx)
        original = torch.from_numpy(self.X_original.values)
        noise_values = torch.zeros_like(X).to(X.device)

        for j in torch.arange(X.shape[1]):
            r = torch.randint(original.shape[0],(X.shape[0],))
            noise_values[:, j] = original[r, j]

        # return (1-mask)*X + mask*imputted
        real = X.mul(1-mask) + X.mul(mask)*.6 # per sample, get all unmasked features and 60% of masked features
        draws = noise_values.mul(mask)*0.4 # per draw, get 40% of features

        return real + draws
    
    @_modify
    def _mixup_draw(self, X0):
        ''' 
        sample from joint marginal distribution
        replace c*d random select columns for with another random row
        do this for each rows in X0
        where c=corruption_rate and d=number of features
        and X0 is assumed to be unnormalized
        '''
        X = torch.clone(X0)
        mask = self._get_mask(X)
        
        # select random rows for each row (can have same row idx)
        original = torch.from_numpy(self.X_original.values)
        r = torch.randint(original.shape[0],(X.shape[0],))
        noise_values = original[r,:]
        noise_values.to(X.device)

        # return (1-mask)*X + mask*imputted
        real = X.mul(1-mask) + X.mul(mask)*.6 # per sample, get all unmasked features and 60% of masked features
        draws = noise_values.mul(mask)*0.4 # per draw, get 40% of features

        return real + draws
    
    @_modify
    def _cluster_replace(self, X0):
        y = self.batch_cluster_labels
        ''' 
        replace with a sample from same cluster label
        '''
        X = torch.clone(X0)
        # mask = self._get_mask(X)
        
        # select random rows for each row (can have same row idx)
        noise_values = torch.zeros_like(X0)
        original = torch.from_numpy(self.X_original.values)

        # TODO: vectorize it if possible
        for i in torch.arange(X0.shape[0]):
            sample_from = original[self.cluster_labels == y[i], :]
            
            r = torch.randint(sample_from.shape[0],(1,1))
            noise_values[i, :] = sample_from[r, :]
        noise_values.to(X.device)

        # # return (1-mask)*X + mask*imputted
        # real = X.mul(1-mask)
        # draws = noise_values.mul(mask)

        return noise_values
    
    @_modify
    def _across_cluster_replace(self, X0):
        y = self.batch_cluster_labels
        ''' 
        replace with a sample from same cluster label
        '''
        X = torch.clone(X0)
        # mask = self._get_mask(X)
        
        # select random rows for each row (can have same row idx)
        noise_values = torch.zeros_like(X0)
        original = torch.from_numpy(self.X_original.values)

        # TODO: vectorize it if possible
        for i in torch.arange(X0.shape[0]):
            sample_from = original[self.cluster_labels != y[i], :]
            
            r = torch.randint(sample_from.shape[0],(1,1))
            noise_values[i, :] = sample_from[r, :]
        noise_values.to(X.device)

        # # return (1-mask)*X + mask*imputted
        # real = X.mul(1-mask)
        # draws = noise_values.mul(mask)

        return noise_values
    
    @_modify
    def _cluster_draw(self, X0):
        y = self.batch_cluster_labels
        ''' 
        same as draw but makes sure sample of different cluster label is selected
        '''
        X = torch.clone(X0)
        mask = self._get_mask(X)
        
        # select random rows for each row (can have same row idx)
        noise_values = torch.zeros_like(X0)
        original = torch.from_numpy(self.X_original.values)

        # TODO: vectorize it if possible
        for i in torch.arange(X0.shape[0]):
            sample_from = original[self.cluster_labels != y[i], :]
            
            r = torch.randint(sample_from.shape[0],(1,1))
            noise_values[i, :] = sample_from[r, :]
        noise_values.to(X.device)

        # return (1-mask)*X + mask*imputted
        real = X.mul(1-mask)
        draws = noise_values.mul(mask)

        return real + draws
    
    @_modify
    def _cluster_draw_feature(self, X0):
        ''' 
        samples as draw_features but pool from different clusters
        '''
        y = self.batch_cluster_labels
        # print(X0.shape, y.shape)

        X = torch.clone(X0)
        mask = self._get_mask(X)
        
        # select random rows for each row (can have same row idx)
        original = torch.from_numpy(self.X_original.values)
        noise_values = torch.zeros_like(X).to(X.device)

        # # TODO: vectorize it if possible
        # for i in torch.arange(X.shape[0]):
        #     for j in torch.arange(X.shape[1]):
        #         sample_from = original[self.cluster_labels != y[i], :]
        #         r = torch.randint(sample_from.shape[0],(1,1))
        #         noise_values[i, j] = sample_from[r, j]
                
                
        for c in self.clusters:
            i = (y == c).nonzero(as_tuple=True)[0] # get all row index where cluster == c;
            sample_from = original[self.cluster_labels != c, :] # we want to draw sample that are not in the same cluster
            
            
            # print('i-shape', i.shape, 'sample-shape', sample_from.shape)
            
            for j in torch.arange(X.shape[1]):
                # for each feature/column draw samples
                r = torch.randint(sample_from.shape[0],(i.shape[0],))
                # print('r-shape', r.shape)
                noise_values[i, j] = sample_from[r, j]
            

        # return (1-mask)*X + mask*imputted
        real = X.mul(1-mask)
        draws = noise_values.mul(mask)

        return real + draws

    @_modify
    def _cluster_draw1(self, X0):
        y = self.batch_cluster_labels
        ''' 
        same as draw but makes sure sample of same cluster label is selected
        '''
        X = torch.clone(X0)
        mask = self._get_mask(X)
        
        # select random rows for each row (can have same row idx)
        noise_values = torch.zeros_like(X0)
        original = torch.from_numpy(self.X_original.values)

        # TODO: vectorize it if possible
        for i in torch.arange(X0.shape[0]):
            # sample_from = original[self.cluster_labels != y[i], :]
            sample_from = original[self.cluster_labels == y[i], :]
            
            r = torch.randint(sample_from.shape[0],(1,1))
            noise_values[i, :] = sample_from[r, :]
        noise_values.to(X.device)

        # return (1-mask)*X + mask*imputted
        real = X.mul(1-mask)
        draws = noise_values.mul(mask)

        return real + draws
    
    @_modify
    def _cluster_draw_feature1(self, X0):
        ''' 
        samples as draw_features but pool from same clusters
        '''
        y = self.batch_cluster_labels
        # print(X0.shape, y.shape)

        X = torch.clone(X0)
        mask = self._get_mask(X)
        
        # select random rows for each row (can have same row idx)
        original = torch.from_numpy(self.X_original.values)
        noise_values = torch.zeros_like(X).to(X.device)

        # # TODO: vectorize it if possible
        # for i in torch.arange(X.shape[0]):
        #     for j in torch.arange(X.shape[1]):
        #         sample_from = original[self.cluster_labels != y[i], :]
        #         r = torch.randint(sample_from.shape[0],(1,1))
        #         noise_values[i, j] = sample_from[r, j]
                
                
        for c in self.clusters:
            i = (y == c).nonzero(as_tuple=True)[0] # get all row index where cluster == c;
            # sample_from = original[self.cluster_labels != c, :] # we want to draw sample that are not in the same cluster
            sample_from = original[self.cluster_labels == c, :] # we want to draw sample that are in the same cluster
            
            
            # print('i-shape', i.shape, 'sample-shape', sample_from.shape)
            
            for j in torch.arange(X.shape[1]):
                # for each feature/column draw samples
                r = torch.randint(sample_from.shape[0],(i.shape[0],))
                # print('r-shape', r.shape)
                noise_values[i, j] = sample_from[r, j]
            

        # return (1-mask)*X + mask*imputted
        real = X.mul(1-mask)
        draws = noise_values.mul(mask)

        return real + draws
    
    @_modify
    def _draw(self, X0):
        ''' 
        sample from joint marginal distribution
        replace c*d random select columns for with another random row
        do this for each rows in X0
        where c=corruption_rate and d=number of features
        and X0 is assumed to be unnormalized
        '''
        X = torch.clone(X0)
        mask = self._get_mask(X)
        
        # select random rows for each row (can have same row idx)
        original = torch.from_numpy(self.X_original.values)
        r = torch.randint(original.shape[0],(X.shape[0],))
        noise_values = original[r,:]
        noise_values.to(X.device)

        # return (1-mask)*X + mask*imputted
        real = X.mul(1-mask)
        draws = noise_values.mul(mask)

        return real + draws
    
    @_modify
    def _draw2(self, X0):
        '''
        same as draw but with mcar/mnar/mar type mask
        '''
        X = torch.clone(X0)
        
        _, X_missing = self.missing_sampler(pd.DataFrame(X), self.missing, None)
        
        # KNNImputer removes features that has all missing
        # KNNImputer(keep_empty_features=True) in sklearn 1.2 added a parameter
        # However, we are using sklearn 1.0.x and cant upgrade due to depencency contraints
        # so we have to do it ourselves
        empty_cols = X_missing.columns[X_missing.isna().all(axis=0)].values
        X_missing.loc[:, empty_cols] = 0
        
        mask = X_missing.isna().values.astype(int)
        mask = torch.from_numpy(mask)
        mask = mask.to(X.device)
        
        # select random rows for each row (can have same row idx)
        original = torch.from_numpy(self.X_original.values)
        r = torch.randint(original.shape[0],(X.shape[0],))
        noise_values = original[r,:]
        noise_values.to(X.device)

        # return (1-mask)*X + mask*imputted
        real = X.mul(1-mask)
        draws = noise_values.mul(mask)

        return real + draws
    
    @_modify
    def _noise(self, X0, mean=0, std=1):
        ''' 
        add gaussian noise, N(mu, std) to c*d random columns for all rows
        where c=corruption_rate and d=number of features    
        '''
        X = torch.clone(X0)

        if mean==0 and std==0: return X

        mask = self._get_mask(X)

        noise_values = torch.empty_like(X).normal_(mean, std)
        # noise_values = noise_values if device<0 else noise_values.to(device)
        noise_values = noise_values.to(X.device)

        # debug_mode and print(noise_values.shape)
        noise = noise_values.mul(mask)

        return X+noise
    
    @_modify
    def _sample(self, X0):
        ''' 
        replace c*d random columns for all rows using original feature distribution
        where c=corruption_rate and d=number of features
        and X0 is assumed to be unnormalized
        '''
        X = torch.clone(X0)
        mask = self._get_mask(X)

        # note all categories are converted to ordinal (long) for working with tensors
        # if df without OHE is passed it will converted to float
        # however non-OHE df is done for SCARF draw
        # other methods use OHEncoded df
        # We need float to work with some tensor functions
        original = torch.from_numpy(self.X_original.values).float()
        means = torch.mean(original, dim=0)
        stdevs = torch.std(original, dim=0)
        
        noise_values = torch.cat([
            torch.empty_like(X[:,i]).normal_(m.item(), s.item()) 
            for i,(m,s) in enumerate(zip(means, stdevs))
        ], dim=-1)

        noise_values = noise_values.reshape(X.shape).contiguous()
        noise_values = noise_values.to(X.device)

        # return (1-mask)*X + mask*imputted
        real = X.mul(1-mask)
        imputed = noise_values.mul(mask)

        return real + imputed

    @_modify
    def _pass(self, X):
        return X
    
    def __call__(self, X):
        
        method_map = {
            'pass': self._pass,
            'noise': self._noise,
            'sample': self._sample,
            'draw-feature': self._draw_feature,
            'draw': self._draw,
            'draw2': self._draw2,
            'knn': self._knn,
            'mice': self._mice,
            'knn1': self._knn1,
            'knn2': self._knn2,
            'knn-test': self._knn_test,
            'mice-test': self._mice_test,
            'mixup-draw': self._mixup_draw,
            'mixup-draw-features': self._mixup_draw_features,
            'cluster-draw': self._cluster_draw,
            'cluster-draw-feature': self._cluster_draw_feature,
            'cluster-draw1': self._cluster_draw1,
            'cluster-draw-feature1': self._cluster_draw_feature1,
            'cluster-replace': self._cluster_replace,
            'across-cluster-replace': self._across_cluster_replace,
        }
        
        return method_map[self.method](X)
    

