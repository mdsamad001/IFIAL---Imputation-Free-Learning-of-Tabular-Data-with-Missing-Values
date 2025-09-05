import torch
import numpy as np
from functools import partial

class MyBool:
    def __init__(self, x:bool):
        self.x = x
    
    def __bool__(self):
        return self.x
    
    def __repr__(self):
        return f"{self.x}"
    
    def set_value(self, x):
        self.x = x

debug_mode = MyBool(False)

def set_debug_mode(mode):
    debug_mode.set_value(mode)


def get_mask(X, corruption_rate):
    '''
    get mask (0 or 1) of X, 
    where c*d random columns for each rows are set to 1
    where c=corruption_rate and d=number of features 
    
    TODO: implement without for-loop
    '''
    n,d = X.shape
    debug_mode and print(X.shape)
    d_corrupt = int(corruption_rate * d)
    x = np.zeros((n,d))

    for i in range(n):
        a = np.arange(1,d+1)
        a1 = np.random.permutation(a)
        x[i,:] = a1
        
    mask = np.where(x<=d_corrupt, 1, 0)
    
    device = X.get_device()
    mask = torch.from_numpy(mask)

    return mask if device<0 else mask.to(device)



def add_noise(X0, corruption_rate=0.6, mu=0, std=0):
    ''' 
    add gaussian noise, N(mu, std) to c*d random columns for all rows
    where c=corruption_rate and d=number of features    
    '''
    X = torch.clone(X0)
    
    if mu==0 and std==0: return X
    
    device = X.get_device()
    mask = get_mask(X, corruption_rate)
    debug_mode and print(mask.shape)
    
    noise_values = torch.empty_like(X).normal_(mu, std)
    noise_values = noise_values if device<0 else noise_values.to(device)
    
    debug_mode and print(noise_values.shape)
    noise = noise_values.mul(mask)

    return X+noise


def corrupt_by_sampling(X0, corruption_rate=0.6):
    ''' 
    replace c*d random columns for all rows using feature distribution
    where c=corruption_rate and d=number of features
    and X0 is assumed to be unnormalized
    '''
    X = torch.clone(X0)
    device = X.get_device()
    mask = get_mask(X, corruption_rate)
    debug_mode and print('mask shape', mask.shape)
    
    # pytorch has no equivalent functions for np.nanstd
    X_np = X.cpu().numpy()
    means = np.nanmean(X_np, axis=0)
    stdevs = np.nanstd(X_np, axis=0)
    
    debug_mode and print('means', means.shape, 'stdevs', stdevs.shape)
    noise_values = torch.cat([
        torch.empty_like(X[:,i]).normal_(m.item(), s.item()) 
        for i,(m,s) in enumerate(zip(means, stdevs))
    ], dim=-1)
    debug_mode and print('before reshape; noise_values', noise_values)
    debug_mode and print(noise_values.shape)
    
    noise_values = noise_values.reshape(X.shape).contiguous()
    debug_mode and print('noise_values', noise_values)
    debug_mode and print(noise_values.shape)
    
    noise_values = noise_values if device<0 else noise_values.to(device)
    
    # return (1-mask)*X + mask*imputted
    real = X.mul(1-mask)
    imputed = noise_values.mul(mask)

    return real + imputed

def corrupt_by_drawing(X0, corruption_rate=0.6):
    ''' 
    replace c*d random columns for all rows using feature distribution
    where c=corruption_rate and d=number of features
    and X0 is assumed to be unnormalized
    '''
    X = torch.clone(X0)
    device = X.get_device()
    mask = get_mask(X, corruption_rate)
    debug_mode and print('mask shape', mask.shape)
    
    ## shuffle 
    # r = torch.randperm(X.shape[0])
    # random rows
    r = torch.randint(X.shape[0],(X.shape[0],))
    noise_values = X[r,:]
    
    
    # random 
    r = torch.randint(X.shape[0],(10,))
    
    # return (1-mask)*X + mask*imputted
    real = X.mul(1-mask)
    draws = noise_values.mul(mask)

    return real + draws

