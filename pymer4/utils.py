from __future__ import division

__all__  = ['get_resource_path',
            '_sig_stars',
            '_robust_estimator',
            '_chunk_boot_ols_coefs',
            '_chunk_perm_ols',
            '_ols',
            '_perm_find']

__author__ = ['Eshin Jolly']
__license__ = "MIT"

from os.path import dirname,join, sep
import numpy as np
from patsy import dmatrices


def get_resource_path():
    """ Get path sample data directory. """
    return join(dirname(__file__), 'resources') + sep

def _sig_stars(val):
    """Adds sig stars to coef table prettier output."""
    star = ''
    if 0 <= val < .001:
        star = '***'
    elif .001 <= val < 0.01:
        star = '**'
    elif .01 <= val < .05:
        star = '*'
    elif .05 <= val < .1:
        star = '.'
    return star

def _robust_estimator(vals,X,robust_estimator='hc0',n_lags=1):
    """
    Computes robust sandwich estimators for standard errors used in OLS computation. Types include:
    'hc0': Huber (1980) sandwich estimator to return robust standard error estimates.
    'hc3': MacKinnon and White (1985) HC3 sandwich estimator. Provides more robustness in smaller samples than HC0 Long & Ervin (2000)
    'hac': Newey-West (1987) estimator for robustness to heteroscedasticity as well as serial auto-correlation at given lags.

    Args:
        vals (np.ndarray): 1d array of residuals
        X (np.ndarray): design matrix used in OLS, e.g. Brain_Data().X
        robust_estimator (str): estimator type, 'hc0' (default), 'hc3', or 'hac'
        n_lags (int): number of lags, only used with 'hac' estimator, default is 1

    Returns:
        stderr (np.ndarray): 1d array of standard errors with length == X.shape[1]

    """

    assert robust_estimator in ['hc0','hc3','hac'], "robust_estimator must be one of hc0, hc3 or hac"

    # Make a sandwich!
    # First we need bread
    bread = np.linalg.pinv(np.dot(X.T,X))

    # Then we need meat
    if robust_estimator == 'hc0':
        V = np.diag(vals**2)
        meat = np.dot(np.dot(X.T,V),X)

    elif robust_estimator == 'hc3':
        V = np.diag(vals**2)/(1-np.diag(np.dot(X,np.dot(bread,X.T))))**2
        meat = np.dot(np.dot(X.T,V),X)

    elif robust_estimator == 'hac':
        weights = 1 - np.arange(n_lags+1.)/(n_lags+1.)

        #First compute lag 0
        V = np.diag(vals**2)
        meat = weights[0] * np.dot(np.dot(X.T,V),X)

        #Now loop over additional lags
        for l in range(1, n_lags+1):

            V = np.diag(vals[l:] * vals[:-l])
            meat_1 = np.dot(np.dot(X[l:].T,V),X[:-l])
            meat_2 = np.dot(np.dot(X[:-l].T,V),X[l:])

            meat += weights[l] * (meat_1 + meat_2)

    # Then we make a sandwich
    vcv = np.dot(np.dot(bread,meat),bread)

    return np.sqrt(np.diag(vcv))

def _ols(x,y,robust,n_lags,all_stats=True):
    """
    Compute OLS on data given formula. Useful for single computation and within permutation schemes.
    """

    # Expects as input pandas series and dataframe
    Y,X = y.values.squeeze(), x.values

    # The good stuff
    b = np.dot(np.linalg.pinv(X),Y)

    if all_stats:

        res = Y - np.dot(X,b)

        if robust:
            se = _robust_estimator(res,X,robust_estimator=robust,n_lags=n_lags)
        else:
            sigma = np.std(res,axis=0,ddof=X.shape[1])
            se = np.sqrt(np.diag(np.linalg.pinv(np.dot(X.T,X)))) * sigma

        t = b / se

        return b, se, t, res

    else:
        return b

def _chunk_perm_ols(x,y,robust,n_lags,seed):
    """
    Permuted OLS chunk.
    """
    # Shuffle y labels
    y = y.sample(frac=1,replace=False,random_state=seed)
    b, s, t, res = _ols(x,y,robust,n_lags,all_stats=True)

    return list(t)

def _chunk_boot_ols_coefs(dat,formula,seed):
    """
    OLS computation of coefficients to be used in a parallelization context.
    """
    # Random sample with replacement from all data
    dat = dat.sample(frac=1,replace=True,random_state=seed)
    y,x = dmatrices(formula,dat,1,return_type='dataframe')
    b = _ols(x,y,robust=None,n_lags=1,all_stats=False)
    return list(b)

def _perm_find(arr,x):
    """
    Find permutation cutoff in array.
    """
    return np.sum(arr >= np.abs(x))/float(len(arr))
