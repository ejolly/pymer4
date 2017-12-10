__all__  = ['get_resource_path','_sig_stars','_robust_estimator']

__author__ = ['Eshin Jolly']
__license__ = "MIT"

from os.path import dirname,join, sep
import numpy as np

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

def _robust_estimator(vals,X,robust_estimator='hc0',nlags=1):
    """
    Computes robust sandwich estimators for standard errors used in OLS computation. Types include:
    'hc0': Huber (1980) sandwich estimator to return robust standard error estimates.
    'hc3': MacKinnon and White (1985) HC3 sandwich estimator. Provides more robustness in smaller samples than HC0 Long & Ervin (2000)
    'hac': Newey-West (1987) estimator for robustness to heteroscedasticity as well as serial auto-correlation at given lags.

    Args:
        vals (np.ndarray): 1d array of residuals
        X (np.ndarray): design matrix used in OLS, e.g. Brain_Data().X
        robust_estimator (str): estimator type, 'hc0' (default), 'hc3', or 'hac'
        nlags (int): number of lags, only used with 'hac' estimator, default is 1

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
        weights = 1 - np.arange(nlags+1.)/(nlags+1.)

        #First compute lag 0
        V = np.diag(vals**2)
        meat = weights[0] * np.dot(np.dot(X.T,V),X)

        #Now loop over additional lags
        for l in range(1, nlags+1):

            V = np.diag(vals[l:] * vals[:-l])
            meat_1 = np.dot(np.dot(X[l:].T,V),X[:-l])
            meat_2 = np.dot(np.dot(X[:-l].T,V),X[l:])

            meat += weights[l] * (meat_1 + meat_2)

    # Then we make a sandwich
    vcv = np.dot(np.dot(bread,meat),bread)

    return np.sqrt(np.diag(vcv))
