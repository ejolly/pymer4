from __future__ import division
import numpy as np
import pandas as pd
from scipy.spatial.distance import squareform

__all__  = ['easy_multivariate_normal',
            'simulate_lm']

__author__ = ['Eshin Jolly']
__license__ = "MIT"

def simulate_lm(num_obs,num_coef,coef_vals=None,corrs=None,mus=0.0,sigmas=1.0,noise_params=(0,1),seed=None):
    """
    Function to quickly simulate a regression model dataset, with continuous predictors.
    Provided a number of observations, number of coefficients, and optionally correlations between predictors, means, and standard deviations of predictors, returns a pandas dataframe with simulated data that can be used to estimate a linear regression using Lm().

    Defaults to returning standard normal (mu = 0; sigma = 1) predictors with no explicit correlations.

    Args:
        num_obs (int): number of total observations, i.e. rows of data
        num_coef (int): number of coefficients/regressors, i.e. columns of data
        coef_vals (list,optional): "true" values of coefficients to generate data.
        If not provided will be randomly generated. Must include a coefficient for the
        intercept as well (i.e. mean of data)
        corrs (ndarray,list,float): correlations between coefficients provided as 2d num_coef x num_coef,
        1d flattend numpy array/list of length (num_features * (num_features-1)) / 2, or a float to be treated
        as the same correlation between all coefficients
        mus (float/list/ndarray): means of columns of predictors
        sigmas (float/list/ndarray): stds of columns of predictors
        noise_params (tup, optional): mean and std of noise added to simulated data
        seed (int): seed for reproducible random number generation

    Returns:
        pandas.core.frame.DataFrame: num_obs x num_coef dataframe

    """

    if seed is not None:
        np.random.seed(seed)
    if coef_vals:
        if len(coef_vals) - num_coef == 0:
            raise ValueError("Missing one coefficient value. Did you provide a value for the intercept term?")
        else:
            assert len(coef_vals) == num_coef + 1, "Number of coefficient values should be num_coef + 1 (for intercept)"

        b = coef_vals
    else:
        b = np.random.rand(num_coef+1)

    assert isinstance(noise_params,tuple) and len(noise_params) == 2, "noise_params should be a tuple of (mean,std)"

    # Generate random design matrix
    if corrs:
        X = easy_multivariate_normal(num_coef,num_obs,corrs,mus,sigmas,seed)
    else:
        X = np.random.randn(num_obs,num_coef)
    # Add intercept
    X = np.column_stack([np.ones((num_obs,1)),X])
    # Generate data
    Y = np.dot(X,b) + np.random.normal(*noise_params,size=num_obs)

    dat = pd.DataFrame(np.column_stack([Y,X[:,1:]]),columns=['DV'] + ['IV'+str(elem+1) for elem in range(X.shape[1]-1)])

    return dat,b

def easy_multivariate_normal(num_features, num_obs, corrs, mu = 0.0, sigma = 1.0, seed = None):
    """
    Function to more easily generate multivariate normal samples provided a correlation matrix or list of correlations (upper triangle of correlation matrix) instead of a covariance matrix.
    Defaults to returning approximately standard normal (mu = 0; sigma = 1) variates.

    Args:
        num_features (int): number of features/variables/dimensions to generate (columns)
        num_obs (int): number of observations/samples to generate (rows)
        corrs (ndarray): num_features x num_features or flattend numpy array of length (num_features * (num_features-1)) / 2
        mu (float/list): mean of each feature across observations; default 0.0
        sigma (float/list): sd of each feature across observations; default 1.0

    Returns:
        X (ndarray): num_obs x num_features data matrix with correlated columns
    """

    if seed is not None:
        np.random.seed(seed)

    if isinstance(mu, list):
        assert len(mu) == num_features, "Number of means must match number of features"
    else:
        mu = [mu] * num_features
    if isinstance(sigma, list):
        assert len(sigma) == num_features, "Number of sds must match number of features"
    else:
        sigma = [sigma] * num_features


    if isinstance(corrs,np.ndarray) and corrs.ndim == 2:
        assert corrs.shape[0] == corrs.shape[1] and np.allclose(corrs,corrs.T) and np.allclose(np.diagonal(corrs),np.ones_like(np.diagonal(corrs))), "Correlation matrix must be square symmetric"
    elif (isinstance(corrs,np.ndarray) and corrs.ndim == 1) or isinstance(corrs,list):
        assert len(corrs) == (num_features * (num_features-1)) / 2, "(num_features * (num_features - 1) / 2) correlation values are required for a flattened array or list"
        corrs = squareform(corrs)
        np.fill_diagonal(corrs,1.0)
    elif isinstance(corrs,float):
        corrs = np.array([corrs] * int(((num_features * (num_features-1)) / 2)))
        corrs = squareform(corrs)
        np.fill_diagonal(corrs,1.0)
    else:
        raise ValueError("Correlations must be num_features x num_feature, flattend numpy array/list or scalar")


    #Rescale correlation matrix by variances, given standard deviations of features
    sd = np.diag(sigma)
    #R * Vars = R * SD * SD
    cov = corrs.dot(sd.dot(sd))

    X = np.random.multivariate_normal(mu, cov, size = num_obs)

    return X
