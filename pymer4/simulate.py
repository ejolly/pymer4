from __future__ import division
import numpy as np
import pandas as pd
from scipy.spatial.distance import squareform
from pymer4.utils import discrete_inverse_logit, isPSD, nearestPSD

__all__  = ['easy_multivariate_normal',
            'simulate_lm',
            'simulate_lmm']

__author__ = ['Eshin Jolly']
__license__ = "MIT"

def simulate_lm(num_obs,num_coef,coef_vals=None,corrs=None,mus=0.0,sigmas=1.0,noise_params=(0,1),family='gaussian',seed=None):
    """
    Function to quickly simulate a regression model dataset, with continuous predictors.
    Provided a number of observations, number of coefficients, and optionally correlations between predictors, means, and standard deviations of predictors, returns a pandas dataframe with simulated data that can be used to estimate a linear regression using Lm(). Using the family='binomial' argument can generate discrete dependent variable values for use with logistic regression.

    Defaults to returning standard normal (mu = 0; sigma = 1) predictors with no explicit correlations.

    Args:
        num_obs (int): number of total observations, i.e. rows of data
        num_coef (int): number of coefficients/regressors, i.e. columns of data
        coef_vals (list,optional): "true" values of coefficients to generate data. If not provided will be randomly generated. Must include a coefficient for the intercept as well (i.e. mean of data)
        corrs (ndarray,list,float): correlations between coefficients provided as 2d num_coef x num_coef, 1d flattend numpy array/list of length (num_features * (num_features-1)) / 2, or a float to be treated as the same correlation between all coefficients
        mus (float/list/ndarray): means of columns of predictors
        sigmas (float/list/ndarray): stds of columns of predictors
        noise_params (tup, optional): mean and std of noise added to simulated data
        family (str): distribution family for the dependent variable. Currently only 'gaussian' (continuous DV) or 'binomial' (discrete DV) are available.
        seed (int): seed for reproducible random number generation

    Returns:

        data
        ground-truth coefficient values


    """

    if seed is not None:
        np.random.seed(seed)

    if coef_vals is not None:
        if len(coef_vals) - num_coef == 0:
            raise ValueError("Missing one coefficient value. Did you provide a value for the intercept term?")
        else:
            assert len(coef_vals) == num_coef + 1, "Number of coefficient values should be num_coef + 1 (for intercept)"

        b = coef_vals
    else:
        b = np.random.rand(num_coef+1)

    if isinstance(mus,list) or isinstance(mus,np.ndarray):
        assert len(mus) == len(b)-1, "mus must match number of num_coef"
    if isinstance(sigmas,list) or isinstance(sigmas,np.ndarray):
        assert len(sigmas) == len(b)-1, "sigmas must match number of num_coef"
    assert isinstance(noise_params,tuple) and len(noise_params) == 2, "noise_params should be a tuple of (mean,std)"

    # Generate random design matrix
    if corrs is not None:
        X = easy_multivariate_normal(num_obs,num_coef,corrs,mus,sigmas,seed)
    else:
        X = np.random.normal(mus,sigmas,size=(num_obs,num_coef))
    # Add intercept
    X = np.column_stack([np.ones((num_obs,1)),X])
    # Generate data
    Y = np.dot(X,b) + np.random.normal(*noise_params,size=num_obs)
    # Apply transform if not linear model
    if family == 'binomial':
        Y = discrete_inverse_logit(Y)
    dat = pd.DataFrame(np.column_stack([Y,X[:,1:]]),columns=['DV'] + ['IV'+str(elem+1) for elem in range(X.shape[1]-1)])

    return dat,b

def simulate_lmm(num_obs,num_coef,num_grps,coef_vals=None,corrs=None,grp_sigmas=.25,mus=0.0,sigmas=1.0,noise_params=(0,1),family='gaussian',seed=None):
    """
    Function to quickly simulate a multi-level regression model dataset, with continuous predictors.
    Provided a number of observations, number of coefficients, number of groups/clusters,
    and optionally correlations between predictors, means, and standard deviations of predictors,
    returns a pandas dataframe with simulated data that can be used to estimate a multi-level model using Lmer(). Using the family='binomial' argument can generate discrete dependent variable values for use with logistic multi-level models.

    Defaults to returning standard normal (mu = 0; sigma = 1) predictors with no explicit correlations and low variance between
    groups (sigma = .25).

    Args:
        num_obs (int): number of observations per cluster/stratum/group
        num_coef (int): number of coefficients/regressors, i.e. columns of data
        num_grps (int): number of cluster/stratums/groups
        coef_vals (list,optional): "true" values of coefficients to generate data. If not provided will be randomly generated. Must include a coefficient for the intercept as well (i.e. mean of data)
        corrs (ndarray,list,float): correlations between coefficients provided as 2d num_coef x num_coef, 1d flattend numpy array/list of length (num_features * (num_features-1)) / 2, or a float to be treated
        as the same correlation between all coefficients
        grp_sigmas (int or list): grp level std around population coefficient values; can be a single value in which case same std is applied around all coefficients or a list for different std; default .25
        mus (float/list/ndarray): means of columns of predictors
        sigmas (float/list/ndarray): stds of columns of predictors
        noise_params (tup, optional): mean and std of noise added to each group's simulated data
        family (str): distribution family for the dependent variable. Currently only 'gaussian' (continuous DV) or 'binomial' (discrete DV) are available.
        seed (int): seed for reproducible random number generation

    Returns:
        data
        group/cluster level coefficients (i.e. BLUPs)
        population coefficient values

    """

    if seed is not None:
        np.random.seed(seed)

    if coef_vals:
        if len(coef_vals) - num_coef == -1:
            raise ValueError("Missing one coefficient value. Did you provide a value for the intercept term?")
        else:
            assert len(coef_vals) == num_coef + 1

        b = coef_vals
    else:
        b = np.random.rand(num_coef+1)

    assert isinstance(noise_params,tuple) and len(noise_params) == 2, "noise_params should be a tuple of (mean,std)"
    assert isinstance(grp_sigmas,int) or isinstance(grp_sigmas,list) or isinstance(grp_sigmas,float), "grp_sigmas should be scalar value or list"
    if not isinstance(grp_sigmas,list):
        grp_sigmas = [grp_sigmas] * (num_coef + 1)
    else:
        assert len(grp_sigmas) == len(b), "The length of a list of grp_sigmas must match the num_coef plus intercept!"

    if isinstance(mus,list) or isinstance(mus,np.ndarray):
        assert len(mus) == len(b)-1, "mus must match number of num_coef"
    if isinstance(sigmas,list) or isinstance(sigmas,np.ndarray):
        assert len(sigmas) == len(b)-1, "sigmas must match number of num_coef"

    # Generate group paramaters (BLUPs)
    blups = np.array([np.random.normal(est,sigma,num_grps) for est,sigma in zip(b,grp_sigmas)]).T

    # Generate data
    for grp in range(blups.shape[0]):
        # Create a random design matrix per group
        if corrs:
            x = easy_multivariate_normal(num_obs,num_coef,corrs,mus,sigmas,seed)
        else:
            x = np.random.normal(mus,sigmas,size=(num_obs,num_coef))
        x = np.column_stack([np.ones((num_obs,1)),x])
        # Use blups to generate group data
        y = np.dot(x,blups[grp,:]) + np.random.normal(*noise_params,size=num_obs)
        if family == 'binomial':
            y = discrete_inverse_logit(y)
        if grp ==0:
            x_all,y_all = x,y
        else:
            y_all = np.append(y_all,y,axis=0)
            x_all = np.append(x_all,x,axis=0)

    grp_ids = np.array([[elem]*num_obs for elem in range(1,num_grps+1)]).ravel()

    data = pd.DataFrame(np.column_stack([y_all,x_all[:,1:],grp_ids]),columns=['DV'] + ['IV'+str(elem+1) for elem in range(x_all.shape[1]-1)] + ['Group'])
    blups = pd.DataFrame(blups,columns= ['Intercept'] + ['IV'+str(elem+1) for elem in range(x_all.shape[1]-1)],index=['Grp'+str(elem+1) for elem in range(num_grps)])
    return data, blups, b

def easy_multivariate_normal(num_obs, num_features, corrs, mu = 0.0, sigma = 1.0, seed = None,forcePSD=True,return_new_corrs=False,nit=100):
    """
    Function to more easily generate multivariate normal samples provided a correlation matrix or list of correlations (upper triangle of correlation matrix) instead of a covariance matrix. Defaults to returning approximately standard normal (mu = 0; sigma = 1) variates. Unlike numpy, if the desired correlation matrix is not positive-semi-definite, will by default issue a warning and find the nearest PSD correlation matrix and generate data with this matrix. This new matrix can optionally be returned used the return_new_corrs argument.

    Args:
        num_obs (int): number of observations/samples to generate (rows)
        corrs (ndarray/list/float): num_features x num_features 2d array, flattend numpy array of length (num_features * (num_features-1)) / 2, or scalar for same correlation on all off-diagonals
        num_features (int): number of features/variables/dimensions to generate (columns)
        mu (float/list): mean of each feature across observations; default 0.0
        sigma (float/list): sd of each feature across observations; default 1.0
        forcePD (bool): whether to find and use a new correlation matrix if the requested one is not positive semi-definite; default False
        return_new_corrs (bool): return the nearest correlation matrix that is positive semi-definite used to generate data; default False
        nit (int): number of iterations to search for the nearest positive-semi-definite correlation matrix is the requested correlation matrix is not PSD; default 100

    Returns:
        ndarray: correlated data as num_obs x num_features array
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

    if not isPSD(corrs):
        if forcePSD:
            # Tell user their correlations are being recomputed if they didnt ask to save them as they might not realize
            if not return_new_corrs:
                print("Correlation matrix is not positive semi-definite. Solved for new correlation matrix.")
            _corrs = np.array(nearestPSD(corrs, nit))

        else:
            raise ValueError("Correlation matrix is not positive semi-definite. Pymer4 will not generate inaccurate multivariate data. Use the forcePD argument to automatically solve for the closest desired correlation matrix.")
    else:
        _corrs = corrs

    #Rescale correlation matrix by variances, given standard deviations of features
    sd = np.diag(sigma)
    #R * Vars = R * SD * SD
    cov = _corrs.dot(sd.dot(sd))
    X = np.random.multivariate_normal(mu, cov, size = num_obs)

    if return_new_corrs:
        return X, _corrs
    else:
        return X
