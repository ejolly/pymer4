from __future__ import division
from pymer4.models import Lmer, Lm
from pymer4.utils import get_resource_path
import pandas as pd
import numpy as np
import os
from scipy.special import logit

def test_gaussian_lm():

    df = pd.read_csv(os.path.join(get_resource_path(),'sample_data.csv'))
    model = Lm('DV ~ IV1 + IV3',data=df)
    model.fit(summarize=False)

    assert model.coefs.shape == (3,8)
    estimates = np.array([ 42.24840439,   0.24114414,  -3.34057784])
    assert np.allclose(model.coefs['Estimate'],estimates,atol=.001)

    # Test robust SE against statsmodels
    standard_se = np.array([ 6.83783939,  0.30393886,  3.70656475])
    assert np.allclose(model.coefs['SE'],standard_se,atol=.001)

    hc0_se = np.array([ 7.16661817,  0.31713064,  3.81918182])
    model.fit(robust='hc0',summarize=False)
    assert np.allclose(model.coefs['SE'],hc0_se,atol=.001)

    hc3_se = np.array([ 7.22466699,  0.31971942,  3.84863701])
    model.fit(robust='hc3',summarize=False)
    assert np.allclose(model.coefs['SE'],hc3_se,atol=.001)

    # Test bootstrapping
    model.fit(summarize=False,conf_int='boot')
    assert model.ci_type == 'boot (500)'

    # Test permutation
    model.fit(summarize=False,permute=500)
    assert model.sig_type == 'permute (500)'

def test_gaussian_lmm():

    df = pd.read_csv(os.path.join(get_resource_path(),'sample_data.csv'))
    model = Lmer('DV ~ IV3 + IV2 + (IV2|Group) + (1|IV3)',data=df)
    model.fit(summarize=False)

    assert model.coefs.shape == (3,8)
    estimates = np.array([ 12.04334602,  -1.52947016,   0.67768509])
    assert np.allclose(model.coefs['Estimate'],estimates,atol=.001)

    assert isinstance(model.fixef,list)
    assert model.fixef[0].shape == (47,3)
    assert model.fixef[1].shape == (3,3)

    assert isinstance(model.ranef,list)
    assert model.ranef[0].shape == (47,2)
    assert model.ranef[1].shape == (3,1)

    assert model.ranef_corr.shape == (1,3)
    assert model.ranef_var.shape == (4,3)

    assert np.allclose(model.coefs.loc[:,'Estimate'],model.fixef[0].mean(),atol=.01)

    # Test prediction
    assert np.allclose(model.predict(model.data,use_rfx=True),model.data.fits)

def test_logistic_lmm():

    df = pd.read_csv(os.path.join(get_resource_path(),'sample_data.csv'))
    model = Lmer('DV_l ~ IV1+ (IV1|Group)',data=df,family='binomial')
    model.fit(summarize=False)

    assert model.coefs.shape == (2,13)
    estimates = np.array([-0.16098421,  0.00296261])
    assert np.allclose(model.coefs['Estimate'],estimates,atol=.001)

    assert isinstance(model.fixef,pd.core.frame.DataFrame)
    assert model.fixef.shape == (47,2)

    assert isinstance(model.ranef,pd.core.frame.DataFrame)
    assert model.ranef.shape == (47,2)

    assert np.allclose(model.coefs.loc[:,'Estimate'],model.fixef.mean(),atol=.01)

    # Test prediction
    assert np.allclose(model.predict(model.data,use_rfx=True),model.data.fits)
    assert np.allclose(model.predict(model.data,use_rfx=True,pred_type='link'),logit(model.data.fits))
