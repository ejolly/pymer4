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
    assert model.sig_type == 'permutation (500)'

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

def test_post_hoc():
    np.random.seed(1)
    df = pd.read_csv(os.path.join(get_resource_path(),'sample_data.csv'))
    df['DV_l2'] = np.random.randint(2,size=df.shape[0])
    model = Lmer('DV ~ IV1*IV3*DV_l*DV_l2 + (IV1|Group)',data=df,family='gaussian')
    model.fit(factors={
    'IV3': ['0.5','1.0','1.5'],
    'DV_l2': ['1','0'],
    'DV_l': ['0','1']
    },summarize=False)

    marginal,contrasts = model.post_hoc(marginal_vars='IV3',p_adjust='dunnet')
    assert marginal.shape[0] == 3
    assert contrasts.shape[0] == 3

    marginal,contrasts = model.post_hoc(marginal_vars = ['IV3','DV_l','DV_l2'])
    assert marginal.shape[0] == 12
    assert contrasts.shape[0] == 66

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

def test_anova():

    np.random.seed(1)
    data = pd.read_csv(os.path.join(get_resource_path(),'sample_data.csv'))
    data['DV_l2'] = np.random.randint(0,4,data.shape[0])
    model = Lmer('DV ~ IV3*DV_l2 + (IV3|Group)',data=data)

def test_poisson_lmm():
    np.random.seed(1)
    df = pd.read_csv(os.path.join(get_resource_path(),'sample_data.csv'))
    df['DV_int'] = np.random.randint(1,10,df.shape[0])
    m = Lmer('DV_int ~ IV3 + (1|Group)',data=df,family='poisson')
    m.fit(summarize=False)
    assert m.family == 'poisson'
    res = np.array([[ 1.64640281e+000,  1.54854277e+000,  1.74426286e+000,
         4.99295118e-002,  3.29745426e+001,  1.88245501e-238],
       [-5.40993372e-002, -1.45285282e-001,  3.70866072e-002,
         4.65242959e-002, -1.16281904e+000,  2.44902930e-001]])
    assert np.allclose(m.coefs.iloc[:,:-1].values, res)
    assert m.coefs.shape == (2,7)
    assert 'Z-stat' in m.coefs.columns

def test_gamma_lmm():

    np.random.seed(1)
    df = pd.read_csv(os.path.join(get_resource_path(),'sample_data.csv'))
    df['DV_g'] = np.random.uniform(1,2,size=df.shape[0])
    m = Lmer('DV_g ~ IV3 + (1|Group)',data=df,family='gamma')
    m.fit(summarize=False)
    assert m.family == 'gamma'
    res = np.array([[ 6.73406340e-01,  6.42966702e-01,  7.03845977e-01,
         1.55307127e-02,  4.33596548e+01,  0.00000000e+00],
       [-6.90225926e-03, -3.33536803e-02,  1.95491618e-02,
         1.34958710e-02, -5.11434888e-01,  6.09046570e-01]])
    assert np.allclose(m.coefs.iloc[:,:-1].values,res)
    assert m.coefs.shape == (2,7)


def test_inverse_gaussian_lmm():

    np.random.seed(1)
    df = pd.read_csv(os.path.join(get_resource_path(),'sample_data.csv'))
    df['DV_g'] = np.random.uniform(1,2,size=df.shape[0])
    m = Lmer('DV_g ~ IV3 + (1|Group)',data=df,family='inverse_gaussian')
    m.fit(summarize=False)
    assert m.family == 'inverse_gaussian'
    res = np.array([[ 4.54086509e-001,  4.12691749e-001,  4.95481268e-001,
         2.11201633e-002,  2.15001419e+001,  1.55212497e-102],
       [-9.15658032e-003, -4.51946155e-002,  2.68814549e-002,
         1.83870905e-002, -4.97989625e-001,  6.18491355e-001]])
    assert np.allclose(m.coefs.iloc[:,:-1].values,res)
    assert m.coefs.shape == (2,7)
