from pymer4.models import Lmer
from pymer4.utils import get_resource_path
import pandas as pd
import numpy as np
import os
from scipy.special import logit

def test_linear():

    df = pd.read_csv(os.path.join(get_resource_path(),'sample_data.csv'))
    model = Lmer('DV ~ IV3 + IV2 + (IV2|Group) + (1|IV3)',data=df)
    model.fit(summarize=False)

    assert model.coefs.shape == (3,8)
    estimates = np.array([ 12.04334602,  -1.52947016,   0.67768509])
    assert np.allclose(model.coefs['Estimate'],estimates)

    assert isinstance(model.fixef,list)
    assert model.fixef[0].shape == (47,3)
    assert model.fixef[1].shape == (3,3)

    assert isinstance(model.ranef,list)
    assert model.ranef[0].shape == (47,2)
    assert model.ranef[1].shape == (3,1)

    assert model.ranef_corr.shape == (1,3)
    assert model.ranef_var.shape == (4,3)

    assert np.allclose(model.coefs.loc[:,'Estimate'],model.fixef[0].mean())

    # Test prediction
    assert np.allclose(model.predict(model.data,use_rfx=True),model.data.fits)

def test_log():

    df = pd.read_csv(os.path.join(get_resource_path(),'sample_data.csv'))
    model = Lmer('DV_l ~ IV1+ (IV1|Group)',data=df,family='binomial')
    model.fit(summarize=False)

    assert model.coefs.shape == (2,13)
    estimates = np.array([-0.16098421,  0.00296261])
    assert np.allclose(model.coefs['Estimate'],estimates)

    assert isinstance(model.fixef,pd.core.frame.DataFrame)
    assert model.fixef.shape == (47,2)

    assert isinstance(model.ranef,pd.core.frame.DataFrame)
    assert model.ranef.shape == (47,2)

    assert np.allclose(model.coefs.loc[:,'Estimate'],model.fixef.mean(),atol=.005)

    # Test prediction
    assert np.allclose(model.predict(model.data,use_rfx=True),model.data.fits)
    assert np.allclose(model.predict(model.data,use_rfx=True,pred_type='link'),logit(model.data.fits))
