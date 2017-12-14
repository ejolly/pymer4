from __future__ import division
import numpy as np
import pandas as pd
from pymer4.models import Lm
from pymer4.simulate import simulate_lm

def test_simulate_lm():

    # Simulate some data
    data, b = simulate_lm(500,
                          3,
                          coef_vals=[10,2.2,-4.1,3],
                          corrs = .5)

    # Model simulated data
    m = Lm('DV ~ IV1+IV2+IV3',data=data)
    m.fit(summarize=False)

    # Check predictors are correlated
    # True - Generated < .1
    corrs = data.iloc[:,1:].corr().values
    corrs = corrs[np.triu_indices(corrs.shape[0],k=1)]
    assert (np.abs(corrs - .5) < .1).all()

    # Check parameter recovery
    # True - Recovered < .15
    assert (np.abs(m.coefs['Estimate'] - b) < .15).all()
