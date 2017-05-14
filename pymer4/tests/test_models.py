from pymer4.models import Lmer
from pymer4.utils import get_resource_path
import pandas as pd
import os

def test_fit():
    df = pd.read_csv(os.path.join(get_resource_path(),'sample_data.csv'))

    model = Lmer('DV_l ~ IV1+ (IV1|Group)',data=df,family='binomial')
    model.fit()
    assert model.coefs.values.shape == (2,13)

    model = Lmer('DV ~ IV1 + (IV1|Group)',data=df)
    model.fit()
    assert model.coefs.values.shape == (2,8)
