from __future__ import division
from pymer4.utils import con2R, R2con
import pandas as pd
import numpy as np


def test_con2R():
    x = np.array([[-1, 0, 0, 1], [-0.5, -0.5, 0.5, 0.5], [-3 / 3, 1 / 3, 1 / 3, 1 / 3]])
    out = con2R(x)
    assert out.shape == (4, 3)
    names = ["1 v s4", "1+2 vs 3+4", "1 vs 2+3+4"]
    out = con2R(x, names=names)
    assert isinstance(out, pd.DataFrame)
    assert [x == y for x, y in zip(out.columns, names)]
    assert out.shape == (4, 3)

    out = con2R(np.array([-1, 0, 1]))
    assert np.allclose(
        out, np.array([[-0.5, 0.40824829], [0.0, -0.81649658], [0.5, 0.40824829]])
    )
