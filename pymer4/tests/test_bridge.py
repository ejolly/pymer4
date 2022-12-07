from pymer4.bridge import pandas2R, R2pandas, numpy2R, R2numpy, con2R
from rpy2 import robjects as ro
import numpy as np
import pandas as pd
from pymer4.utils import get_resource_path
import os


def test_pandas(df):
    # py -> R
    rdf = pandas2R(df)
    # py -> R -> py
    df_fromR = R2pandas(rdf)

    # We compare values because the dtypes differ
    assert np.allclose(df.to_numpy(), df_fromR.to_numpy())
    # pandas dtypes:
    # Group      int64
    # IV1      float64
    # DV_l       int64
    # DV       float64
    # IV2      float64
    # IV3      float64
    # R dtypes:
    # Group      int32
    # IV1      float64
    # DV_l       int32
    # DV       float64
    # IV2      float64
    # IV3      float64

    # pandas loaded with numeric indices
    assert df.index.dtype == int
    # Whereas R uses str df indices
    assert df_fromR.index.dtype == object


def test_numpy():
    arr = np.random.randn(100, 100)
    rarr = numpy2R(arr)
    arr_fromR = R2numpy(rarr)
    assert np.allclose(arr, arr_fromR)

    letters = ro.r.letters
    assert isinstance(letters, ro.vectors.StrVector)
    letters_fromR = R2numpy(letters)
    assert isinstance(letters_fromR, np.ndarray)
    letters_fromR_toR = numpy2R(letters_fromR)
    # Even we we convert back rpy2 keeps it as an array
    assert not isinstance(letters_fromR_toR, ro.vectors.StrVector)


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
