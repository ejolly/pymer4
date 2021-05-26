from __future__ import division
from pymer4.utils import con2R, R2con, get_resource_path, result_to_table
import pandas as pd
import numpy as np
from pymer4.models import Lm
import os


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


def test_result_to_table():
    df = pd.read_csv(os.path.join(get_resource_path(), "sample_data.csv"))
    model = Lm("DV ~ IV1 + IV3", data=df)
    model.fit(summarize=False)

    formatted = result_to_table(model, drop_intercept=False)

    assert isinstance(formatted, pd.DataFrame)
    assert formatted.shape == (3, 6)
    assert set(["Predictor", "b", "ci", "t", "df", "p"]) == set(formatted.columns)
    assert formatted.iloc[0, -1] == "< .001"

    formatted = result_to_table(model, drop_intercept=True)

    assert isinstance(formatted, pd.DataFrame)
    assert formatted.shape == (2, 6)
