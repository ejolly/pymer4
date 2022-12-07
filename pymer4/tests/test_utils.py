from pymer4.utils import get_resource_path, result_to_table
import pandas as pd
from pymer4.models import Lm
import os


def test_result_to_table(df):
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
