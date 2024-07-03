import os

import pandas as pd
from pytest import fixture

from pymer4.utils import get_resource_path


@fixture(scope="module")
def gammas():
    return pd.read_csv(os.path.join(get_resource_path(), "gammas.csv")).rename(
        columns={"BOLD signal": "bold"}
    )


@fixture(scope="module")
def df():
    return pd.read_csv(os.path.join(get_resource_path(), "sample_data.csv"))


@fixture(scope="module")
def ranef_as_dataframe_correct_results():
    return pd.read_csv(
        os.path.join(get_resource_path(), "ranef_as_dataframe_correct_results.csv"),
        dtype={"grp": object},
    )
