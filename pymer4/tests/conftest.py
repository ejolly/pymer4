import pandas as pd
from pymer4.utils import get_resource_path
import os
from pytest import fixture


@fixture(scope="module")
def gammas():
    return pd.read_csv(os.path.join(get_resource_path(), "gammas.csv")).rename(
        columns={"BOLD signal": "bold"}
    )


@fixture(scope="module")
def df():
    return pd.read_csv(os.path.join(get_resource_path(), "sample_data.csv"))
