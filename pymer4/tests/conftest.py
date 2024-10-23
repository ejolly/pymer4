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


@fixture(scope="module")
def ranef_as_dataframe_correct_results():
    # grp is an int in the sample data but in general grp can be a string.
    # Therefore, we read grp as an object dtype to match the R output.
    return pd.read_csv(
        os.path.join(get_resource_path(), "ranef_as_dataframe_correct_results.csv"),
        dtype={"grp": object},
    )
