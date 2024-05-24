__all__ = ["save_model", "load_model", "load_dataset"]

import pandas as pd
import seaborn as sns
import os
from pymer4 import get_resource_path
from bambi import load_data


def load_dataset(name="sampledata"):
    """
    Load a sample dataset from the pymer4 resources, seaborn, or bambi.

    Args:
        name (str, optional): name of dataset. Defaults to "sampledata".

    Returns:
        pd.DataFrame: dataframe of dataset
    """

    if name == "sampledata":
        return pd.read_csv(os.path.join(get_resource_path(), "sample_data.csv"))

    if name in [
        "anagrams",
        "anscombe",
        "attention",
        "brain_networks",
        "car_crashes",
        "diamonds",
        "dots",
        "dowjones",
        "exercise",
        "flights",
        "fmri",
        "geyser",
        "glue",
        "healthexp",
        "iris",
        "mpg",
        "penguins",
        "planets",
        "seaice",
        "taxis",
        "tips",
        "titanic",
    ]:
        return sns.load_dataset(name)

    return load_data(name)


def save_model(model, filepath, compression="zlib", **kwargs):

    raise NotImplementedError("Model i/o not yet supported")


def load_model(filepath):

    raise NotImplementedError("Model i/o not yet supported")
