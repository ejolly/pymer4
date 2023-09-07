__all__ = ["save_model", "load_model"]

import os
from .models import Lmer
from rpy2.robjects.packages import importr
from joblib import dump, load

base = importr("base")


def save_model(model, filepath, **kwargs):
    """
    Function for saving pymer4 models. All models are saved using joblib.dump files so
    filepath extensions should end with .joblib. For Lmer models an additional
    filepath.robj file will be created to retain all R objects.

    Args:
        model (pymer4.models): an instance of a pymer4 model
        filepath (str): full filepath string ending .joblib
        kwargs: optional keyword arguments to joblib.dump
    """

    filepath = str(filepath)
    if not filepath.endswith(".joblib"):
        raise IOError("filepath must end with .joblib")

    rds_file = filepath.replace(".joblib", ".rds")

    # Save the python object
    dump(model, filepath, **kwargs)
    assert os.path.exists(filepath)
    # Now deal with model object in R if needed
    if model.model_obj is not None:
        base.saveRDS(model.model_obj, rds_file)
        assert os.path.exists(rds_file)


def load_model(filepath):
    """
    Function for loading pymer4 models. A file path ending in .joblib should be provided. For Lmer models an additional filepath.robj should be located in the same directory.

    Args:
        model (pymer4.models): an instance of a pymer4 model
        filepath (str): full filepath string ending with .joblib
    """

    filepath = str(filepath)
    if not filepath.endswith(".joblib"):
        raise IOError("filepath must end with .joblib")

    rds_file = filepath.replace(".joblib", ".rds")

    # Load python object
    model = load(filepath)

    # Now deal with model object in R if needed
    if isinstance(model, Lmer):
        model.model_obj = base.readRDS(rds_file)
    return model
