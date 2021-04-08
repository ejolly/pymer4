__all__ = ["save_model", "load_model"]

import os
from .models import Lm, Lm2, Lmer
from .utils import _df_meta_to_arr
from rpy2.robjects.packages import importr
import deepdish as dd
import pandas as pd
import warnings
from tables import NaturalNameWarning

base = importr("base")


def save_model(model, filepath, compression="zlib", **kwargs):
    """
    Function for saving pymer4 models. All models are saved in .h5 or .hdf5 files so filepath extensions should include this. For Lmer models an additional filepath.robj file will be created to retain all R objects.

    Args:
        model (pymer4.models): an instance of a pymer4 model
        filepath (str): full filepath string ending with .h5 or .hd5f
        compression (string): what kind of compression to use; zlib is the default which should be universally accessible, but for example 'blosc' will be faster and produce smaller files. See more here: https://bit.ly/33x9JD7
        kwargs: optional keyword arguments to deepdish.io.save
    """

    if filepath.endswith(".h5") or filepath.endswith(".hdf5"):

        filename = filepath.split(".")[0]

        # Separate out model attributes that are not pandas dataframes (or lists conatins dataframes) or R model objects
        simple_atts, data_atts = {}, {}
        for k, v in vars(model).items():
            skip = False
            if k == "model_obj":
                skip = True
            elif isinstance(v, pd.DataFrame):
                skip = True
            elif isinstance(v, list):
                if any([isinstance(elem, pd.DataFrame) for elem in v]):
                    skip = True
            if not skip:
                simple_atts[k] = v
            else:
                data_atts[k] = v
        simple_atts["model_class"] = model.__class__.__name__

        # Now deal with other attributes
        data_atts_separated = {}
        for k, v in data_atts.items():
            if k != "model_obj":
                # Deconstruct pandas dataframes
                if isinstance(v, pd.DataFrame):
                    cols, idx = _df_meta_to_arr(v)
                    vals = v.values
                    dtypes = v.dtypes.to_dict()
                    data_atts_separated[f"df_cols__{k}"] = cols
                    data_atts_separated[f"df_idx__{k}"] = idx
                    data_atts_separated[f"df_vals__{k}"] = vals
                    data_atts_separated[f"df_dtypes__{k}"] = dtypes
                elif isinstance(v, list):
                    for i, elem in enumerate(v):
                        if isinstance(elem, pd.DataFrame):
                            cols, idx = _df_meta_to_arr(elem)
                            vals = elem.values
                            dtypes = elem.dtypes.to_dict()
                            data_atts_separated[f"list_{i}_cols__{k}"] = cols
                            data_atts_separated[f"list_{i}_idx__{k}"] = idx
                            data_atts_separated[f"list_{i}_vals__{k}"] = vals
                            data_atts_separated[f"list_{i}_dtypes__{k}"] = dtypes
                        else:
                            raise TypeError(
                                f"Value is list but list item is {type(elem)} not pd.DataFrame"
                            )

        # Combine all attributes into a single dict and save with dd
        model_atts = {}
        model_atts["simple_atts"] = simple_atts
        model_atts["data_atts"] = data_atts_separated
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=FutureWarning)
            warnings.simplefilter("ignore", category=NaturalNameWarning)
            dd.io.save(filepath, model_atts, compression=compression, **kwargs)
        assert os.path.exists(filepath)

        # Now deal with model object in R if needed
        if model.model_obj is not None:
            base.saveRDS(model.model_obj, f"{filename}.rds")
            assert os.path.exists(f"{filename}.rds")
    else:
        raise IOError("filepath must end with .h5 or .hdf5")


def load_model(filepath):
    """
    Function for loading pymer4 models. A file path ending in .h5 or .hdf5 should be provided. For Lmer models an additional filepath.robj should be located in the same directory.

    Args:
        model (pymer4.models): an instance of a pymer4 model
        filepath (str): full filepath string ending with .h5 or .hd5f
    """

    if filepath.endswith(".h5") or filepath.endswith(".hdf5"):
        if not os.path.exists(filepath):
            raise IOError("File not found!")

        # Load h5 first
        model_atts = dd.io.load(filepath)
        # Figure out what kind of model we're dealing with
        if model_atts["simple_atts"]["model_class"] == "Lmer":
            model = Lmer("", [])
        elif model_atts["simple_atts"]["model_class"] == "Lm2":
            model = Lm2("", [], "")
        elif model_atts["simple_atts"]["model_class"] == "Lm":
            model = Lm("", [])

        # Set top level attributes
        for k, v in model_atts["simple_atts"].items():
            if k != "model_class":
                setattr(model, k, v)
        # Make sure the model formula is a python string string so that rpy2 doesn't complain
        model.formula = str(model.formula)

        # Set data attributes
        # Container for already set items
        completed = []
        for k, v in model_atts["data_atts"].items():
            # Re-assembe dataframes
            if k.startswith("df_"):
                # First check if we haven't set it yet
                if k not in completed:
                    # Get the id of this deconstructed df
                    item_name = k.split("__")[-1]
                    vals_name = f"df_vals__{item_name}"
                    cols_name = f"df_cols__{item_name}"
                    idx_name = f"df_idx__{item_name}"
                    dtype_name = f"df_dtypes__{item_name}"
                    # Reconstruct the dataframe
                    df = pd.DataFrame(
                        model_atts["data_atts"][vals_name],
                        columns=[
                            e.decode("utf-8") if isinstance(e, bytes) else e
                            for e in model_atts["data_atts"][cols_name]
                        ],
                        index=[
                            e.decode("utf-8") if isinstance(e, bytes) else e
                            for e in model_atts["data_atts"][idx_name]
                        ],
                    ).astype(model_atts["data_atts"][dtype_name])
                    setattr(model, item_name, df)
                    # Add it to the list of completed items
                    completed.extend([item_name, vals_name, idx_name, dtype_name])
            # Same idea for list items
            elif k.startswith("list_"):
                if k not in completed:
                    # Get the id of the deconstructed list
                    item_name = k.split("__")[-1]
                    item_idx = [e for e in k.split("__")[0] if e.isdigit()][0]
                    vals_name = f"list_{item_idx}_vals__{item_name}"
                    cols_name = f"list_{item_idx}_cols__{item_name}"
                    idx_name = f"list_{item_idx}_idx__{item_name}"
                    dtype_name = f"list_{item_idx}_dtypes__{item_name}"
                    # Reconstruct the dataframe
                    df = pd.DataFrame(
                        model_atts["data_atts"][vals_name],
                        columns=[
                            e.decode("utf-8") if isinstance(e, bytes) else e
                            for e in model_atts["data_atts"][cols_name]
                        ],
                        index=[
                            e.decode("utf-8") if isinstance(e, bytes) else e
                            for e in model_atts["data_atts"][idx_name]
                        ],
                    ).astype(model_atts["data_atts"][dtype_name])
                    # Check if the list already exists if so just append to it
                    if hasattr(model, item_name):
                        current_items = getattr(model, item_name)
                        if current_items is not None:
                            current_items += [df]
                            setattr(model, item_name, current_items)
                        else:
                            setattr(model, item_name, [df])
                    # Otherwise create it
                    else:
                        setattr(model, item_name, [df])
                    # Add to the list of completed items
                    completed.extend([item_name, vals_name, idx_name, dtype_name])
        # Now deal with model object in R if needed
        if isinstance(model, Lmer):
            filename = filepath.split(".")[0]
            model.model_obj = base.readRDS(f"{filename}.rds")
        return model
    else:
        raise IOError("filepath must end with .h5 or .hdf5")
