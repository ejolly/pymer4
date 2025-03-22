import numpy as np
from rpy2.robjects.conversion import localconverter
from rpy2.robjects import pandas2ri, numpy2ri
import rpy2.robjects as ro
import polars as pl
from functools import wraps
from warnings import warn

__all__ = [
    "polars2R",
    "R2polars",
    "R2numpy",
    "numpy2R",
    "strVector2dict",
    "ensure_r_input",
    "ensure_py_output",
    "sanitize_polars_columns",
    "drop_rownames",
    "handle_contrasts",
]


def strVector2dict(strVector):
    "Recursively convert an R StrVector into a Python dict with all Python types. Ignores R 'call' and 'terms'. Useful for seeing an lm() or lmer() model as a Python dict."
    temp = dict(zip(strVector.names, strVector))

    out = dict()
    for key, orig_value in temp.items():
        # Numerics
        if isinstance(orig_value, (ro.vectors.FloatVector, ro.vectors.IntVector)):
            new_value = R2numpy(orig_value)

        # Nested dicts
        elif isinstance(orig_value, ro.vectors.ListVector):
            new_value = strVector2dict(orig_value)

        # Strings
        elif isinstance(orig_value, ro.vectors.StrVector):
            new_value = list(orig_value)

        # Ignore formula and call
        elif key in ["call", "terms"]:
            break

        else:
            warn(f"Ignoring Unhandled: Key: {key}, Type: {type(orig_value)}")
            # new_value = orig_value

        out[key] = new_value
    return out


def polars2R(df):
    """Local conversion of polars dataframe to R dataframe as recommended by rpy2"""
    if isinstance(df, pl.DataFrame):
        with localconverter(ro.default_converter + pandas2ri.converter):
            data = ro.conversion.get_conversion().py2rpy(df.to_pandas())
        return data
    return df


def R2polars(rdf):
    """Local conversion of R dataframe to polars as recommended by rpy2"""
    if not isinstance(rdf, pl.DataFrame):
        with localconverter(ro.default_converter + pandas2ri.converter):
            pandas_df = ro.conversion.get_conversion().rpy2py(rdf)
            pandas_df = pandas_df.map(
                lambda elem: np.nan if elem is ro.NA_Character else elem
            )
        return pl.from_pandas(pandas_df)
    return rdf


def numpy2R(arr):
    """Local conversion of numpy array to R array as recommended by rpy2"""
    if isinstance(arr, np.ndarray):
        with localconverter(ro.default_converter + numpy2ri.converter):
            data = ro.conversion.get_conversion().py2rpy(arr)
        return data
    return arr


def R2numpy(rarr):
    """Local conversion of R array to numpy as recommended by rpy2"""
    if not isinstance(rarr, np.ndarray):
        return np.asarray(rarr)
    return rarr


def _santize_column_names(df):
    """Replace all column names including '.' with '_'"""
    new_cols = []
    for col in df.columns:
        col = col.strip("")
        col = col.replace(".", "_")
        col = col.lstrip("_")
        new_cols.append(col)
    df.columns = new_cols
    return df


def ensure_py_output(func):
    """
    Decorator that converts R outputs to Python equivalents:
    - R FloatVector -> numpy array
    - R dataframe/tibble -> polars dataframe
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)

        # Check if result is an R FloatVector
        if isinstance(result, ro.vectors.FloatVector):
            return R2numpy(result)

        # Check if result is an R dataframe
        elif isinstance(result, ro.vectors.DataFrame):
            return R2polars(result)

        # Check if result is a tibble
        elif hasattr(result, "rclass") and "tbl_df" in list(result.rclass):
            return R2polars(result)

        return result

    return wrapper


def drop_rownames(func):
    """
    Decorator that checks if the output is a polars dataframe
    and removes the 'rownames' column if it exists
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)

        # Check if result is a polars dataframe
        if isinstance(result, pl.DataFrame):
            # Remove rownames column if it exists
            if "rownames" in result.columns:
                return result.drop("rownames")

        # Return the original result if it's not a polars dataframe
        # or if it doesn't have a rownames column
        return result

    return wrapper


def sanitize_polars_columns(func):
    """
    Decorator that fixes R-style column names that include "." and
    spaces and replaces them with "_" while stripping white-space
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)

        # Check if result is a polars dataframe
        if isinstance(result, pl.DataFrame):
            # Remove rownames column if it exists
            return _santize_column_names(result)

        # Return the original result if it's not a polars dataframe
        # or if it doesn't have a rownames column
        return result

    return wrapper


def ensure_r_input(func):
    """
    Decorator that checks if a function has a 'data' kwarg that's a polars dataframe.
    If so, converts it to an R dataframe before passing to the function.

    This is useful for R functions like lm() that expect an R dataframe.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        # Check if there are any polars dataframes in the arguments and conver them
        converted_args = [
            polars2R(arg) if isinstance(arg, pl.DataFrame) else arg for arg in args
        ]
        # Check if there's a 'data' keyword argument and it's a polars dataframe
        if "data" in kwargs and isinstance(kwargs["data"], pl.DataFrame):
            # Convert polars to R
            kwargs["data"] = polars2R(kwargs["data"])

        # Call the original function with the converted arguments
        return func(*converted_args, **kwargs)

    return wrapper


def handle_contrasts(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Convert Python contrasts to R
        if "contrasts" in kwargs and isinstance(kwargs["contrasts"], dict):
            kwargs["contrasts"] = ro.ListVector(kwargs["contrasts"])

        # Convert Python contrasts to R
        elif "contr" in kwargs and isinstance(kwargs["contr"], dict):
            kwargs["contr"] = ro.ListVector(kwargs["contr"])

        # Call the original function with the converted arguments
        return func(*args, **kwargs)

    return wrapper
