import numpy as np
from rpy2.robjects.conversion import localconverter
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri, numpy2ri
import rpy2.robjects as ro
import polars as pl
from polars import selectors as cs
from functools import wraps

lib_base = importr("base")

__all__ = [
    "polars2R",
    "R2polars",
    "R2numpy",
    "numpy2R",
    "to_dict",
    "ensure_r_input",
    "ensure_py_output",
    "sanitize_polars_columns",
    "con2R",
    "convert_argkwarg_model",
    "convert_argkwarg_dataframe",
    "convert_argkwarg_list",
    "convert_argkwarg_numpy",
    "convert_argkwarg_none",
    "convert_argkwarg_dict",
]


def to_dict(listVector):
    """Recursively convert an R ListVector into a Python dict with all Python types. Ignores R 'call' and 'terms'. Useful for seeing an ``lm()`` or ``lmer()`` model object or the output of ``summary()`` as a Python dict."""

    if not isinstance(listVector, ro.vectors.ListVector):
        raise TypeError("Input must be an R ListVector")

    temp = dict(zip(listVector.names, listVector))

    out = dict()
    for key, orig_value in temp.items():
        # Ignore formula and call
        if key in ["call", "terms"]:
            continue
        # Numerics
        elif isinstance(
            orig_value,
            (ro.vectors.FloatVector, ro.vectors.IntVector, ro.vectors.BoolVector),
        ):
            new_value = R2numpy(orig_value)

        # Nested ListVectors
        elif isinstance(orig_value, ro.vectors.ListVector):
            try:
                new_value = to_dict(orig_value)
            except Exception:
                # raise Exception(f"Failed on key = {key}")
                new_value = orig_value

        # StrVectors
        elif isinstance(orig_value, ro.vectors.StrVector):
            new_value = list(orig_value)

        # Data frames
        elif isinstance(orig_value, ro.vectors.DataFrame) or (
            hasattr(orig_value, "rclass") and "tbl_df" in list(orig_value.rclass)
        ):
            new_value = R2polars(orig_value)

        else:
            # warn(f"Ignoring Unhandled: Key: {key}, Type: {type(orig_value)}")
            continue

        out[key] = new_value
    return out


def convert_argkwarg_dataframe(arg):
    """Convert args/kwargs that are Python DataFrames to proper R type(s)"""
    if isinstance(arg, pl.DataFrame):
        return polars2R(arg)
    return arg


def convert_argkwarg_list(arg):
    """Convert args/kwargs that are Python lists to proper R type(s)"""
    if isinstance(arg, list):
        if any(isinstance(elem, str) for elem in arg):
            arg = ro.StrVector(arg)
        elif any(isinstance(elem, float) for elem in arg):
            arg = ro.FloatVector(arg)
        elif any(isinstance(elem, int) for elem in arg):
            arg = ro.IntVector(arg)
        elif any(isinstance(elem, bool) for elem in arg):
            arg = ro.BoolVector(arg[0], bool)
    return arg


def convert_argkwarg_numpy(arg):
    if isinstance(arg, np.ndarray):
        return numpy2R(arg)
    return arg


def convert_argkwarg_none(arg):
    """Convert args/kwargs that are Python None to proper R type(s)"""
    if arg is None:
        # arg = ro.NA_Real
        arg = ro.NULL
    return arg


def convert_argkwarg_dict(arg):
    """Convert args/kwargs that are Python dicts to proper R type(s)"""
    if isinstance(arg, dict):
        out = dict()
        for key, value in arg.items():
            if isinstance(value, pl.DataFrame):
                out[key] = convert_argkwarg_dataframe(value)
            elif isinstance(value, list):
                out[key] = convert_argkwarg_list(value)
            elif isinstance(value, np.ndarray):
                out[key] = convert_argkwarg_numpy(value)
            elif value is None:
                out[key] = convert_argkwarg_none(value)
            elif isinstance(value, dict):
                # Recursive
                out[key] = ro.ListVector(convert_argkwarg_dict(value))
            else:
                out[key] = value

        return out
    return arg


def convert_argkwarg_model(arg):
    """Convert arg/kwargs that are pymer4 model objects to access their r_model attribute"""
    from ..models.base import model

    if isinstance(arg, model):
        return arg.r_model
    return arg


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
        pl_df = pl.from_pandas(pandas_df)
        pl_df = sanitize_polars_columns(pl_df)
        return pl_df
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


def _Rdot_to_Pyunder(df):
    """Replace all column names including '.' with '_' and strip whitespace"""
    new_cols = []
    for c in df.columns:
        c = c.strip("")
        c = c.replace(".", "_")
        c = c.lstrip("_")
        new_cols.append(c)
    df.columns = new_cols
    return df


def ensure_py_output(func):
    """
    Decorator that converts R outputs to Python equivalents. Currently this includes:

    - R FloatVector -> numpy array
    - R StrVector -> list
    - R dataframe/tibble -> polars dataframe
    - R ListVector of Dataframes -> list of polars dataframes
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)

        # Check if result is an R FloatVector
        if isinstance(result, ro.vectors.FloatVector):
            result = R2numpy(result)

        # Check if result is an R ListVector
        elif isinstance(result, ro.vectors.StrVector):
            result = list(result)

        # Check if result is an R dataframe
        elif isinstance(result, ro.vectors.DataFrame):
            result = R2polars(result)

        # Check if result is a tibble
        elif hasattr(result, "rclass") and "tbl_df" in list(result.rclass):
            result = R2polars(result)

        # Check if result is an R ListVector
        # typically a list of DataFrames
        elif isinstance(result, ro.vectors.ListVector):
            if isinstance(result[0], ro.vectors.DataFrame):
                out = []
                for df in result:
                    row_names = list(lib_base.row_names(df))
                    df = (
                        R2polars(df)
                        .with_columns(level=np.array(row_names))
                        .select("level", cs.exclude("level"))
                    )
                    out.append(df)
                result = out
            if len(result) == 1:
                result = result[0]

        return result

    return wrapper


def _drop_rownames(result):
    """
    Drops the `rownames` column some R funcs add
    """

    return result.drop("rownames", strict=False)


def sanitize_polars_columns(result):
    """
    Clean up polars columns using auxillary functions
    """

    result = _Rdot_to_Pyunder(result)
    result = _drop_rownames(result)

    return result


def ensure_r_input(func):
    """Decorator that converts function arguments that are Pyton types into corresponding R types. Currently this includes:

    - polars DataFrames
    - python lists
    - numpy arrays
    - python dictionaries
    - python None types
    - pymer4 model objects
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        converted_args = []
        for arg in args:
            arg = convert_argkwarg_dataframe(arg)
            arg = convert_argkwarg_list(arg)
            arg = convert_argkwarg_numpy(arg)
            arg = convert_argkwarg_dict(arg)
            arg = convert_argkwarg_none(arg)
            arg = convert_argkwarg_model(arg)
            converted_args.append(arg)

        converted_kwargs = convert_argkwarg_dict(kwargs)

        # Call the original function with the converted arguments
        return func(*converted_args, **converted_kwargs)

    return wrapper


def con2R(arr):
    """
    Convert human-readable contrasts into a form that R requires. Works like the `make.contrasts() <https://www.rdocumentation.org/packages/gmodels/versions/2.18.1/topics/make.contrasts>`_ function from the `gmodels <https://cran.r-project.org/web/packages/gmodels/index.html>`_ package, in that it will auto-solve for the remaining orthogonal k-1 contrasts if fewer than k-1 contrasts are specified.

    Arguments:
        arr (np.ndarray): 1d or 2d numpy array with each row reflecting a unique contrast and each column a factor level

    Returns:
        A 2d numpy array useable with the contrasts argument of R models
    """

    if isinstance(arr, list):
        arr = np.array(arr)
    if arr.ndim < 2:
        arr = np.atleast_2d(arr)
    elif arr.ndim > 2:
        raise ValueError(
            f"input array should be 1d or 2d but a {arr.ndim}d array was passed"
        )

    nrow, ncol = arr.shape[0], arr.shape[1]

    # At most k-1 contrasts are possible
    if nrow >= ncol:
        raise ValueError(
            f"Too many contrasts requested ({nrow}). Must be less than the number of factor levels ({ncol})."
        )

    # Pseudo-invert request contrasts
    value = np.linalg.pinv(arr)
    v_nrow, v_ncol = value.shape[0], value.shape[1]

    # Upper triangle of R is the same as result from qr() in R
    Q, R = np.linalg.qr(np.column_stack([np.ones((v_nrow, 1)), value]), mode="complete")
    if np.linalg.matrix_rank(R) != v_ncol + 1:
        raise ValueError(
            "Singular contrast matrix. Some of the requested contrasts are perfectly co-linear."
        )
    cm = Q[:, 1:ncol]
    cm[:, :v_ncol] = value

    return cm


def R2con(arr):
    """
    Convert R-flavored contrast matrix to intepretable contrasts as would be specified by user. `Reference <https://goo.gl/E4Mms2>`_

    Args:
        arr (np.ndarry): 2d contrast matrix output from R's contrasts() function.

    Returns:
        np.ndarray: 2d array organized as contrasts X factor levels
    """

    intercept = np.ones((arr.shape[0], 1))
    mat = np.column_stack([intercept, arr])
    inv = np.linalg.inv(mat)
    return inv
