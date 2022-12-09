import numpy as np
import pandas as pd
from rpy2.robjects.conversion import localconverter
from rpy2.robjects import pandas2ri, numpy2ri
import rpy2.robjects as robjects

__all__ = ["pandas2R", "R2pandas", "con2R", "R2con", "numpy2R", "R2numpy"]


def pandas2R(df):
    """Local conversion of pandas dataframe to R dataframe as recommended by rpy2"""
    with localconverter(robjects.default_converter + pandas2ri.converter):
        data = robjects.conversion.get_conversion().py2rpy(df)
    return data


def R2pandas(rdf):
    """Local conversion of R dataframe to pandas as recommended by rpy2"""
    with localconverter(robjects.default_converter + pandas2ri.converter):
        data = robjects.conversion.get_conversion().rpy2py(rdf)
    return data


def R2con(arr):
    """
    Convert R-flavored contrast matrix to intepretable contrasts as would be specified by user. Reference: https://goo.gl/E4Mms2

    Args:
        arr (np.ndarry): 2d contrast matrix output from R's contrasts() function.

    Returns:
        np.ndarray: 2d array organized as contrasts X factor levels
    """

    intercept = np.ones((arr.shape[0], 1))
    mat = np.column_stack([intercept, arr])
    inv = np.linalg.inv(mat)
    return inv


def con2R(arr, names=None):
    """
    Convert human-readable contrasts into a form that R requires. Works like the make.contrasts() function from the gmodels package, in that it will auto-solve for the remaining orthogonal k-1 contrasts if fewer than k-1 contrasts are specified.

    Arguments:
        arr (np.ndarray): 1d or 2d numpy array with each row reflecting a unique contrast and each column a factor level
        names (list/np.ndarray): optional list of contrast names which will cast the return object as a dataframe

    Returns:
        A 2d numpy array or dataframe useable with the contrasts argument of glmer
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
    if names is not None:
        if not isinstance(names, (list, np.ndarray)):
            raise TypeError("names should be a list or numpy array")
        elif len(names) != nrow:
            raise ValueError(
                "names should have the same number of items as contrasts (rows)"
            )

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

    if names is not None:
        cm = pd.DataFrame(cm, columns=names)
    return cm


def numpy2R(arr):
    """Local conversion of R array to numpy as recommended by rpy2"""
    with localconverter(robjects.default_converter + numpy2ri.converter):
        data = robjects.conversion.get_conversion().rpy2py(arr)
    return data


def R2numpy(rarr):
    """Local conversion of R array to numpy as recommended by rpy2"""
    return np.asarray(rarr)
