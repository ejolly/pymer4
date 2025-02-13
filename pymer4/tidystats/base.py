from rpy2.robjects.packages import importr
from .bridge import ensure_py_output

__all__ = ["summary", "row_names", "names"]

lib_base = importr("base")


# @ensure_py_output
def summary(arg):
    """Generic function to product results summaries"""
    return lib_base.summary(arg)


@ensure_py_output
def row_names(arg):
    return lib_base.row_names(arg)


@ensure_py_output
def names(arg):
    return lib_base.names(arg)
