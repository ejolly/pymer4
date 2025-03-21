from rpy2.robjects.packages import importr
from ._bridge import ensure_py_output

__all__ = ["summary"]

lib_base = importr("base")


@ensure_py_output
def summary(arg):
    """Generic function to product results summaries"""
    return lib_base.summary(arg)
