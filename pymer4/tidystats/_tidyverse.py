from rpy2.robjects.packages import importr
from ._bridge import ensure_py_output, ensure_r_input

__all__ = ["as_tibble", "distinct"]

lib_tibble = importr("tibble")
lib_dplyr = importr("dplyr")


@ensure_py_output
@ensure_r_input
def as_tibble(*args, **kwargs):
    """Convert R object to tibble

    Returns:
        DataFrame: converted tibble
    """
    return lib_tibble.as_tibble(*args, **kwargs)


@ensure_py_output
@ensure_r_input
def distinct(*args, **kwargs):
    """Keeps distinct rows [reference](https://dplyr.tidyverse.org/reference/distinct.html)

    Returns:
        DataFrame: unique rows
    """
    return lib_dplyr.distinct(*args, **kwargs)
