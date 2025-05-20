from rpy2.robjects.packages import importr
from .bridge import ensure_py_output, ensure_r_input

__all__ = ["as_tibble"]

lib_tibble = importr("tibble")


@ensure_py_output
@ensure_r_input
def as_tibble(*args, **kwargs):
    """Coerce input to a `tibble <https://www.rdocumentation.org/packages/tibble/versions/3.2.1/topics/as_tibble>`_

    Returns:
        dataframe (DataFrame): polars DataFrame
    """
    return lib_tibble.as_tibble(*args, **kwargs)
