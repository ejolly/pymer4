from rpy2.robjects.packages import importr
from ._bridge import ensure_py_output, ensure_r_input

__all__ = ["clean_names", "get_dupes"]

lib_janitor = importr("janitor")


@ensure_py_output
@ensure_r_input
def clean_names(*args, **kwargs):
    """Returns Dataframe with clean-up column names [reference](https://www.rdocumentation.org/packages/janitor/versions/1.2.0/topics/clean_names)

    Returns:
        DataFrame: original dataframe with renamed columns
    """
    return lib_janitor.clean_names(*args, **kwargs)


@ensure_py_output
@ensure_r_input
def get_dupes(*args, **kwargs):
    """Get duplicate rows passing in one ore more column names as string

    Returns:
        DataFrame: Duplicated roows
    """
    return lib_janitor.get_dupes(*args, **kwargs)
