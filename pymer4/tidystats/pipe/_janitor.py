from ._Pipe import pipeable
from .._janitor import clean_names as clean_names_, get_dupes as get_dupes_

__all__ = ["clean_names", "get_dupes"]


@pipeable(None)
def clean_names(model, **kwargs):
    return clean_names_(model, **kwargs)


@pipeable(None)
def get_dupes(model, **kwargs):
    return get_dupes_(model, **kwargs)
