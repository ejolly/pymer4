from ._Pipe import pipeable
from .._base import summary as summary_

__all__ = ["summary"]


@pipeable(None)
def summary(model, **kwargs):
    return summary_(model, **kwargs)
