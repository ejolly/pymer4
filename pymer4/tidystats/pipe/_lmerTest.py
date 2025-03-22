from ._Pipe import pipeable
from polars import DataFrame
from .._lmerTest import lmer as lmer_, glmer as glmer_

__all__ = ["lmer", "glmer"]


@pipeable(DataFrame)
def lmer(data, formula, **kwargs):
    return lmer_(formula, data, **kwargs)


@pipeable(DataFrame)
def glmer(data, formula, **kwargs):
    return glmer_(formula, data, **kwargs)
