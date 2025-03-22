from polars import DataFrame
from ._Pipe import pipeable
from .._stats import (
    anova as anova_,
    coef as coef_,
    resid as resid_,
    glm as glm_,
    model_matrix as model_matrix_,
    lm as lm_,
)

__all__ = ["lm", "glm", "anova", "coef", "resid", "model_matrix"]


@pipeable(DataFrame)
def lm(data, formula, **kwargs):
    return lm_(formula, data, **kwargs)


@pipeable(DataFrame)
def glm(data, formula, **kwargs):
    return glm_(formula, data, **kwargs)


@pipeable(None)
def anova(model, **kwargs):
    return anova_(model, **kwargs)


@pipeable(None)
def coef(model, **kwargs):
    return coef_(model, **kwargs)


@pipeable(None)
def resid(model, **kwargs):
    return resid_(model, **kwargs)


@pipeable(None)
def model_matrix(model, **kwargs):
    return model_matrix_(model, **kwargs)
