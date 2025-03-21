from rpy2.robjects.packages import importr
from ._bridge import ensure_py_output, ensure_r_input, handle_contrasts
from ._tidyverse import as_tibble, distinct

__all__ = [
    "lm",
    "glm",
    "anova",
    "coef",
    "coefficients",
    "resid",
    "residuals",
    "model_matrix",
]

lib_stats = importr("stats")


@ensure_py_output
@handle_contrasts
@ensure_r_input
def lm(*args, **kwargs):
    return lib_stats.lm(*args, **kwargs)


@ensure_py_output
@handle_contrasts
@ensure_r_input
def glm(*args, **kwargs):
    return lib_stats.glm(*args, **kwargs)


@ensure_py_output
def anova(*args, **kwargs):
    return lib_stats.anova(*args, **kwargs)


@ensure_py_output
def coef(*args, **kwargs):
    return lib_stats.coef(*args, **kwargs)


def coefficients(*args, **kwargs):
    "Alias for coef()"
    return coef(*args, **kwargs)


@ensure_py_output
def resid(*args, **kwargs):
    return lib_stats.residuals(*args, **kwargs)


def residuals(*args, **kwargs):
    "Alias for resid()"
    return resid(*args, **kwargs)


@ensure_py_output
def model_matrix(model):
    return distinct(as_tibble(lib_stats.model_matrix(model)))
