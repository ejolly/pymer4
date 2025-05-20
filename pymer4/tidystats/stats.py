from rpy2.robjects.packages import importr
from .bridge import ensure_py_output, ensure_r_input
from .tibble import as_tibble

__all__ = ["lm", "glm", "anova", "resid", "model_matrix"]

lib_stats = importr("stats")


@ensure_r_input
def lm(*args, **kwargs):
    """Fit a linear-model using `stats::lm <https://www.rdocumentation.org/packages/stats/versions/3.6.2/topics/lm>`_

    Args:
        formula (str): model formula
        data (pl.DataFrame): polars dataframe

    Returns:
        model (R ListVector): R model object
    """
    return lib_stats.lm(*args, **kwargs)


@ensure_r_input
def glm(*args, **kwargs):
    """Fit a generalized linear model using `stats::glm <https://www.rdocumentation.org/packages/stats/versions/3.6.2/topics/glm>`_

    Args:
        formula (str): model formula
        family (str): glm `family <https://www.rdocumentation.org/packages/stats/versions/3.6.2/topics/family>`_
        data (pl.DataFrame): polars dataframe
    Returns:
        model (R ListVector): R model object
    """
    return lib_stats.glm(*args, **kwargs)


@ensure_py_output
@ensure_r_input
def anova(*args, **kwargs):
    """
    Compare one or more models using `stats::anova.glm <https://www.rdocumentation.org/packages/stats/versions/3.6.2/topics/anova.glm>`.

    Can also calculate `stats::anova <https://www.rdocumentation.org/packages/stats/versions/3.6.2/topics/anova>`_ from a fitted model, but prefer `ts.joint_tests` from `emmeans` to ensure balanced Type-III SS inferences"""
    return lib_stats.anova(*args, **kwargs)


@ensure_py_output
@ensure_r_input
def resid(model, *args, **kwargs):
    """Extract model `residuals <https://www.rdocumentation.org/packages/stats/versions/3.6.2/topics/residuals>`_"""
    return lib_stats.residuals(model, *args, **kwargs)


@ensure_r_input
def model_matrix(model, unique=True):
    """
    Extract model `design matrix <https://www.rdocumentation.org/packages/stats/versions/3.6.2/topics/model.matrix>`_

    Args:
        model (R model): `lm`, `glm`, `lmer`, or `glmer` model
        unique (bool; optional): return a dataframe the size of the model's data; default False
    """

    @ensure_py_output
    def return_tibble(model):
        return as_tibble(lib_stats.model_matrix(model))

    out = return_tibble(model)
    out = out.unique(maintain_order=True) if unique else out
    return out
