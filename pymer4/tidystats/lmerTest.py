from numpy import ndarray
from rpy2.robjects.packages import importr
from .bridge import ensure_py_output, ensure_r_input

__all__ = ["lmer", "glmer", "fixef", "ranef"]

lib_lmerTest = importr("lmerTest")
lib_lmer = importr("lme4")


@ensure_r_input
def lmer(*args, **kwargs):
    """Fit a linear-mixed-model using [lme4](https://www.rdocumentation.org/packages/lme4/versions/1.1-37/topics/lmer) and get inferential stats using [lmerTest](https://www.rdocumentation.org/packages/lmerTest/versions/3.1-3/topics/lmerTest-package)

    Args:
        formula (str): model formula
        data (pl.DataFrame): polars dataframe

    Returns:
        model (R RS4): R model object
    """
    return lib_lmerTest.lmer(*args, **kwargs)


@ensure_r_input
def glmer(*args, **kwargs):
    """Fit a generalized linear-mixed-modelglmm using [lme4](https://www.rdocumentation.org/packages/lme4/versions/1.1-37/topics/glmer)

    Args:
        formula (str): model formula
        family (str): glm [family](https://www.rdocumentation.org/packages/stats/versions/3.6.2/topics/family)
        data (pl.DataFrame): polars dataframe
    Returns:
        model (R RS4): R model object
    """
    return lib_lmer.glmer(*args, **kwargs)


@ensure_py_output
def fixef(model, *args, **kwargs) -> ndarray:
    """Extract model [fixed-effects](https://www.rdocumentation.org/packages/lme4/versions/1.1-37/topics/fixef)"""
    return lib_lmer.fixef_merMod(model, *args, **kwargs)


@ensure_py_output
def ranef(model, *args, **kwargs) -> ndarray:
    """Extract model [random-effects/conditional-modes](https://www.rdocumentation.org/packages/lme4/versions/1.1-37/topics/ranef)"""
    return lib_lmer.ranef_merMod(model, *args, **kwargs)


@ensure_r_input
def lmer_control(*args, **kwargs):
    """Set control options for lmer models"""
    return lib_lmer.lmerControl(*args, **kwargs)
