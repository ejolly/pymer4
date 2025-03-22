from rpy2.robjects.packages import importr
from ._bridge import ensure_py_output, ensure_r_input, handle_contrasts

__all__ = ["lmer", "glmer"]

lib_lmerTest = importr("lmerTest")


@ensure_py_output
@handle_contrasts
@ensure_r_input
def lmer(*args, **kwargs):
    """Fit lmm using lme4 and get inferential stats using lmerTest [reference](https://rdrr.io/cran/lmerTest/man/lmerTest-package.html)

    Returns:
        R model: estimated model
    """
    return lib_lmerTest.lmer(*args, **kwargs)


@ensure_py_output
@handle_contrasts
@ensure_r_input
def glmer(*args, **kwargs):
    """Fit glmm using lme4 and get inferential stats using lmerTest [reference](https://rdrr.io/cran/lmerTest/man/lmerTest-package.html)

    Returns:
        R model: estimated model
    """
    return lib_lmerTest.glmer(*args, **kwargs)
