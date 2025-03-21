from rpy2.robjects.packages import importr
from ._bridge import ensure_py_output

__all__ = ["lmer"]

lib_lmerTest = importr("lmerTest")


@ensure_py_output
def lmer(*args, **kwargs):
    """Fit lmm using lme4 and get inferential stats using lmerTest [reference](https://rdrr.io/cran/lmerTest/man/lmerTest-package.html)

    Returns:
        R model: estimated model
    """
    return lib_lmerTest.lmer(*args, **kwargs)
