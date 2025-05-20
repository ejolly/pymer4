from numpy import ndarray
from rpy2.robjects.packages import importr
from .bridge import ensure_py_output, ensure_r_input
from rpy2.robjects import RS4, r
from .bridge import R2polars, R2numpy
import polars as pl

__all__ = ["lmer", "glmer", "fixef", "ranef", "bootMer", "is_singular"]

lib_lmerTest = importr("lmerTest")
lib_lmer = importr("lme4")
lib_broom = importr("broom")


@ensure_r_input
def lmer(*args, **kwargs):
    """Fit a linear-mixed-model using `lmer <https://www.rdocumentation.org/packages/lme4/versions/1.1-37/topics/lmer>`_ and get inferential stats using `lmerTest <https://www.rdocumentation.org/packages/lmerTest/versions/3.1-3/topics/lmerTest-package>`_

    Args:
        formula (str): model formula
        data (pl.DataFrame): polars dataframe

    Returns:
        model (R RS4): R model object
    """
    return lib_lmerTest.lmer(*args, **kwargs)


@ensure_r_input
def glmer(*args, **kwargs):
    """Fit a generalized linear-mixed-model using `glmer <https://www.rdocumentation.org/packages/lme4/versions/1.1-37/topics/glmer>`_

    Args:
        formula (str): model formula
        family (str): glm `family <https://www.rdocumentation.org/packages/stats/versions/3.6.2/topics/family>`_
        data (pl.DataFrame): polars dataframe
    Returns:
        model (R RS4): R model object
    """
    return lib_lmer.glmer(*args, **kwargs)


@ensure_py_output
def fixef(model, *args, **kwargs) -> ndarray:
    """Extract model fixed-effects using `fixef <https://www.rdocumentation.org/packages/lme4/versions/1.1-37/topics/fixef>`_"""
    return lib_lmer.fixef_merMod(model, *args, **kwargs)


@ensure_py_output
def ranef(model, *args, **kwargs) -> ndarray:
    """Extract model random-effects/conditional-modes using `ranef <https://www.rdocumentation.org/packages/lme4/versions/1.1-37/topics/ranef>`_"""
    return lib_lmer.ranef_merMod(model, *args, **kwargs)


@ensure_r_input
def lmer_control(*args, **kwargs):
    """Set control options for lmer models"""
    return lib_lmer.lmerControl(*args, **kwargs)


# TODO: Add support for returning coef(model) for BLUPs CIs
@ensure_r_input
def bootMer(
    model,
    nsim=1000,
    parallel="multicore",
    ncpus=4,
    conf_level=0.95,
    conf_method="perc",
    exponentiate=False,
    save_boots=True,
    **kwargs,
):
    """Bootstrap model parameters using `bootMer <https://www.rdocumentation.org/packages/lme4/versions/1.1-37/topics/bootMer>`_ Extracts fixed effects using ``fixef()`` and random-effects using ``broom.mixed::tidy()``

    Args:
        model (R model): `lmer` or `glmer` model
        nsim (int, optional): Number of bootstrap samples. Defaults to 1000.
        parallel (str, optional): Parallelization method. Defaults to "multicore".
        ncpus (int, optional): Number of cores to use. Defaults to 4.
        conf_level (float, optional): Confidence level. Defaults to 0.95.
        conf_method (str, optional): Confidence interval method. Defaults to "perc".
        exponentiate (bool, optional): Whether to exponentiate the results. Defaults to False.
    """
    if not isinstance(model, RS4):
        raise TypeError(
            "To perform bootstrapping on lm/glm models, use the boot() function"
        )
    r_string = """
            function(model){{
            # Get fixed effects
            fe <- fixef(model)
            re <- broom.mixed::tidy(model, effects="ran_pars")
            vc <- re$estimate
            names(vc) <- paste(re$term, re$group, sep="___")
            return(c(fe, vc))
            }}
            """
    extract_func = r(r_string)
    out = lib_lmer.bootMer(
        model, extract_func, nsim=nsim, parallel=parallel, ncpus=ncpus, **kwargs
    )
    cis = R2polars(
        lib_broom.tidy_boot(
            out,
            conf_int=True,
            conf_level=conf_level,
            conf_method=conf_method,
            exponentiate=exponentiate,
        )
    )

    if save_boots:
        boots = R2numpy(out.rx2("t"))
        boots = pl.DataFrame(boots, schema=cis["term"].to_list())
        return cis, boots

    return cis, out


@ensure_r_input
def is_singular(model):
    """Check if a model is singular using the implementation in `lmerTest <https://www.rdocumentation.org/packages/lmerTest/versions/3.1-3/topics/isSingular>`_

    Args:
        model (R model): `lmer` or `glmer` model
    """
    return R2numpy(lib_lmer.isSingular(model))[0] == 1
