from rpy2.robjects.packages import importr
from .bridge import ensure_py_output, ensure_r_input, to_dict, R2numpy, R2polars
from .tibble import as_tibble
from rpy2.robjects import RS4, ListVector, r
import polars as pl

lib_stats = importr("stats")
lib_lmer = importr("lme4")
lib_boot = importr("boot")
lib_broom = importr("broom")
lib_broom_mixed = importr("broom.mixed")

__all__ = ["coef", "predict", "simulate", "confint", "boot"]


@ensure_py_output
@ensure_r_input
def coef(model, *args, **kwargs):
    """Extract coefficients from ``lm``, ``glm``, ``lmer`` or ``glmer`` models. Uses `lme4::coef.merMod <https://www.rdocumentation.org/packages/lme4/versions/1.1-37/topics/coef.merMod>`_ for linear-mixed-models and `stats::coef.lm <https://www.rdocumentation.org/packages/stats/versions/3.6.2/topics/coef.lm>`_ for linear models.

    Args:
        model (R ListVector or RS4): R model

    Returns:
        coefficients (ndarray or DataFrame): numpy array of coefficients (`lm` and `glm`) or polars DataFrame of BLUPs (`lmer` and `glmer`)
    """

    if isinstance(model, RS4):
        func = lib_lmer.coef_merMod
    elif isinstance(model, ListVector):
        func = lib_stats.coef

    return func(model, *args, **kwargs)


@ensure_py_output
@ensure_r_input
def predict(model, *args, **kwargs):
    """Generate predictions from a model given existing or new data. Uses `lme4::predict.merMod <https://www.rdocumentation.org/packages/lme4/versions/1.1-37/topics/predict.merMod>`_ for linear-mixed-models, `stats::predict.lm <https://www.rdocumentation.org/packages/stats/versions/3.6.2/topics/predict.lm>`_ for linear models, and `stats::predict.glm <https://www.rdocumentation.org/packages/stats/versions/3.6.2/topics/predict.glm>`_ for generalized linear models.

    Args:
        model (R ListVector or RS4): R model

    Returns:
        predictions (ndarray): numpy array of predictions same length as the model's data or input data
    """
    if isinstance(model, RS4):
        func = lib_lmer.predict_merMod
    elif isinstance(model, ListVector):
        method = to_dict(model).get("method", None)
        if method and method[0] == "glm.fit":
            func = lib_stats.predict_glm
        else:
            func = lib_stats.predict

        func = lib_stats.predict
    return func(model, *args, **kwargs)


@ensure_py_output
@ensure_r_input
def simulate(model, *args, **kwargs):
    """Simulate a new dataset from a model. Uses `lme4::simulate.merMod <https://www.rdocumentation.org/packages/lme4/versions/1.1-37/topics/simulate.merMod>`_ for linear-mixed-models, `stats::simulate.lm <https://www.rdocumentation.org/packages/stats/versions/3.6.2/topics/simulate.lm>`_ for linear models, and `stats::simulate.glm <https://www.rdocumentation.org/packages/stats/versions/3.6.2/topics/simulate.glm>`_ for general linear models.

    Args:
        model (R ListVector or RS4): R model

    Returns:
        dataset (DataFrame): polars DataFrame with number of columns = `nsim` (1 by default)
    """
    if isinstance(model, RS4):
        func = lib_lmer.simulate_merMod
    elif isinstance(model, ListVector):
        method = to_dict(model).get("method", None)
        if method and method[0] == "glm.fit":
            func = lib_stats.simulate_glm
        else:
            func = lib_stats.simulate
    return func(model, *args, **kwargs)


@ensure_r_input
def confint(model, *args, as_df=True, **kwargs):
    """Confidence intervals including via bootstrapping using `stats::confint <https://www.rdocumentation.org/packages/stats/versions/3.6.2/topics/confint>`_ or `lme4::confint.merMod <https://www.rdocumentation.org/packages/lme4/versions/1.1-37/topics/confint.merMod>`_"""

    if isinstance(model, RS4):
        func = lib_lmer.confint_merMod
    elif isinstance(model, ListVector):
        method = to_dict(model).get("method", None)
        if method and method[0] == "glm.fit":
            func = lib_stats.confint_glm
        else:
            func = lib_stats.confint_lm
    result = func(model, *args, **kwargs)
    if as_df:
        return as_tibble(result)
    return result


# NOTE: Experimental
@ensure_py_output
@ensure_r_input
def boot(
    data,
    model,
    formula,
    R,
    family=None,
    link=None,
    conf_method="perc",
    conf_level=0.95,
    return_boots=False,
    **kwargs,
):
    """
    NOTE: Experimental - may not reliably handle ``glm`` models. Currently unused.
    Generate bootstrapped confidence intervals for a model using `boot::boot <https://www.rdocumentation.org/packages/boot/versions/1.3-31/topics/boot>`_ and `broom::tidy.boot <https://broom.tidymodels.org/reference/tidy.boot.html>`_ or for lme4 model using `lme4::confint.merMod <https://www.rdocumentation.org/packages/lme4/versions/1.1-37/topics/confint.merMod>`_

    Args:
        data (DataFrame): polars DataFrame to resample
        model (R model): model object
        formula (str): model formula
        R (int): number of bootstrap samples
        family (str, optional): family for glm models. Defaults to None.
        conf_method (str, optional): how to calculated intervalsl: "perc", "bca", "basic", "norm". Defaults to "perc".
        conf_level (float, optional): _description_. Defaults to 0.95.


    Returns:
        summary (DataFrame): bootstrap results
    """
    import warnings

    warnings.warn(
        "This function is experimental and not reliable because it does not guarantee that each bootstrapped model is called in the same way as the input model"
    )

    if isinstance(model, RS4):
        raise NotImplementedError(
            "To perform bootstrapping on lmer models, use the `bootMer()`"
        )
    elif isinstance(model, ListVector):
        method = to_dict(model).get("method", None)
        if method and method[0] == "glm.fit":
            r_string = f"""
                    function(d, indices){{
                    d <- d[indices,]
                    fit <- glm({formula}, data=d, family={family}, link={link})
                    return(coef(fit))
                    }}
                    """
        else:
            r_string = f"""
                    function(d, indices){{
                    d <- d[indices,]
                    fit <- lm({formula}, data=d)
                    return(coef(fit))
                    }}
                    """
        boot_func = r(r_string)
        out = lib_boot.boot(data, boot_func, R, **kwargs)
        cis = R2polars(
            lib_broom.tidy_boot(
                out, conf_int=True, conf_level=conf_level, conf_method=conf_method
            )
        )

        if return_boots:
            boots = R2numpy(out.rx2("t"))
            boots = pl.DataFrame(boots, schema=cis["term"].to_list())
            return cis, boots

    return cis
