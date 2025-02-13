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
    """Extract coefficients from `lm`, `glm`, `lmer` or `glmer` models

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
    """Generate predictions from a model given existing or new data

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
    """Simulate a new dataset from a model

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


@ensure_py_output
@ensure_r_input
def confint(model, *args, **kwargs):
    if isinstance(model, RS4):
        func = lib_lmer.confint_merMod
    elif isinstance(model, ListVector):
        method = to_dict(model).get("method", None)
        if method and method[0] == "glm.fit":
            func = lib_stats.confint_glm
        else:
            func = lib_stats.confint_lm
    return as_tibble(func(model, *args, **kwargs))


# TODO: Switch out to `easystats` ?
# @ensure_py_output
@ensure_r_input
def boot(
    data,
    model,
    formula,
    R,
    family=None,
    conf_method="perc",
    conf_level=0.95,
    return_boots=False,
    **kwargs,
):
    """
    Generate bootstrapped confidence intervals for a model using [`boot`](https://www.rdocumentation.org/packages/boot/versions/1.3-31/topics/boot) and [`tidy.boot`](https://broom.tidymodels.org/reference/tidy.boot.html) or for lme4 model using [`confint`](https://www.rdocumentation.org/packages/lme4/versions/1.1-37/topics/confint.merMod)

    Args:
        data (DataFrame): polars DataFrame to resample
        model (R model): model object
        formula (str): model formula
        R (int): number of bootstrap samples
        family (str, optional): family for glm models. Defaults to None.
        conf_method (str, optional): how to calculated intervalsl: "perc", "bca", "basic", "norm". Defaults to "perc".
        conf_level (float, optional): _description_. Defaults to 0.95.

    Raises:
        NotImplementedError: _description_

    Returns:
        summary (DataFrame): bootstrap results
    """

    if isinstance(model, RS4):
        raise NotImplementedError(
            "To perform bootstrapping on lmer models, use the `tidy` function with `conf_method='boot'"
        )
    elif isinstance(model, ListVector):
        method = to_dict(model).get("method", None)
        if method and method[0] == "glm.fit":
            r_string = f"""
                    function(d, indices){{
                    d <- d[indices,]
                    fit <- glm({formula}, data=d, family={family})
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
