from rpy2.robjects.packages import importr
from .tibble import as_tibble
from rpy2.robjects import RS4, ListVector
from .bridge import ensure_py_output, to_dict

__all__ = [
    "report",
    "bootstrap_model",
    "model_performance",
    "model_performance_cv",
    "model_icc",
    "get_fixed_params",
    "get_param_names",
]
report_lib = importr("report")
params = importr("parameters")
performance = importr("performance")
insight = importr("insight")


def report(model, **kwargs):
    """Generate a report for a model using the implementation in [`easystats`](https://easystats.github.io/report/reference/report.html)

    Args:
        model (R model): `lm`, `glm`, `lmer`, or `glmer` model
    """
    return report_lib.report(model, **kwargs)


def bootstrap_model(r_model, nboot=1000, parallel="snow", n_cpus=4, as_df=True):
    """Generate bootstrap samples for model coefficients. Supports parallelization

    Args:
        r_model (R model): `lm`, `glm`, `lmer`, or `glmer` model
        nboot (int, optional): Number of bootstrap samples. Defaults to 1000.
        parallel (str, optional): Parallelization method. Defaults to "snow".
        n_cpus (int, optional): Number of CPUs to use. Defaults to 4.
        as_df (bool, optional): Whether to return a polars DataFrame. Defaults to True.
    """
    if not isinstance(r_model, (RS4, ListVector)):
        r_model = r_model.r_model
    out = params.bootstrap_model(
        r_model, parallel=parallel, n_cpus=n_cpus, iterations=nboot
    )
    if as_df:
        return as_tibble(out)
    return out


# TODO:
# 1) Join .result_fit_stats with performance.model_performance
# 2) Rename: result_fit_stats to result_performance
@ensure_py_output
def model_performance(r_model, **kwargs):
    """Calculate model performance using the implementation in [`easystats`](https://easystats.github.io/performance/reference/model_performance.html)

    Args:
        r_model (R model): `lm`, `glm`, `lmer`, or `glmer` model

    """
    if not isinstance(r_model, (RS4, ListVector)):
        r_model = r_model.r_model
    return performance.model_performance(r_model, **kwargs)


@ensure_py_output
def model_performance_cv(r_model, method="k_fold", stack=False, **kwargs):
    """Calculate cross-validated model performance using the implementation in [`easystats`](https://easystats.github.io/performance/reference/performance_cv.html)

    Args:
        r_model (R model): `lm`, `glm`, `lmer`, or `glmer` model
        method (str, optional): Method for cross-validation. Defaults to "k_fold".
        stack (bool, optional): Whether to stack the results. Defaults to False.

    """
    if not isinstance(r_model, (RS4, ListVector)):
        r_model = r_model.r_model
    return performance.performance_cv(r_model, method=method, stack=stack, **kwargs)


@ensure_py_output
def model_icc(r_model, by_group=True, **kwargs):
    """Calculate the intraclass correlation coefficient (ICC) for a model using the implementation in [`easystats`](https://easystats.github.io/performance/reference/icc.html)

    Args:
        r_model (R model): `lm`, `glm`, `lmer`, or `glmer` model
        by_group (bool, optional): Whether to calculate the ICC for each group. Defaults to True.

    Returns:
        DataFrame: Table of ICCs
    """
    if not isinstance(r_model, (RS4, ListVector)):
        r_model = r_model.r_model
    return performance.icc(r_model, by_group=by_group, **kwargs)


@ensure_py_output
def get_fixed_params(r_model):
    """Get the parameters for a model using the implementation in [`easystats`](https://easystats.github.io/insight/reference/get_parameters.html)

    Args:
        r_model (R model): `lm`, `glm`, `lmer`, or `glmer` model
    """
    return insight.get_parameters(r_model, effects="fixed")


def get_param_names(r_model):
    """Get the parameter names for a model using the implementation in [`easystats`](https://easystats.github.io/insight/reference/find_parameters.html)

    Args:
        r_model (R model): `lm`, `glm`, `lmer`, or `glmer` model

    Returns:
        tuple: Fixed and random parameter names
    """
    params_obj = insight.find_parameters(r_model)
    fixed_params = params_obj.rx2("conditional")
    fixed_params = list(fixed_params) if fixed_params else None
    random_params = params_obj.rx2("random")
    random_params = to_dict(random_params) if random_params else None
    return fixed_params, random_params
