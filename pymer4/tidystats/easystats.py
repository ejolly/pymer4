from rpy2.robjects.packages import importr
from .bridge import ensure_py_output, to_dict, R2numpy, ensure_r_input

__all__ = [
    "report",
    "bootstrap_model",
    "model_performance",
    "model_performance_cv",
    "model_icc",
    "get_fixed_params",
    "get_param_names",
    "model_params",
    "is_mixed_model",
    "is_converged",
]
report_lib = importr("report")
params = importr("parameters")
performance = importr("performance")
insight = importr("insight")


def report(model, **kwargs):
    """Generate a report for a model using the implementation in `easystats <https://easystats.github.io/report/reference/report.html>`_

    Args:
        model (R model): `lm`, `glm`, `lmer`, or `glmer` model
    """
    return report_lib.report(model, **kwargs)


@ensure_py_output
@ensure_r_input
def bootstrap_model(r_model, nboot=1000, parallel="multicore", n_cpus=4, **kwargs):
    """Generate bootstrap samples for model fixed effects coefficients using the implementation in `parameters::bootstrap_model <https://easystats.github.io/parameters/reference/bootstrap_model.html>`_

    Args:
        r_model (R model): `lm`, `glm`, `lmer`, or `glmer` model
        nboot (int, optional): Number of bootstrap samples. Defaults to 1000.
        parallel (str, optional): Parallelization method. Defaults to "snow".
        n_cpus (int, optional): Number of CPUs to use. Defaults to 4.
    """
    out = params.bootstrap_model(
        r_model, parallel=parallel, n_cpus=n_cpus, iterations=nboot, **kwargs
    )
    return out


@ensure_py_output
@ensure_r_input
def bootstrap_params(
    r_model,
    centrality="mean",
    nboot=1000,
    parallel="snow",
    n_cpus=4,
    **kwargs,
):
    """Generate bootstrapped summary statistics for model fixed effects coefficients using the implementation in `parameters::bootstrap_parameters <https://easystats.github.io/parameters/reference/bootstrap_parameters.html>`_. Does not return bootstrap samples.

    Args:
        r_model (R model): `lm`, `glm`, `lmer`, or `glmer` model
        centrality (str, optional): Centrality measure. Defaults to "mean".
        nboot (int, optional): Number of bootstrap samples. Defaults to 1000.
        parallel (str, optional): Parallelization method. Defaults to "snow".
        n_cpus (int, optional): Number of CPUs to use. Defaults to 4.
        as_df (bool, optional): Whether to return a polars DataFrame. Defaults to True.
    """
    out = params.bootstrap_parameters(
        r_model,
        centrality=centrality,
        parallel=parallel,
        n_cpus=n_cpus,
        iterations=nboot,
        **kwargs,
    )
    return out


# TODO:
# 1) Join .result_fit_stats with performance.model_performance
# 2) Rename: result_fit_stats to result_performance
@ensure_py_output
@ensure_r_input
def model_performance(r_model, **kwargs):
    """Calculate model performance using the implementation in `performance::model_performance <https://easystats.github.io/performance/reference/model_performance.html>`_

    Args:
        r_model (R model): `lm`, `glm`, `lmer`, or `glmer` model

    """
    return performance.model_performance(r_model, **kwargs)


@ensure_py_output
@ensure_r_input
def model_performance_cv(r_model, method="k_fold", stack=False, **kwargs):
    """Calculate cross-validated model performance using the implementation in `performance::performance_cv <https://easystats.github.io/performance/reference/performance_cv.html>`_

    Args:
        r_model (R model): `lm`, `glm`, `lmer`, or `glmer` model
        method (str, optional): Method for cross-validation. Defaults to "k_fold".
        stack (bool, optional): Whether to stack the results. Defaults to False.

    """
    return performance.performance_cv(r_model, method=method, stack=stack, **kwargs)


@ensure_py_output
@ensure_r_input
def model_icc(r_model, by_group=True, **kwargs):
    """Calculate the intraclass correlation coefficient (ICC) for a model using the implementation in `performance::icc <https://easystats.github.io/performance/reference/icc.html>`_

    Args:
        r_model (R model): `lm`, `glm`, `lmer`, or `glmer` model
        by_group (bool, optional): Whether to calculate the ICC for each group. Defaults to True.

    Returns:
        DataFrame: Table of ICCs
    """
    return performance.icc(r_model, by_group=by_group, **kwargs)


@ensure_py_output
@ensure_r_input
def get_fixed_params(r_model):
    """Get the fixed-effects parameters for a model using the implementation in `insight::get_parameters <https://easystats.github.io/insight/reference/get_parameters.html>`_

    Args:
        r_model (R model): `lm`, `glm`, `lmer`, or `glmer` model
    """
    return insight.get_parameters(r_model, effects="fixed")


@ensure_r_input
def get_param_names(r_model):
    """Get the parameter names for a model using the implementation in `insight::find_parameters <https://easystats.github.io/insight/reference/find_parameters.html>`_

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


@ensure_py_output
@ensure_r_input
def model_params(r_model, **kwargs):
    """Get model parameters using the implementation in `parameters::model_parameters <https://easystats.github.io/parameters/reference/model_parameters.html>`_ and standardize names using the implementation in `insight::standardize_names <https://easystats.github.io/insight/reference/standardize_names.html>`_

    Args:
        r_model (R model): `lm`, `glm`, `lmer`, or `glmer` model
        effects (str, optional): Whether to include fixed or random effects. Defaults to "fixed".
        exponentiate (bool, optional): Whether to exponentiate the parameters. Defaults to False.
    """
    out = params.model_parameters(r_model, **kwargs)
    out = insight.standardize_names(out, style="broom")
    return out


@ensure_r_input
def is_mixed_model(r_model):
    """Check if a model is a mixed model using the implementation in `insight::is_mixed_model <https://easystats.github.io/insight/reference/is_mixed_model.html>`_

    Args:
        r_model (R model): `lm`, `glm`, `lmer`, or `glmer` model
    """
    return R2numpy(insight.is_mixed_model(r_model)).astype(bool)[0]


@ensure_r_input
def is_converged(r_model):
    """Check if a model is converged using the implementation in `insight::is_converged <https://easystats.github.io/insight/reference/is_converged.html>`_

    Args:
        r_model (R model): `lm`, `glm`, `lmer`, or `glmer` model

    Returns:
        tuple: Whether the model converged and the convergence message
    """
    convergence = insight.is_converged(r_model)
    did_converge = R2numpy(convergence).astype(bool)[0]
    message = f"Convergence status\n: {str(convergence)}"
    return did_converge, message
