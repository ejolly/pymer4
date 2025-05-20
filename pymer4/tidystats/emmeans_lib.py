from rpy2.robjects.packages import importr
import rpy2.robjects as ro
from .tibble import as_tibble
from .bridge import ensure_r_input, ensure_py_output

__all__ = ["joint_tests", "emmeans", "emtrends", "ref_grid"]

lib_emmeans = importr("emmeans")


@ensure_py_output
@ensure_r_input
def joint_tests(model, **kwargs):
    """Compute ANOVA-style F-tests using `emmeans::joint_tests <https://www.rdocumentation.org/packages/emmeans/versions/1.3.5.1/topics/joint_tests>`_

    Args:
        model (R model): `lm`, `glm`, `lmer` model

    Returns:
        DataFrame: F-statistics table of main effects/interactions
    """
    return as_tibble(lib_emmeans.joint_tests(model, **kwargs))


@ensure_py_output
@ensure_r_input
def emmeans(model, specs, contrasts: str | dict | None = None, **kwargs):
    """This function combines functionality from `emmeans::emmeans <https://www.rdocumentation.org/packages/emmeans/versions/1.3.5.1/topics/emmeans>`_  and `emmeans::contrast <https://www.rdocumentation.org/packages/emmeans/versions/1.3.5.1/topics/contrast>`_, by first generating a grid and then optionally computing contrasts over it if ``contrasts`` is not None.

    Args:
        model (R model): `lm`, `glm`, `lmer` model
        specs (str): name of predictor
        by (str/list): additional predictors to subset by
        contrasts (str | 'pairwise' | 'poly' | dict | None, optional): how to specify comparisonwithin `specs`. Defaults to None.

    Returns:
        DataFrame: Table of marginal effects and/or means
    """

    emm_grid = lib_emmeans.emmeans(model, spec=specs, **kwargs)

    if contrasts is None:
        return as_tibble(emm_grid)
    else:
        # By is already set by the grid above
        _ = kwargs.pop("by", None)
        interaction = kwargs.pop("interaction", None)
        if interaction is None:
            return as_tibble(lib_emmeans.contrast(emm_grid, contrasts, **kwargs))
        else:
            # We make multiple calls to contrast because the recursive nature of interaction kwarg, doesn't seem to be supported by rpy2
            marginals = lib_emmeans.contrast(emm_grid, contrasts, **kwargs)
            return as_tibble(
                lib_emmeans.contrast(
                    marginals, method=interaction, by=ro.NULL, **kwargs
                )
            )


@ensure_py_output
@ensure_r_input
def emtrends(model, contrasts: str | dict | None = None, **kwargs):
    """This function combines functionality from `emmeans::emtrends <https://www.rdocumentation.org/packages/emmeans/versions/1.3.5.1/topics/emtrends>`_ and `emmeans::contrast <https://www.rdocumentation.org/packages/emmeans/versions/1.3.5.1/topics/contrast>`_, by first generating a grid and then optionally computing contrasts over it if ``contrasts`` is not None.

    Args:
        model (R model): `lm`, `glm`, `lmer` model
        specs (str): name of predictor
        by (str/list): additional predictors to subset by
        contrasts (str | 'pairwise' | 'poly' | dict | None, optional): how to specify comparisonwithin `specs`. Defaults to None.

    Returns:
        DataFrame: Table of marginal effects and/or means
    """

    emm_grid = lib_emmeans.emtrends(model, **kwargs)

    if contrasts is None:
        return as_tibble(emm_grid)
    else:
        # By is already set by the grid above
        _ = kwargs.pop("by", None)
        interaction = kwargs.pop("interaction", None)
        if interaction is None:
            return as_tibble(lib_emmeans.contrast(emm_grid, contrasts, **kwargs))
        else:
            # We make multiple calls to contrast because the recursive nature of interaction kwarg, doesn't seem to be supported by rpy2
            marginals = lib_emmeans.contrast(emm_grid, contrasts, **kwargs)
            return as_tibble(
                lib_emmeans.contrast(
                    marginals, method=interaction, by=ro.NULL, **kwargs
                )
            )


@ensure_py_output
@ensure_r_input
def ref_grid(model, *args, **kwargs):
    """Create a reference grid of model predictions. Uses `emmeans::ref_grid <https://www.rdocumentation.org/packages/emmeans/versions/1.3.5.1/topics/ref_grid>`_ and `emmeans::summary_emmGrid <https://www.rdocumentation.org/packages/emmeans/versions/1.3.5.1/topics/summary_emmGrid>`_."""
    ref = lib_emmeans.ref_grid(model, *args, **kwargs)
    grid_summary = lib_emmeans.summary_emmGrid(ref)
    return grid_summary
