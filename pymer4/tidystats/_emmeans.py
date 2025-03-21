from rpy2.robjects.packages import importr
import rpy2.robjects as ro
from ._tidyverse import as_tibble

__all__ = ["joint_tests", "emmeans"]

lib_emmeans = importr("emmeans")


def joint_tests(model):
    """Compute joint tests of terms in a model [reference](https://rdrr.io/cran/emmeans/man/joint_tests.html)

    Args:
        model (R model): `lm`, `glm`, `lmer` model

    Returns:
        DataFrame: F-statistics table of main effects/interactions
    """
    return as_tibble(lib_emmeans.joint_tests(model))


def emmeans(model, specs, contrasts: str | dict | None = None, **kwargs):
    """This function combines `emmeans` and `contrast` from the emmeans package, by first generating a grid and then computing contrasts over it

    Args:
        model (R model): `lm`, `glm`, `lmer` model
        specs (str): name of predictor
        contrasts (str | 'pairwise' | dict | None, optional): how to specify comparisonwithin `specs`. Defaults to None.

    Returns:
        DataFrame: Table of marginal effects and/or means
    """

    emm_grid = lib_emmeans.emmeans(model, spec=specs, **kwargs)

    if contrasts is None:
        return as_tibble(emm_grid)

    if isinstance(contrasts, dict):
        r_contrasts = dict()

        # Convert Python lists with numeric contrast codes
        # two R float vectors
        for k, v in contrasts.items():
            r_contrasts[k] = ro.FloatVector(v)

        # Convert Python dict to R ListVector
        r_contrasts = ro.ListVector(r_contrasts)

        return as_tibble(lib_emmeans.contrast(emm_grid, r_contrasts))

    if isinstance(contrasts, str):
        return as_tibble(lib_emmeans.contrast(emm_grid, contrasts))
