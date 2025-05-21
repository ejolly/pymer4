"""
Pymer4 model types.
"""

import numpy as np
from polars import col, when, selectors as cs
from .lm import lm
from .glm import glm
from .lmer import lmer
from .glmer import glmer
from ..tidystats.stats import anova
from ..tidystats.tables import compare_anova_table

__all__ = ["lm", "glm", "lmer", "glmer", "compare"]


def compare(*models, as_dataframe=False, test="F"):
    """Compare 2 or models using an F-test or Likelihood-Ratio-Test. Uses the `anova <https://www.rdocumentation.org/packages/stats/versions/3.6.2/topics/anova>`_ function in R.

    Args:
        as_dataframe (bool, optional): Return a dataframe instead of a table. Defaults to False.
        test (str, optional): Test to use for model comparison. Defaults to 'F' for lm/glm models and 'LRT' for lmer/glmer models

    Returns:
        result (polars.DataFrame or GreatTables): A dataframe with the model comparison results
    """

    # Refit if any models are not fit or were fit via different methods
    if not all(m.fitted for m in models) or any(
        m.result_boots is not None for m in models
    ):
        for m in models:
            # Drop augmented columns
            m.data = m.data.select(m._data_cols)
            m.fit()
    if test is None:
        if any(isinstance(m, (lmer, glmer)) for m in models):
            test = "LRT"
        else:
            test = "F"

    out = anova(*models, test=test)
    aics = np.array([m.result_fit_stats["AIC"].item() for m in models])
    bics = np.array([m.result_fit_stats["BIC"].item() for m in models])
    log_likelihoods = np.array([m.result_fit_stats["logLik"].item() for m in models])
    out = out.with_columns(
        AIC=aics,
        BIC=bics,
        logLik=log_likelihoods,
    ).select(col("AIC", "BIC", "logLik"), cs.exclude(["AIC", "BIC", "logLik"]))

    # Handle non-numeric p-vals when we can't perform a test due to same number of
    # parameters
    pcol = out.columns[-1]
    out = out.with_columns(
        when(col(pcol).lt(0.0)).then(None).otherwise(col(pcol)).alias(pcol).cast(float)
    )

    return out if as_dataframe else compare_anova_table(out, *models)
