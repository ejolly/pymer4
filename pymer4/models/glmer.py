from .lmer import lmer
from .base import enable_logging
from ..tidystats.lmerTest import glmer as glmer_
from ..tidystats.tables import summary_glmm_table
from polars import DataFrame
from rpy2.robjects.packages import importr

lib_stats = importr("stats")


class glmer(lmer):
    """Generalized linear mixed effects model estimated via ML/REML. Inherits from ``lmer``.

    This class implements generalized linear mixed effects models using Maximum Likelihood
    or Restricted Maximum Likelihood estimation. It extends the linear mixed effects model
    class to handle different response distributions and link functions while accounting
    for random effects.

    Args:
        formula (str): R-style formula specifying the model, including random effects
        data (DataFrame): Input data for the model
        family (str): Response distribution family (e.g. "gaussian", "binomial"). Defaults to "gaussian"
        link (str): Link function to use. Defaults to "default" which uses the canonical link for each family
    """

    def __init__(self, formula, data, family="gaussian", link="default", **kwargs):
        super().__init__(formula, data, family=family, link=link, **kwargs)
        self._r_func = glmer_
        self._summary_func = summary_glmm_table

    def _bootstrap(self, exponentiate=False, **kwargs):
        """Call super()._bootstrap() with exponentiate=True if True"""
        super()._bootstrap(exponentiate=exponentiate, **kwargs)

    @enable_logging
    def fit(
        self,
        exponentiate=False,
        summary=False,
        conf_method="wald",
        nboot=1000,
        save_boots=True,
        type_predict="response",
        parallel="multicore",
        ncpus=4,
        conf_type="perc",
        **kwargs,
    ):
        """Fit a generalized linear mixed effects model using ``glmer()`` in R.

        Args:
            summary (bool, optional): Whether to return the model summary. Defaults to False
            conf_method (str, optional): Method for confidence interval calculation. Defaults to "parametric"
            ci_type (str, optional): Type of bootstrap confidence intervals. Defaults to "perc"
            ddf_method (str, optional): Method for computing denominator degrees of freedom. Defaults to "Satterthwaite"
            nboot (int, optional): Number of bootstrap samples. Defaults to 1000
            conf_level (float, optional): Confidence level for intervals. Defaults to 0.95
            type_predict (str, optional): Type of prediction to compute ("response" or "link"). Defaults to "response"

        Returns:
            GT, optional: Model summary if ``summary=True``
        """
        super().fit(
            exponentiate=exponentiate,
            conf_method=conf_method,
            nboot=nboot,
            save_boots=save_boots,
            parallel=parallel,
            ncpus=ncpus,
            conf_type=conf_type,
            type_predict=type_predict,
            **kwargs,
        )
        if summary:
            return self.summary()

    def predict(self, data: DataFrame, use_rfx=True, type_predict="response", **kwargs):
        """Make predictions using new data accounting for the link function.

        Args:
            data (DataFrame): Input data for predictions
            use_rfx (bool, optional): Whether to include random effects in predictions. Defaults to True. Equivalent to ``re.form = NULL`` in R if True, ``re.form = NA`` if False
            type_predict (str, optional): Type of prediction to compute ("response" or "link"). Defaults to "response"
            **kwargs: Additional arguments passed to predict function

        Returns:
            ndarray: Predicted values
        """
        return super().predict(data, use_rfx=use_rfx, type=type_predict, **kwargs)
