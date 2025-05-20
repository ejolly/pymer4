from .lm import lm
from polars import DataFrame
from ..tidystats.stats import glm as glm_
from ..tidystats.tables import summary_glm_table
from rpy2.robjects.packages import importr

lib_stats = importr("stats")


class glm(lm):
    """Generalized linear model estimated via MLE. Inherits from lm.

    This class implements generalized linear models using Maximum Likelihood Estimation.
    It extends the base linear model class to handle different response distributions
    and link functions.

    Args:
        formula (str): R-style formula specifying the model
        data (DataFrame): Input data for the model
        family (str): Response distribution family (e.g. "gaussian", "binomial"). Defaults to "gaussian"
        link (str): Link function to use. Defaults to "default" which uses the canonical link for each family
    """

    def __init__(self, formula, data, family="gaussian", link="default", **kwargs):
        super().__init__(formula, data, family=family, link=link, **kwargs)
        self._r_func = glm_
        self._summary_func = summary_glm_table

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
        """Fit a GLM using ``glm()`` in R.

        Args:
            exponentiate (bool, optional): Whether to exponentiate the parameter estimates to the odds scale. Defaults to False
            summary (bool, optional): Whether to return the model summary. Defaults to False
            conf_method (str, optional): Method for confidence interval calculation. Defaults to "wald". Alternatively, ``"boot"`` for bootstrap CIs.
            nboot (int, optional): Number of bootstrap samples. Defaults to 1000
            save_boots (bool, optional): Whether to save bootstrap samples. Defaults to True
            type_predict (str, optional): Type of prediction to compute ("response" or "link"). Defaults to "response"
            parallel (str, optional): Parallelization for bootstrapping. Defaults to "multicore"
            ncpus (int, optional): Number of cores to use for parallelization. Defaults to 4
            conf_type (str, optional): Type of confidence interval to calculate. Defaults to "perc"
            **kwargs: Additional arguments passed to the R GLM function

        Returns:
            GT, optional: Model summary if ``summary=True``
        """

        super().fit(
            conf_method=conf_method,
            exponentiate=exponentiate,
            nboot=nboot,
            save_boots=save_boots,
            type_predict=type_predict,
            parallel=parallel,
            ncpus=ncpus,
            conf_type=conf_type,
            **kwargs,
        )
        if summary:
            return self.summary()

    def predict(self, data: DataFrame, type_predict="response", **kwargs):
        """Make predictions from the model accounting for the link function.

        Args:
            data (DataFrame): Data to make predictions on
            type_predict (str, optional): Type of prediction to compute ("response" or "link"). Defaults to "response"
            **kwargs: Additional keyword arguments passed to predict function

        Returns:
            ndarray: Predicted values
        """
        return super().predict(data, type=type_predict, **kwargs)
