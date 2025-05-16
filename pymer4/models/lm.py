from .base import model, requires_fit
from ..tidystats.stats import lm as lm_
from ..tidystats.tables import summary_lm_table
from ..tidystats.multimodel import boot


class lm(model):
    """Linear model for Ordinary Least Squares (OLS) regression.

    This class implements a linear regression model using OLS estimation. It inherits from the base model
    class and provides additional functionality specific to linear regression, including bootstrapping
    confidence intervals and summary statistics.

    Args:
        formula (str): R-style formula specifying the model.
        data (pandas.DataFrame): Input data for the model.

    """

    def __init__(self, formula, data, **kwargs):
        """Initialize the linear model.

        Args:
            formula (str): R-style formula specifying the model.
            data: Input data for the model.

        Attributes:
            conf_method (str, optional): Method used for confidence interval calculation.
            boot_type (str, optional): Type of bootstrap method used when conf_method is "boot".
            nboot (int, optional): Number of bootstrap samples.
            conf_level (float, optional): Confidence level for intervals.
        """
        super().__init__(formula, data, **kwargs)
        self._r_func = lm_
        self._summary_func = summary_lm_table

    def _2_get_tidy_summary(self, **kwargs):
        """Process and format the model summary.

        This method gets the base model summary and renames the 'statistic' column to 't_stat'
        for OLS regression output.
        """
        # Get base model tidy summary
        super()._2_get_tidy_summary(**kwargs)
        # Rename to t-stat for OLS
        self.result_fit = self.result_fit.rename({"statistic": "t_stat"})

    def _post_fit(
        self, conf_method, ci_type, nboot, conf_level, save_boots, add_df=True
    ):
        """Process post-fitting operations including adding dof to summary and confidence interval calculation.

        Args:
            conf_method (str): Method for confidence interval calculation.
            ci_type (str): How to calculate CIs; only applies to `conf_method='boot'`
            nboot (int): Number of bootstrap samples.
            conf_level (float): Confidence level for intervals.
            save_boots (bool): Whether to save bootstrap samples.
        """
        self.conf_method = conf_method
        self.ci_type = ci_type
        self.nboot = nboot
        self.conf_level = conf_level

        # Add df to result_fit
        if add_df:
            self.result_fit = self.result_fit.with_columns(
                df=self.result_fit_stats["df_residual"].item()
            ).select(
                "term",
                "estimate",
                "conf_low",
                "conf_high",
                "std_error",
                "t_stat",
                "df",
                "p_value",
            )

        # If bootsrapped CI's requested replace result_fit with bootstrapped CI's
        if conf_method == "boot":
            results = self._bootci(
                nboot=nboot,
                conf_method=ci_type,
                conf_level=conf_level,
                return_boots=save_boots,
            )
            if save_boots:
                self.result_boots = results[1]
                self.result_fit = self.result_fit.with_columns(
                    results[0].select("conf_low", "conf_high")
                )
            else:
                self.result_fit = self.result_fit.with_columns(
                    results.select("conf_low", "conf_high")
                )

    def fit(
        self,
        summary=False,
        conf_method="parametric",
        ci_type="bca",
        nboot=1000,
        conf_level=0.95,
        save_boots=False,
        **kwargs,
    ):
        """Fit a model using ``lm()`` in R.

        Args:
            summary (bool, optional): Whether to return the model summary. Defaults to False.
            conf_method (str, optional): Method for confidence interval calculation. Defaults to "parametric". Alternatively, ``"boot"`` for bootstrap CIs.
            ci_type (str, optional): How to calculate CIs; only applies to ``conf_method='boot'``. Defaults to ``"bca"``. Other options include ``"norm"``, ``"basic"``, ``"perc"``
            nboot (int, optional): Number of bootstrap samples. Defaults to 1000.
            conf_level (float, optional): Confidence level for intervals. Defaults to 0.95.
            save_boots (bool, optional): Whether to save bootstrap samples. Defaults to False.
            **kwargs: Additional arguments passed to the base model's fit method.

        Returns:
            ``GT``, optional: Model summary if ``summary=True``, otherwise ``None``.
        """
        super().fit(**kwargs)
        self._post_fit(conf_method, ci_type, nboot, conf_level, save_boots)
        if summary:
            return self.summary()

    @requires_fit
    def _bootci(
        self,
        nboot=1000,
        conf_method="bca",
        conf_level=0.95,
        return_boots=False,
    ):
        """Calculate bootstrap confidence intervals for model parameters.

        Args:
            nboot (int, optional): Number of bootstrap samples. Defaults to 1000.
            conf_method (str, optional): Type of bootstrap method. Defaults to "bca".
            conf_level (float, optional): Confidence level for intervals. Defaults to 0.95.
            return_boots (bool, optional): Whether to return bootstrap samples. Defaults to False.

        Returns:
            Union[polars.DataFrame, Tuple[polars.DataFrame, Any]]: If ``return_boots`` is ``True``,returns a tuple of (results, boots), otherwise returns just the results. Results include term, estimate, confidence intervals, and standard error.
        """
        self.nboot = nboot
        results = boot(
            self.data,
            self.r_model,
            self.formula,
            nboot,
            self.family,
            conf_method,
            conf_level,
            return_boots,
        )
        # return_boots is True
        if isinstance(results, tuple):
            boots = results[1]
            results = results[0]
            results = (
                results.drop("bias")
                .rename({"statistic": "estimate"})
                .select("term", "estimate", "conf_low", "conf_high", "std_error")
            )
            return results, boots
        return results
