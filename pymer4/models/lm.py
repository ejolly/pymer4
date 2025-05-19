from .base import model
from ..tidystats.stats import lm as lm_
from ..tidystats.tables import summary_lm_table
from ..tidystats.easystats import bootstrap_model
import polars as pl
from polars import col


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

    def _bootstrap(
        self,
        nboot,
        save_boots,
        parallel="multicore",
        ncpus=4,
        **kwargs,
    ):
        """
        Get bootstrap estimates of model parameters. We use the implementation in `easystats::bootstrap_model()` which handles calling [`boot`](https://rdrr.io/cran/boot/man/boot.html) for us and aggregating results. This is a less error prone that using `boot()` directly as it requires an R function string via rpy2. The downside we that we can only offer percentile confidence intervals calculated from the bootstrap distribution.

        Args:
            nboot (int): Number of bootstrap samples.
            save_boots (bool): Whether to save bootstrap samples.
            **kwargs: Additional arguments passed to `bootstrap_model`.

        """
        result_boot = bootstrap_model(
            self.r_model, nboot=nboot, parallel=parallel, ncpus=ncpus, **kwargs
        )

        boot_cis = (
            result_boot.select(
                pl.quantile("*", 0.025).name.suffix("_lower"),
                pl.quantile("*", 0.975).name.suffix("_upper"),
            )
            .unpivot()
            .with_columns(
                col("variable")
                .str.split_exact("_", 1)
                .struct.rename_fields(["term", "ci_bound"])
                .struct.unnest()
            )
            .drop("variable")
            .select("term", "ci_bound", "value")
        )
        lower = boot_cis.filter(col("ci_bound") == "lower").select("value").to_series()
        upper = boot_cis.filter(col("ci_bound") == "upper").select("value").to_series()
        self.result_fit = self.result_fit.with_columns(
            conf_low=lower,
            conf_high=upper,
        )
        if save_boots:
            self.result_boots = result_boot

    def fit(
        self,
        summary=False,
        conf_method="wald",
        nboot=1000,
        save_boots=True,
        parallel="multicore",
        ncpus=4,
        conf_type="perc",
        **kwargs,
    ):
        """Fit a model using ``lm()`` in R.

        Args:
            summary (bool, optional): Whether to return the model summary. Defaults to False.
            conf_method (str, optional): Method for confidence interval calculation. Defaults to "wald". Alternatively, ``"boot"`` for bootstrap CIs.
            nboot (int, optional): Number of bootstrap samples. Defaults to 1000.
            save_boots (bool, optional): Whether to save bootstrap samples. Defaults to True.
            parallel (str, optional): Parallelization for bootstrapping. Defaults to "multicore"
            ncpus (int, optional): Number of cores to use for parallelization. Defaults to 4
            conf_type (str, optional): Type of confidence interval to calculate. Defaults to "perc"
            **kwargs: Additional arguments to ``easystats::model_parameters()``

        Returns:
            ``GT``, optional: Model summary if ``summary=True``, otherwise ``None``.
        """
        super().fit(
            conf_method=conf_method,
            nboot=nboot,
            save_boots=save_boots,
            parallel=parallel,
            ncpus=ncpus,
            conf_type=conf_type,
            **kwargs,
        )
        if conf_method == "boot":
            self._bootstrap(nboot, save_boots, **kwargs)
        if summary:
            return self.summary()
