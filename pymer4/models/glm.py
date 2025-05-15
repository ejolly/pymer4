from .base import requires_fit
from .lm import lm
from polars import col
from ..tidystats.stats import glm as glm_
from ..tidystats.tables import summary_glm_table
from ..tidystats.broom import augment
from ..tidystats.plutils import join_on_common_cols
from ..tidystats.multimodel import predict
from ..rfuncs import get_summary
from ..expressions import logit2odds
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

    def __init__(self, formula, data, family="gaussian", link="default"):
        super().__init__(formula, data)
        self._r_func = glm_
        self._summary_func = summary_glm_table
        self.family = family
        self.link = link
        self._type_predict = "response"
        self.result_fit_odds = None
        self.result_fit_probs = None
        self.result_bootci_odds = None
        self.result_bootci_probs = None
        self._configure_family_link()

    def _configure_family_link(self):
        """Configure the R family and link function objects for the model."""
        family = getattr(lib_stats, self.family)
        self._r_family_link = (
            family() if self.link == "default" else family(link=self.link)
        )
        self._convert_logit2odds = self.family == "binomial" and self.link in [
            "default",
            "logit",
        ]

    def __repr__(self):
        """Return string representation of the model.

        Returns:
            str: String representation including class name, fitted status, formula, family and link
        """
        out = "{}(fitted={}, formula={}, family={}, link={})".format(
            self.__class__.__module__,
            self.fitted,
            self.formula,
            self.family,
            self.link,
        )
        return out

    def _1_setup_R_model(self, **kwargs):
        """Set up the R model with family and link function.

        Args:
            **kwargs: Additional keyword arguments passed to the R GLM function
        """
        self.r_model = self._r_func(
            self.formula, self.data, family=self._r_family_link, **kwargs
        )

    def _2_get_tidy_summary(self, **kwargs):
        """Gets summary of fixed effects and adds additional attributes with odds-scale estimates for binomial models with logit link."""
        if self._convert_logit2odds:
            # Get odds-scale estimates first and save them
            super()._2_get_tidy_summary(exponentiate=True, **kwargs)
            self.result_fit_odds = self.result_fit.clone().rename({"t_stat": "z_stat"})

            # Rerun for unntransformed logit scale estimates
            super()._2_get_tidy_summary(**kwargs)

        else:
            super()._2_get_tidy_summary(**kwargs)

        # Rename to z-stat for GLM
        self.result_fit = self.result_fit.rename({"t_stat": "z_stat"})

    def _3_get_coefs(self):
        """Extract model coefficients (fixed effects)."""

        super()._3_get_coefs()
        if self._convert_logit2odds:
            self.params_odds = self.params.select(
                col("Parameter"), col("Estimate").map_batches(logit2odds)
            )

    def _5_get_augment_fits_resids(self, type_predict="response"):
        """Add model predictions and residuals accounting for the link function.

        Args:
            type_predict (str): Type of prediction to compute ("response" or "link"). Defaults to "response"
        """
        # Add predictions but incorporate the link scale
        self._type_predict = type_predict
        self.data = join_on_common_cols(
            self.data,
            augment(self.r_model, type_predict=type_predict),
        )

    def fit(
        self,
        summary=False,
        conf_method="parametric",
        type_predict="response",
        ci_type="bca",
        nboot=1000,
        conf_level=0.95,
        save_boots=False,
        show_odds=False,
        **kwargs,
    ):
        """Fit a GLM using ``glm()`` in R.

        Args:
            summary (bool, optional): Whether to return the model summary. Defaults to False
            conf_method (str, optional): Method for confidence interval calculation. Defaults to "parametric". Alternatively, ``"boot"`` for bootstrap CIs.
            type_predict (str, optional): Type of prediction to compute ("response" or "link"). Defaults to "response"
            ci_type (str, optional): How to calculate CIs; only applies to ``conf_method='boot'``. Defaults to ``"bca"``. Other options include ``"norm"``, ``"basic"``, ``"perc"``
            nboot (int, optional): Number of bootstrap samples. Defaults to 1000
            conf_level (float, optional): Confidence level for intervals. Defaults to 0.95
            save_boots (bool, optional): Whether to save bootstrap samples. Defaults to False
            **kwargs: Additional arguments passed to the R GLM function

        Returns:
            GT, optional: Model summary if ``summary=True``
        """
        # 1) Setup R model
        self._1_setup_R_model(**kwargs)

        # 2) Get tidy fixed effects summary table
        self._2_get_tidy_summary()

        # 3) Get fixed effects; coefs for lms; BLUPs for lmms
        self._3_get_coefs()
        super()._4_get_glance_fit_stats()

        # 5) Add predictions to data accounting for link
        self._5_get_augment_fits_resids(type_predict=type_predict)

        # 6) Get model design matrix
        super()._6_get_design_matrix()

        self.fitted = True

        # Handle df and bootstrapped CIs
        super()._post_fit(
            conf_method, ci_type, nboot, conf_level, save_boots, add_df=False
        )

        if summary:
            return self.summary(show_odds=show_odds)

    @requires_fit
    def summary(self, show_odds=False, pretty=True, decimals=3):
        """Print a nicely formatted summary table of model results.

        For binomial models with logit link, can optionally show results in probability
        or odds scale instead of log-odds.

        Args:
            show_odds (bool): Whether to show results in odds scale. Defaults to False
            decimals (int): Number of decimal places to round to. Defaults to 3

        Returns:
            GT: A formatted summary table
        """
        # Summary table can switch between coefficient scales for binomial models
        if pretty:
            return self._summary_func(self, show_odds, decimals)
        print(get_summary(self.r_model))

    @requires_fit
    def predict(self, *args, type_predict="response", **kwargs):
        """Make predictions from the model.

        Args:
            data (DataFrame): Data to make predictions on
            type_predict (str): Type of prediction to compute ("response" or "link"). Defaults to "response"
            **kwargs: Additional keyword arguments passed to predict function

        Returns:
            ndarray: Predicted values
        """
        return predict(self.r_model, *args, type=type_predict, **kwargs)
