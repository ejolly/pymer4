from .lmer import lmer
from .base import requires_fit, enable_logging
from ..tidystats.lmerTest import glmer as glmer_
from ..tidystats.tables import summary_glmm_table
from ..tidystats.multimodel import predict
from ..rfuncs import get_summary
from ..expressions import logit2odds, logit2prob
from polars import col
import polars.selectors as cs
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

    def __init__(self, formula, data, family="gaussian", link="default"):
        super().__init__(formula, data)
        self._r_func = glmer_
        self._summary_func = summary_glmm_table
        self.family = family.capitalize() if family == "gamma" else family
        self.link = link
        self._type_predict = "response"
        self.result_fit_odds = None
        self.result_fit_probs = None
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
            **kwargs: Additional keyword arguments passed to the R GLMM function
        """
        self.r_model = self._r_func(
            self.formula, self.data, family=self._r_family_link, **kwargs
        )

    def _2_get_tidy_summary(self, ddf_method="Satterthwaite", **kwargs):
        """Get summary of fixed effects and rename statistic column to z_stat for GLMMs.

        Args:
            ddf_method (str): Method for computing denominator degrees of freedom. Defaults to "Satterthwaite"
        """
        if self._convert_logit2odds:
            # Get odds-scale estimates first and save them
            super()._2_get_tidy_summary(
                ddf_method=ddf_method,
                exponentiate=True,
                exponentiate_ran_coefs=True,
                **kwargs,
            )
            self.result_fit_odds = self.result_fit.clone().rename({"t_stat": "z_stat"})

            # Rerun for unntransformed logit scale estimates
            super()._2_get_tidy_summary(ddf_method=ddf_method, **kwargs)
        else:
            super()._2_get_tidy_summary(ddf_method=ddf_method, **kwargs)

        # Rename to z-stat for GLM
        self.result_fit = self.result_fit.rename({"t_stat": "z_stat"})

    def _3_get_coefs(self):
        """Extract model coefficients (fixed effects)."""

        super()._3_get_coefs()

        # Note: we can't update .params until super()._3_get_coefs() is called
        # which is why we append to the method call here
        if self._convert_logit2odds:
            self.params_odds = self.params.select(
                col("Parameter"), col("Estimate").map_batches(logit2odds)
            )

    def _5_get_augment_fits_resids(self, type_predict="response"):
        """Add model predictions and residuals accounting for the link function.

        For binomial models with logit link, transforms predictions to probability scale
        when type_predict="response".

        Args:
            type_predict (str): Type of prediction to compute ("response" or "link"). Defaults to "response"
        """
        # Add predictions but incorporate the link scale; only for logit model
        # broom offers this for lm/glm, but broom.mixed doesn't so we do it manually
        super()._5_get_augment_fits_resids()

        if type_predict == "response" and self._convert_logit2odds:
            self.data = self.data.with_columns(logit2prob(col("fitted")))
            self._type_predict = "response"

    def _post_fit(self, conf_method, ci_type, nboot, conf_level):
        super()._post_fit(conf_method, ci_type, nboot, conf_level)

        # Convert rfx to odds scale if binomial
        if self._convert_logit2odds:
            self.ranef_odds = self.ranef.with_columns(logit2odds(cs.exclude("level")))
            self.fixef_odds = self.fixef.with_columns(logit2odds(cs.exclude("level")))

    @enable_logging
    def fit(
        self,
        summary=False,
        conf_method="parametric",
        ci_type="perc",
        ddf_method="Satterthwaite",
        nboot=1000,
        conf_level=0.95,
        type_predict="response",
        show_odds=False,
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
            **kwargs: Additional arguments passed to the R glmer function

        Returns:
            GT, optional: Model summary if ``summary=True``
        """
        # 1) Setup R model; overwrite like glm does for lm
        self._1_setup_R_model(**kwargs)

        # 2) Get tidy fixed effects summary table; overwrite like glm does for lm
        self._2_get_tidy_summary(ddf_method=ddf_method)

        # 3) Get fixed effects; coefs for lms; BLUPs for lmms; call lmer's methods
        self._3_get_coefs()
        super()._4_get_glance_fit_stats()

        # 5) Add predictions to data accounting for link; overwrite like glm does for lm
        self._5_get_augment_fits_resids(type_predict=type_predict)

        # 6) Get model design matrix; call lmer's method
        super()._6_get_design_matrix()

        # Call lmer's method
        self.fitted = True
        self._post_fit(conf_method, ci_type, nboot, conf_level)

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
