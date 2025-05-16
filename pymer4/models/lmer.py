from .base import enable_logging, requires_fit
from .lm import lm
from ..tidystats.broom import tidy
from ..tidystats.lmerTest import ranef, lmer as lmer_
from ..tidystats.multimodel import coef, confint
from ..tidystats.tables import summary_lmm_table
from ..tidystats.stats import anova
from ..tidystats.emmeans_lib import joint_tests
from polars import DataFrame
from rpy2.robjects import NULL, NA_Real


class lmer(lm):
    """Linear mixed effects model estimated via ML/REML. Inherits from ``lm``.

    This class implements linear mixed effects models using Maximum Likelihood or
    Restricted Maximum Likelihood estimation. It extends the base linear model class
    to handle random effects and nested data structures.

    Args:
        formula (str): R-style formula specifying the model, including random effects
        data (DataFrame): Input data for the model
    """

    def __init__(self, formula, data, **kwargs):
        """Initialize the linear mixed effects model.

        Args:
            formula (str): R-style formula specifying the model, including random effects
            data (DataFrame): Input data for the model
        """
        super().__init__(formula, data, **kwargs)
        self._r_func = lmer_
        self._summary_func = summary_lmm_table
        # In addition to params like lm models
        # lmer models have fixed-effects ("BLUPs")
        # and random-effects ("deviances") for each cluster
        self.fixef = None
        self.ranef = None
        self.ranef_var = None

    def _2_get_tidy_summary(self, ddf_method="Satterthwaite", **kwargs):
        """Get summary of fixed effects using Satterthwaite degrees of freedom.

        Args:
            ddf_method (str): Method for computing denominator degrees of freedom. Defaults to "Satterthwaite"
        """
        self.result_fit = tidy(
            self.r_model,
            effects="fixed",
            conf_int=True,
            ddf_method=ddf_method,
            **kwargs,
        ).drop("effect", strict=False)

        # For glmer sub-class we don't have df column
        if self.family is None:
            cols = [
                "term",
                "estimate",
                "conf_low",
                "conf_high",
                "std_error",
                "statistic",
                "df",
                "p_value",
            ]
        else:
            cols = [
                "term",
                "estimate",
                "conf_low",
                "conf_high",
                "std_error",
                "statistic",
                "p_value",
            ]
        self.result_fit = self.result_fit.select(cols)
        self.result_fit = self.result_fit.rename({"statistic": "t_stat"})

    def _post_fit(self, conf_method, ci_type, nboot, conf_level):
        """Process post-fitting operations including confidence interval calculation and random effects extraction.

        Args:
            conf_method (str): Method for confidence interval calculation
            ci_type (str): Type of bootstrap confidence intervals
            nboot (int): Number of bootstrap samples
            conf_level (float): Confidence level for intervals
        """
        # Save meta-data
        self.conf_method = conf_method
        self.ci_type = ci_type
        self.nboot = nboot
        self.conf_level = conf_level

        # BLUPs
        self.fixef = coef(self.r_model)

        # RFX deviances
        self.ranef = ranef(self.r_model)

        # RFX variance-covariance
        self.ranef_var = tidy(self.r_model, effects="ran_pars").drop("effect")

        # If bootsrapped CI's requested replace result_fit with bootstrapped CI's
        if conf_method == "parametric":
            return
        else:
            fix_cis, rfx_cis = self._bootci(
                method=conf_method,
                conf_method=ci_type,
                conf_level=conf_level,
                nboot=nboot,
            )
            self.result_fit = self.result_fit.with_columns(
                conf_low=fix_cis[:, 0], conf_high=fix_cis[:, 1]
            )
            self.ranef_var = self.ranef_var.with_columns(
                conf_low=rfx_cis[:, 0], conf_high=rfx_cis[:, 1]
            )

    @enable_logging
    def fit(
        self,
        summary=False,
        conf_method="parametric",
        ci_type="perc",
        ddf_method="Satterthwaite",
        nboot=1000,
        conf_level=0.95,
        **kwargs,
    ):
        """Fit a linear mixed effects model using ``lmer()`` in R with Satterthwaite degrees of freedom and p-values calculated using ``lmerTest``. Unlike ``lm`` models, ``lmer`` models do not support saving bootstrap samples when ``conf_method="boot"``.

        Args:
            summary (bool, optional): Whether to return the model summary. Defaults to False
            conf_method (str, optional): Method for confidence interval calculation. Defaults to ``"parametric"``. Alternatively, ``"boot"`` for bootstrap CIs.
            ci_type (str, optional): Type of bootstrap confidence intervals. Defaults to "perc"
            ddf_method (str, optional): Method for computing denominator degrees of freedom. Defaults to "Satterthwaite"
            nboot (int, optional): Number of bootstrap samples. Defaults to 1000
            conf_level (float, optional): Confidence level for intervals. Defaults to 0.95
            **kwargs: Additional arguments passed to the R lmer function

        Returns:
            GT, optional: Model summary if ``summary=True``
        """

        super()._1_setup_R_model(**kwargs)
        self._2_get_tidy_summary(ddf_method)
        super()._3_get_coefs()
        super()._4_get_glance_fit_stats()
        super()._5_get_augment_fits_resids()
        super()._6_get_design_matrix()
        self.fitted = True
        self._post_fit(conf_method, ci_type, nboot, conf_level)
        if summary:
            return self.summary()

    @enable_logging
    def anova(self, auto_ss_3=True, **fitkwargs):
        """Calculate a Type-III ANOVA table for the model using ``joint_tests()`` in R.

        Args:
            auto_ss_3 (bool): whether to automatically use balanced contrasts when calculating the result via `joint_tests()`. When False, will use the contrasts specified with `set_contrasts()` which defaults to `"contr.treatment"` and R's `anova()` function; Default is True.
        """
        if not self.fitted:
            self.fit(**fitkwargs)
        if auto_ss_3:
            self.result_anova = joint_tests(
                self.r_model, mode="satterthwaite", lmer_df="satterthwaite"
            )
        else:
            self.result_anova = anova(self.r_model)

    def _bootci(
        self,
        method="profile",
        nboot=1000,
        conf_method="perc",
        conf_level=0.95,
        **kwargs,
    ):
        """Calculate confidence intervals for model parameters.

        Uses `lme4's confint.merMod <https://www.rdocumentation.org/packages/lme4/versions/1.1-36/topics/confint.merMod>`_
        to compute confidence intervals for both fixed and random effects parameters.
        Despite the name, this function can use non-bootstrap methods (profile, Wald).

        Args:
            method (str, optional): Method for computing intervals. Defaults to ``"profile"``. Alternatively, ``"Wald"`` or ``"boot"``.
            nboot (int, optional): Number of bootstrap samples. Defaults to 1000
            conf_method (str, optional): Type of bootstrap confidence intervals. Defaults to "perc"
            conf_level (float, optional): Confidence level for intervals. Defaults to 0.95
            **kwargs: Additional arguments passed to confint

        Returns:
            tuple: (fix_cis, rfx_cis) - Fixed effects CIs and random effects CIs as polars DataFrames
        """
        cis = confint(
            self.r_model,
            method=method,
            nsim=nboot,
            boot_type=conf_method,
            level=conf_level,
        )
        # Split fixed and random
        rfx_cis = cis[: self.ranef_var.height, :]
        fix_cis = cis[self.ranef_var.height :, :]
        return fix_cis, rfx_cis

    @enable_logging
    @requires_fit
    def emmeans(self, marginal_var, by=None, p_adjust="sidak", **kwargs):
        """Compute marginal means and/or contrasts between factor levels. ``marginal_var`` is the predictor whose levels will have means or contrasts computed. ``by`` is an optional predictor to marginalize over. If ``contrasts`` is not specified, only marginal means are returned

        Args:
            marginal_var (str): name of predictor to compute means or contrasts for
            by (str/list): additional predictors to marginalize over
            contrasts (str | 'pairwise' | 'poly' | dict | None, optional): how to specify comparison within `marginal_var`. Defaults to None.
            p_adjust (str): multiple comparisons adjustment method. One of: none, tukey (default), bonf, sidak, fdr, holm, dunnet, mvt (monte-carlo multi-variate T, aka exact tukey/dunnet).

        Returns:
            DataFrame: Table of marginal means or contrasts
        """

        return super().emmeans(
            marginal_var,
            by,
            mode="satterthwaite",
            lmer_df="satterthwaite",
            lmerTest_limit=999999,
            p_adjust=p_adjust,
            **kwargs,
        )

    @requires_fit
    def predict(self, data: DataFrame, use_rfx=True, **kwargs):
        """Make predictions using new data.

        Args:
            data (DataFrame): Input data for predictions
            use_rfx (bool, optional): Whether to include random effects in predictions. Defaults to True. Equivalent to ``re.form = NULL`` in R if True, ``re.form = NA`` if False
            **kwargs: Additional arguments passed to predict function

        Returns:
            ndarray: Predicted values
        """
        re_form = NULL if use_rfx else NA_Real
        return super().predict(data, re_form=re_form, **kwargs)

    @requires_fit
    def simulate(self, nsim: int = 1, use_rfx=True, **kwargs):
        """Simulate values from the fitted model.

        Args:
            nsim (int, optional): Number of simulations to run. Defaults to 1
            use_rfx (bool, optional): Whether to include random effects in simulations. Defaults to True.
                Equivalent to ``re.form = NULL`` in R if True, ``re.form = NA`` if False
            **kwargs: Additional arguments passed to simulate function

        Returns:
            DataFrame: Simulated values with the same number of rows as the original data
                and columns equal to nsim
        """
        re_form = NULL if use_rfx else NA_Real
        return super().simulate(nsim, re_form=re_form, **kwargs)
