from .base import enable_logging, requires_fit, model
from ..tidystats.broom import tidy
from ..tidystats.lmerTest import ranef, lmer as lmer_, bootMer
from ..tidystats.multimodel import coef
from ..tidystats.easystats import get_param_names, is_converged
from ..tidystats.tables import summary_lmm_table
from ..expressions import logit2odds
from polars import DataFrame, col
import polars.selectors as cs
from rpy2.robjects import NULL, NA_Real


class lmer(model):
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
        self.fixef = None
        self.ranef = None
        self.ranef_var = None
        self.convergence_status = None

    def _handle_rfx(self, **kwargs):
        """Sets `.ranef_var` using ``broom.mixed::tidy()`` and ``lme4::ranef()`` and ``lme4::coef()`` to get random effects and BLUPs. Manually exponentiates random effects if ``exponentiate=True`` since ``broom.mixed::tidy()`` does not do this."""

        self.ranef_var = tidy(
            self.r_model, effects="ran_pars", conf_int=True, **kwargs
        ).drop("effect", strict=False)
        self.ranef = ranef(self.r_model)
        self.fixef = coef(self.r_model)

        # Ensure multiple rfx are returned as a dict
        if isinstance(self.fixef, list):
            fixed_names, random_names = get_param_names(self.r_model)
            self.fixef = dict(zip(random_names.keys(), self.fixef))
            self.ranef = dict(zip(random_names.keys(), self.ranef))

        # Exponentiate params if requested
        exponentiate = kwargs.get("exponentiate", False)
        if exponentiate:
            if isinstance(self.fixef, dict):
                self.fixef = {
                    k: v.with_columns(col("level"), logit2odds(cs.exclude("level")))
                    for k, v in self.fixef.items()
                }
            else:
                self.fixef = self.fixef.with_columns(
                    col("level"), logit2odds(cs.exclude("level"))
                )
            if isinstance(self.ranef, dict):
                self.ranef = {
                    k: v.with_columns(col("level"), logit2odds(cs.exclude("level")))
                    for k, v in self.ranef.items()
                }
            else:
                self.ranef = self.ranef.with_columns(
                    col("level"), logit2odds(cs.exclude("level"))
                )

    def _bootstrap(
        self,
        nboot,
        save_boots,
        conf_method="perc",
        parallel="multicore",
        ncpus=4,
        conf_level=0.95,
        **kwargs,
    ):
        """Get bootstrapped estimates of model parameters using `lme4's confint.merMod <https://www.rdocumentation.org/packages/lme4/versions/1.1-36/topics/confint.merMod>`_. Unlike with `lm()`, we don't use `easystats` functions because they don't return the full bootstrap distribution for rfx, only ffx. We use `tidy` to summarize the bootstrap distributions and can therefore can use all the `conf_method` that it supports (e.g. `"perc"`, `"bca"`, `"norm"`, `"basic"`).

        Args:
            conf_method (str, optional): Type of bootstrap confidence intervals. Defaults to "perc"
            nboot (int, optional): Number of bootstrap samples. Defaults to 1000
            parallel (str, optional): Parallelization method. Defaults to "multicore"
            ncpus (int, optional): Number of CPUs to use. Defaults to 4
            conf_level (float, optional): Confidence level for intervals. Defaults to 0.95
            save_boots (bool, optional): Whether to save bootstrap samples. Defaults to True
            **kwargs: Additional arguments passed to confint

        Returns:
            tuple: (fix_cis, rfx_cis) - Fixed effects CIs and random effects CIs as polars DataFrames
        """
        cis, boots = bootMer(
            self.r_model,
            nsim=nboot,
            conf_level=conf_level,
            conf_method=conf_method,
            parallel=parallel,
            ncpus=ncpus,
            save_boots=save_boots,
            **kwargs,
        )
        self.cis = cis

        # Fixed CIs
        fixed_names = self.params["term"].to_list()
        fixed_lower = (
            cis.filter(col("term").is_in(fixed_names)).select("conf_low").to_series()
        )
        fixed_upper = (
            cis.filter(col("term").is_in(fixed_names)).select("conf_high").to_series()
        )
        self.result_fit = self.result_fit.with_columns(
            conf_low=fixed_lower,
            conf_high=fixed_upper,
        )

        # Drop fixed-effect rows and split out term col to term and group cols
        ranef_cis = (
            cis.filter(~col("term").is_in(fixed_names))
            .with_columns(
                col("term")
                .str.split_exact("___", 2)
                .explode()
                .struct.rename_fields(["term", "group"])
                .struct.unnest()
            )
            .select("group", "term", "conf_low", "conf_high")
        )
        self.ranef_var = self.ranef_var.drop("conf_low", "conf_high").join(
            ranef_cis, on=["term", "group"]
        )

        if save_boots:
            self.result_boots = boots

    @enable_logging
    def fit(
        self,
        summary=False,
        conf_method="satterthwaite",
        nboot=1000,
        save_boots=True,
        parallel="multicore",
        ncpus=4,
        conf_type="perc",
        bootMer_kwargs={},
        **kwargs,
    ):
        """Fit a linear mixed effects model using ``lmer()`` in R with Satterthwaite degrees of freedom and p-values calculated using ``lmerTest``.

        Args:
            summary (bool, optional): Whether to return the model summary. Defaults to False
            conf_method (str, optional): Method for confidence interval calculation. Defaults to ``"satterthwaite"``. Alternatively, ``"boot"`` for bootstrap CIs.
            nboot (int, optional): Number of bootstrap samples. Defaults to 1000
            parallel (str, optional): Parallelization for bootstrapping. Defaults to "multicore"
            ncpus (int, optional): Number of cores to use for parallelization. Defaults to 4
            conf_type (str, optional): Type of confidence interval to calculate. Defaults to "perc"

        Returns:
            GT, optional: Model summary if ``summary=True``
        """

        # Use super to get fixed effects via easystats::model_parameters()
        if conf_method == "boot":
            if self.family is None:
                default_conf_method = "satterthwaite"
            else:
                default_conf_method = "wald"
        super().fit(
            conf_method=conf_method if conf_method != "boot" else default_conf_method,
            effects="fixed",
            ci_random=False,
            parallel=parallel,
            ncpus=ncpus,
            conf_type=conf_type,
            **kwargs,
        )

        # Store the conf_method in the fit_kwargs since we overwrite it in the super call
        self._fit_kwargs["conf_method"] = conf_method

        # Get random effects
        self._handle_rfx(**kwargs)

        if conf_method == "boot":
            self._bootstrap(
                nboot=nboot,
                save_boots=save_boots,
                conf_method=conf_type,
                parallel=parallel,
                ncpus=ncpus,
                **bootMer_kwargs,
            )

        # Handle convergence & singularity warnings
        did_converge, message = is_converged(self.r_model)
        self.convergence_status = message
        if not did_converge:
            self.r_console.append(message)

        if summary:
            return self.summary()

    @enable_logging
    def anova(
        self,
        summary=False,
        auto_ss_3=True,
        jointtest_kwargs={"mode": "satterthwaite", "lmer_df": "satterthwaite"},
        anova_kwargs={},
    ):
        """Calculate a Type-III ANOVA table for the model using ``joint_tests()`` in R.

        Args:
            summary (bool): whether to return the ANOVA summary. Defaults to False
            auto_ss_3 (bool): whether to automatically use balanced contrasts when calculating the result via `joint_tests()`. When False, will use the contrasts specified with `set_contrasts()` which defaults to `"contr.treatment"` and R's `anova()` function; Default is True.
            jointtest_kwargs (dict): additional arguments to pass to `joint_tests()` Defaults to using Satterthwaite degrees of freedom
            anova_kwargs (dict): additional arguments to pass to `anova()`
        """
        super().anova(
            auto_ss_3=auto_ss_3,
            jointtest_kwargs=jointtest_kwargs,
            anova_kwargs=anova_kwargs,
        )
        if summary:
            return self.summary_anova()

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
