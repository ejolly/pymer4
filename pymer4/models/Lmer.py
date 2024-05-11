"""
Pymer4 Lmer Class
=================

Main class to wrap Bambi's model interface
"""

from copy import copy
import numpy as np
import pandas as pd
import bambi as bmb
import arviz as az
from pandas.api.types import CategoricalDtype
from contextlib import redirect_stdout, nullcontext
import os
from pymer4.utils import _select_az_params


class Lmer(object):
    """
    Model class to hold data outputted from fitting lmer in R and converting to Python object. This class stores as much information as it can about a merMod object computed using lmer and lmerTest in R. Most attributes will not be computed until the fit method is called.

    Args:
        formula (str): Complete lmer-style model formula
        data (pandas.core.frame.DataFrame): input data
        family (string): what distribution family (i.e.) link function to use for the generalized model; default is gaussian (linear model)

    Attributes:
        fitted (bool): whether model has been fit
        formula (str): model formula
        data (pd.DataFrame): model copy of input data
        grps (dict): groups and number of observations per groups recognized by lmer
        design_matrix (pd.DataFrame): model design matrix determined by lmer
        AIC (float): model Akaike information criterion
        BIC (float): model Bayesian information criterion
        logLike (float): model Log-likelihood
        family (string): model family
        warnings (list): warnings output from R or Python
        ranef (pd.DataFrame/list): cluster-level differences from population parameters, i.e. difference between coefs and fixefs; returns list if multiple cluster variables are used to specify random effects (e.g. subjects and items)
        fixef (pd.DataFrame/list): cluster-level parameters; returns list if multiple cluster variables are used to specify random effects (e.g. subjects and items)
        coefs (pandas.core.frame.DataFrame/list): model summary table of population parameters
        ranef_var (pd.DataFrame): random effects variances
        ranef_corr(pd.DataFrame): random effects correlations
        residuals (numpy.ndarray): model residuals
        inference_obj (arviz.InfereceData): xarray inference data
        model_obj (bambi model): bambi model object
        fits (pd.DataFrame): predictions with uncertainty ("y-hats")

    """

    def __init__(self, formula, data, family="gaussian", **kwargs):

        self.family = family
        implemented_fams = ["gaussian"]
        # implemented_fams = [
        #     "gaussian",
        #     "binomial",
        #     "gamma",
        #     "inverse_gaussian",
        #     "poisson",
        # ]
        if self.family not in implemented_fams:
            raise ValueError(f"Family must be one of: {implemented_fams}!")
        self.fitted = False
        self.formula = formula.replace(" ", "")
        self.data = copy(data)
        self.grps = None
        self.AIC = None
        self.BIC = None
        self.logLike = None
        self.warnings = []
        self.ranef_var = None
        self.ranef_corr = None
        self.ranef = None
        self.fixef = None
        self.design_matrix = None
        self.design_matrix_rfx = None
        self.residuals = None
        self.coefs = None
        self.model_obj = None
        self.factors = None
        self.contrast_codes = None
        self.ranked_data = False
        self.marginal_estimates = None
        self.marginal_contrasts = None
        self.sig_type = None
        self.factors_prev_ = None
        self.contrasts = None
        self.backend = None
        self.draws = 2000
        self.tune = 1000
        self.terms = {}
        self.fits = None
        self.inference_obj = None
        self.prior_predictions = None
        self.posterior_predictions = None

        # Initialize bambi model object and extract attributes
        self.model_obj = bmb.Model(
            self.formula, data=self.data, family=self.family, **kwargs
        )
        self.model_obj.build()

        # Get bambi's internal design matrix object
        fm_design_matrix = self.model_obj.response_component.design

        # Store DV, IVs, and random effects
        self.terms = dict(
            common_terms=list(fm_design_matrix.common.terms.keys()),
            group_terms=list(fm_design_matrix.group.terms.keys()),
            response_term=fm_design_matrix.response.name,
        )

        # Fixed-effect design matrix
        self.design_matrix = fm_design_matrix.common.as_dataframe()

        # Random-effects names, design matrices, and group sizes
        self.grps = dict()
        self.design_matrix_rfx = dict()

        # Get rfx group sizes and optionally design matrix
        # We need to slice the combined rfx numpy array bambi gives us
        # using stored column slices for each rfx term
        # rfx_design_matrix = np.array(fm_design_matrix.group)

        for rfx in self.terms["group_terms"]:

            # Get the number of groups for this rfx term
            grp_size = len(fm_design_matrix.group.terms[rfx].groups)

            # Store dicts
            self.grps[rfx] = grp_size
            # We can use the stored slice range to get the correct columns

            # NOTE: If we want to store the rfx design matrices per term
            # design_mat = rfx_design_matrix[:, fm_design_matrix.group.slices[rfx]]
            # self.design_matrix_rfx[rfx] = pd.DataFrame(
            #     design_mat, columns=fm_design_matrix.group.terms[rfx].groups
            # )

    def __repr__(self):
        out = "{}(fitted = {}, formula = {}, family = {})".format(
            self.__class__.__module__, self.fitted, self.formula, self.family
        )
        return out

    def anova(self, force_orthogonal=False):
        """
        Return a type-3 ANOVA table from a fitted model. Like R, this method does not ensure that contrasts are orthogonal to ensure correct type-3 SS computation. However, the force_orthogonal flag can refit the regression model with orthogonal polynomial contrasts automatically guaranteeing valid SS type 3 inferences. Note that this will overwrite factors specified in the last call to `.fit()`

        Args:
            force_orthogonal (bool): whether factors in the model should be recoded using polynomial contrasts to ensure valid type-3 SS calculations. If set to True, previous factor specifications will be saved in `model.factors_prev_`; default False

        Returns:
            pd.DataFrame: Type 3 ANOVA results
        """

        raise NotImplementedError("This method is not yet implemented")

    def fit(
        self,
        summary=True,
        draws=1000,
        tune=1000,
        inference_method="nuts_numpyro",
        progressbar=False,
        verbose=False,
        rank=False,
        ordered=False,
        rank_group="",
        rank_exclude_cols=[],
        **kwargs,
    ):

        # Hide basic compilation messages if lowest verbosity (0)
        with open(os.devnull, "w") as f:
            with redirect_stdout(f) if verbose else nullcontext():
                self.inference_obj = self.model_obj.fit(
                    draws=draws,
                    tune=tune,
                    inference_method=inference_method,
                    progressbar=progressbar,
                    **kwargs,
                )

        # Set flag now for other internal ops like .predict call
        self.fitted = True

        # Create summary tables for fixed and random effects
        self._build_results()

        # Return summary table if requested
        if summary:
            return self.summary()

    def _build_results(self):

        # Population level parameters
        self.coefs = az.summary(
            self.inference_obj,
            kind="all",
            var_names=["~|", "~_sigma"],
            filter_vars="like",
            hdi_prob=0.95,
        ).rename(
            columns={
                "mean": "Estimate",
                "sd": "SD",
                "mcse_mean": "SE",
                "r_hat": "Rubin_Gelman",
                "hdi_2.5%": "2.5_ci",
                "hdi_97.5%": "97.5_ci",
            }
        )[
            ["Estimate", "SD", "2.5_ci", "97.5_ci", "SE", "Rubin_Gelman"]
        ]

        # NOTE: These are equivalent to calleing ranef() in R, not fixef(), i.e. they are cluster level *deviances* from population parameters. Add them to the coefs table to get parameter estimates per cluster
        # Cluster level effects
        self.ranef = az.summary(
            self.inference_obj,
            kind="all",
            var_names=["|"],
            filter_vars="like",
            hdi_prob=0.95,
        ).rename(
            columns={
                "mean": "Estimate",
                "sd": "SD",
                "mcse_mean": "SE",
                "r_hat": "Rubin_Gelman",
                "hdi_2.5%": "2.5_ci",
                "hdi_97.5%": "97.5_ci",
            }
        )[
            ["Estimate", "SD", "2.5_ci", "97.5_ci", "SE", "Rubin_Gelman"]
        ]
        # Filter out row with variance across rfx, as the az.summary includes it
        to_remove = self.ranef.filter(like="_sigma", axis=0).index
        self.ranef = self.ranef[~self.ranef.index.isin(to_remove)]

        # Variance of ranfx
        self.ranef_var = az.summary(
            self.inference_obj,
            kind="all",
            var_names=["_sigma"],
            filter_vars="like",
            hdi_prob=0.95,
        ).rename(
            columns={
                "mean": "Estimate",
                "sd": "SD",
                "mcse_mean": "SE",
                "r_hat": "Rubin_Gelman",
                "hdi_2.5%": "2.5_ci",
                "hdi_97.5%": "97.5_ci",
            }
        )[
            ["Estimate", "SD", "2.5_ci", "97.5_ci", "SE", "Rubin_Gelman"]
        ]

        # TODO:
        # Create summary table for cluster level parameters

        # TODO: Fix me
        # Fits/predictions marginalizing over posterior
        # This adds a column to self.inference_obj.posterior named 'DV_mean'
        # self.model_obj.predict(
        #     self.inference_obj, inplace=True, include_group_specific=True, kind="mean"
        # )

        # This adds a new attribute on self.inference_obj called
        # .posterior_predictions that contains a column called 'DV'
        # self.model_obj.predict(
        #     self.inference_obj, inplace=True, include_group_specific=True, kind="pps"
        # )

        # Aggregate them by calling predict using the same data the model was estimated
        # with. By default this uses the posterior estimates of the mean response var,
        # jnstead of the posterior predictive dist. But aggregating them gives the same
        # estimates when calculated on the same data the model was fit on
        # if verbose:
        #     print("Sampling predictions marginalizing over posteriors...")
        # self.fits = self.predict(data=None, summarize=True, kind="ppc").drop(
        #     columns=["Kind"]
        # )
        # self.posterior_predictions = self.fits

        # self.data["fits"] = self.fits["Estimate"].copy()

        # Fits/predictions marginalizing over prior
        # NOTE: Move to init?
        # if verbose:
        #     print("Sampling predictions marginalizing over priors...")
        # priors = self.model_obj.prior_predictive(draws=self.draws)

        # TODO: Fixme
        # Storing everything in a single inference object is conveninent but plots are
        # getting screwed up. E.g. plot with kind='trace' is incuding posterior
        # predictive traces which take a realllly long time to plot
        # Solution 1: Try using separate inference objects
        # Solution 2: tweak plotting code
        # self.inference_obj.add_groups(
        #     {"prior": priors.prior, "prior_predictive": priors.prior_predictive}
        # )
        # self.prior_predictions = (
        #     az.summary(
        #         self.inference_obj.prior_predictive,
        #         kind="stats",
        #         var_names=[self.model_obj.response.name],
        #         filter_vars="like",
        #         hdi_prob=0.95,
        #         stat_focus="mean",
        #     ).rename(
        #         columns={
        #             "mean": "Estimate",
        #             "sd": "SD",
        #             "hdi_2.5%": "2.5_ci",
        #             "hdi_97.5%": "97.5_ci",
        #         }
        #     )
        # )[["Estimate", "SD", "2.5_ci", "97.5_ci"]]

    def _plot_priors(self, **kwargs):
        """Helper function for .plot_summary when requesting prior plots because this
        calls a custom bambi method rather than an arviz function"""

        hdi_prob = kwargs.pop("hdi_prob", 0.95)
        hdi_prob = kwargs.pop("ci", 95) / 100
        params = kwargs.pop("params", "default")
        if params == "default":
            var_names = None
        elif params in ["coef", "coefs", "fixefs", "fixef"]:
            var_names = self.terms["common_terms"]
        elif params in ["ranefs", "rfx", "ranef"]:
            var_names = self.terms["group_terms"]
        return self.model_obj.plot_priors(
            hdi_prob=hdi_prob, var_names=var_names, **kwargs
        )

    def diagnostics(self, params="default", **kwargs):

        if not self.fitted:
            raise RuntimeError("Model must be fitted to plot summary!")

        var_names = _select_az_params(params)
        return az.summary(
            self.inference_obj,
            kind="diagnostics",
            var_names=var_names,
            filter_vars="like",
            **kwargs,
        )

    def _get_terms_for_plotting(self, params):
        """Helper function to aid in variable selection with az.plot_* funcs"""

        if params in ["default"]:
            var_names = self.terms["common_terms"] + [
                f"{e}_sigma" for e in self.terms["group_terms"]
            ]
            filter_vars = None

        elif params in ["coef", "coefs"]:
            var_names = self.terms["common_terms"]
            filter_vars = None

        # TODO: split up into separate elifs after adding fixed computation
        elif params in ["fixef", "fixefs", "ranef", "ranefs", "rfx"]:
            var_names = self.terms["group_terms"]
            filter_vars = "like"

        elif params in ["response"]:
            var_names = self.terms["response_term"]
            filter_vars = "like"

        elif params in [None, "all"]:
            var_names = None
            filter_vars = None
        else:
            raise ValueError(f"params = {params} not understood")

        return var_names, filter_vars

    def plot_summary(
        self, kind="trace", dist="posterior", params="default", ci=95, **kwargs
    ):

        if not self.fitted:
            raise RuntimeError("Model must be fitted to plot summary!")

        hdi_prob = kwargs.pop("ci", 95) / 100
        kwargs.update({"hdi_prob": hdi_prob})

        # Trace plots for inspecting sampler
        if kind == "trace":
            if dist != "posterior":
                raise ValueError(
                    f"{kind} plots are only supported with dist='posterior'"
                )
            if "combined" not in kwargs:
                kwargs.update({"combined": False})
            _ = kwargs.pop("hdi_prob", None)
            var_names, filter_vars = self._get_terms_for_plotting(params)
            plot_func = az.plot_trace

        # Summary plots for model terms and HDIs/CIs
        elif kind in ["summary", "forest", "ridge"]:
            if dist != "posterior":
                raise ValueError(
                    f"{kind} plots are only supported with dist='posterior'"
                )
            if "combined" not in kwargs:
                kwargs.update({"combined": True})
            var_names, filter_vars = self._get_terms_for_plotting(params)
            if kind == "ridge":
                kwargs.update({"kind": "ridgeplot"})
            plot_func = az.plot_forest

        # Posterior distribution plots
        elif kind in ["posterior_dist", "posterior", "posteriors"]:
            var_names, filter_vars = self._get_terms_for_plotting(params)
            plot_func = az.plot_posterior

        # Prior distribution plots
        # Different plotting call cause it's through bambi
        elif kind in ["prior_dist", "prior", "priors"]:

            # By default plot all priors
            params = None if params == "default" else params
            var_names, _ = self._get_terms_for_plotting(params)
            # If requesting only rfx we need exact names so append _sigma to name
            if params in ["fixef", "fixefs", "ranef", "ranefs", "rfx"]:
                var_names = [f"{name}_sigma" for name in var_names]
            _ = kwargs.pop("dist", None)
            kwargs.update({"var_names": var_names})
            return self.model_obj.plot_priors(**kwargs)

        # Y-hat/prediction plots
        elif kind in ["ppc", "yhat", "preds", "predictions", "fits"]:
            _ = kwargs.pop("hdi_prob", None)
            _ = kwargs.pop("dist", None)
            return az.plot_ppc(self.inference_obj, group=dist, **kwargs)
        else:
            raise ValueError(f"${kind} plot not supported")

        return plot_func(
            self.inference_obj,
            var_names=var_names,
            filter_vars=filter_vars,
            **kwargs,
        )

    def plot_priors(self, **kwargs):
        return self.plot_summary(kind="priors", dist="priors", **kwargs)

    def plot_posteriors(self, **kwargs):
        return self.plot_summary(kind="posteriors", dist="posteriors", **kwargs)

    def simulate(self, num_datasets, use_rfx=True, verbose=False):
        """
        Simulate new responses based upon estimates from a fitted model. By default group/cluster means for simulated data will match those of the original data. Unlike predict, this is a non-deterministic operation because lmer will sample random-efects values for all groups/cluster and then sample data points from their respective conditional distributions.

        Args:
            num_datasets (int): number of simulated datasets to generate. Each simulation always generates a dataset that matches the size of the original data
            use_rfx (bool): wehther to match group/cluster means in simulated data
            verbose (bool): whether to print R messages to console

        Returns:
            np.ndarray: simulated data values
        """
        raise NotImplementedError("This method is not yet implemented")

    def predict(self, data=None, **kwargs):
        """
        Make predictions given new data or return predictions on data model was fit to.
        Predictions include uncertainty and are summarized using highest density
        intervals (bayesian confidence intervals) controlled via the 'ci' kwarg.
        data must be a dataframe that contains the same columns as the model.data
        (i.e. all the predictor variables used to fit the model). If using random
        effects to make predictions, input data must also contain a column for the group
        identifier that were used to fit the model random effects terms. Using random
        effects to make predictions only makes sense if predictions are being made about the same groups/clusters.

        Args:
            use_rfx (bool; optional): whether to condition on random effects when making
            predictions; Default True
            kind (str; optional) the type of prediction required. Can be ``"mean"`` or
            ``"pps"``. The first returns draws from the posterior distribution of the
            mean, while the latter returns the draws from the posterior predictive
            distribution (i.e. the posterior probability distribution for a new
            observation).Defaults to ``"mean"``.
            data (pandas.core.frame.DataFrame; optional): input data to make predictions
            on. Defaults to using same data model was fit with
            summarize (bool; optional): whether to aggregate predictions by stat_focus
            and ci; Default True
            ci (int; optional): highest density interval (bayesian confidence) interval;
            Default to 95%
            stat_focus (str; optional): the aggregation stat if summarize=True; Default
            'mean'

        Returns:
            pd.DataFrame: data frame of predictions and uncertainties. Also saved to
            model.fits if data is None

        """

        use_rfx = kwargs.pop("use_rfx", True)
        summarize = kwargs.pop("summarize", True)
        hdi_prob = kwargs.pop("ci", 95) / 100
        stat_focus = kwargs.pop("stat_focus", "mean")

        # Only guard against external use
        if not self.fitted:
            raise RuntimeError("Model must be fitted to generate predictions!")

        # If no data is passed, just return the predictions from the estimated
        # posteriors and observed data. Otherwise call bambi's predict method with the
        # new data as a kwarg
        if kwargs.get("data", None) is None:
            predictions = self.inference_obj
        else:
            predictions = self.model_obj.predict(
                self.inference_obj,
                inplace=False,
                include_group_specific=use_rfx,
                **kwargs,
            )
        # So we aggregate them using arviz and filter out the single row containing the
        # sigma of the distribution of predictions, since we already have uncertainty
        # per prediction
        if summarize:
            summary = (
                az.summary(
                    predictions,
                    kind="stats",
                    var_names=[self.model_obj.response.name],
                    filter_vars="like",
                    hdi_prob=hdi_prob,
                    stat_focus=stat_focus,
                )
                .filter(regex=".*?\[.*?\].*?", axis=0)
                .assign(Kind=kwargs.get("kind", "mean"))
                .rename(
                    columns={
                        "mean": "Estimate",
                        "sd": "SD",
                        "hdi_2.5%": "2.5_ci",
                        "hdi_97.5%": "97.5_ci",
                    }
                )
            )[["Estimate", "SD", "2.5_ci", "97.5_ci", "Kind"]]

        else:
            summary = None

        if summarize:
            return summary
        else:
            return predictions

    def _pprint_ranef_var(self):
        """
        Format model rfx variances to look like lme4. Used by .summary()
        """

        df = self.ranef_var.copy()
        # Format index
        new_index = []
        names = []
        for name in df.index:
            n = name.split("_sigma")[0]
            if n in self.terms["group_terms"]:
                term_name, group_name = n.split("|")
                term_name = "(Intercept)" if term_name == "1" else term_name
                new_index.append(group_name)
                names.append(term_name)
            else:
                new_index.append("Residual")
                names.append(n)
        df.index = new_index

        # Format columns
        df = (
            df.assign(Name=names)
            .drop(columns=["SD", "SE"])
            .rename(columns={"Estimate": "Std"})
        )[["Name", "Std", "2.5_ci", "97.5_ci", "Rubin_Gelman"]]

        return df.round(3)

    def summary(self):
        """
        Summarize the output of a fitted model.

        Returns:
            pd.DataFrame: R/statsmodels style summary

        """

        if not self.fitted:
            raise RuntimeError("Model must be fitted to generate summary!")

        print(f"Linear mixed model fit by: {self.backend}\n")

        print("Formula: {}\n".format(self.formula))
        print("Family: {}\n".format(self.family))
        print(
            "Number of observations: %s\t Groups: %s\n"
            % (self.data.shape[0], self.grps)
        )
        # print("Log-likelihood: %.3f \t AIC: %.3f\n" % (self.logLike, self.AIC))
        print("Random effects:\n")
        print("%s\n" % (self._pprint_ranef_var()))
        if self.coefs is None:
            print("No fixed effects estimated\n")
            return
        else:
            print("Fixed effects:\n")
            return self.coefs.round(3)

    def plot(
        self,
        param,
        figsize=(8, 6),
        xlabel="",
        ylabel="",
        plot_fixef=True,
        plot_ci=True,
        grps=[],
        ax=None,
    ):
        """
        Plot random and group level parameters from a fitted model

        Args:
            param (str): model parameter (column name) to plot
            figsize (tup): matplotlib desired figsize
            xlabel (str): x-axis label
            ylabel (str): y-axis label
            plot_fixef (bool): plot population effect fit of param?; default True
            plot_ci (bool): plot computed ci's of population effect?; default True
            grps (list): plot specific group fits only; must correspond to index values in model.fixef
            ax (matplotlib.axes.Axes): axis handle for an existing plot; if provided will ensure that random parameter plots appear *behind* all other plot objects.

        Returns:
            plt.axis: matplotlib axis handle

        """

        if not self.fitted:
            raise RuntimeError("Model must be fit before plotting!")

        raise NotImplementedError("This method is not yet implemented")
