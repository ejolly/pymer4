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
from warnings import warn


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

    def __init__(
        self, formula, data, family="gaussian", summarize_prior_with="mean", **kwargs
    ):

        # TODO: Add docstring info about families and link functions
        # Initialize attributes
        self.fitted = False  # whether model has been fit
        self.model_obj = None  # bambi model object
        self.inference_obj = None  # arviz inference object after fitting
        self.backend = None  # inference backend used
        self.coef_prior = None  # prior summary statistics
        self.coef_posterior = None  # posterior summary statistics
        self.coef = None  # alias for self.coef_posterior
        self.fixef = None  # cluster level parameters
        self.ranef = None  # cluster level differences from population parameters
        self.ranef_var = None  # cluster level variance
        self.prior_summary_statistic = summarize_prior_with  # how we summarize priors
        self.posterior_summary_statistic = "mean"  # how we summarize posteriors

        # NOTE: We need to convert binomial to bernoulli for bambi
        family = "bernoulli" if family == "binomial" else family

        # Copy data to avoid modifying original
        self.data = copy(data)

        # Initialize bambi model object and extract attributes
        self.model_obj = self._build_bambi_model(self.data, formula, family, **kwargs)
        self.family = self.model_obj.family.name
        self.formula = self.model_obj.formula.main

        # Get bambi's internal design matrix object
        fm_design_matrix = self.model_obj.response_component.design

        # Store DV, IVs, and random effects
        self.terms = dict(
            common_terms=list(fm_design_matrix.common.terms.keys()),
            group_terms=list(fm_design_matrix.group.terms.keys()),
            response_term=fm_design_matrix.response.name,
        )

        # Fixed-effects design matrix
        self.design_matrix = fm_design_matrix.common.as_dataframe()

        # Random-effects names and group sizes
        self.grps = {
            rfx: len(fm_design_matrix.group.terms[rfx].groups)
            for rfx in self.terms["group_terms"]
        }

        # Sample from priors to get summarized predictive distributions
        # for each model term and summarize
        self._build_priors()

        # NOTE: Only if we want to store design matrices for rfx terms
        # self._build_rfx_design_matrices(fm_design_matrix)

    def __repr__(self):
        out = "{}(fitted = {}, formula = {}, family = {})".format(
            self.__class__.__module__, self.fitted, self.formula, self.family
        )
        return out

    @staticmethod
    def _build_bambi_model(data, formula, family, **kwargs):

        model_obj = bmb.Model(formula, data=data, family=family, **kwargs)
        model_obj.build()
        return model_obj

    @staticmethod
    def _fit_bambi_model(
        model_obj, verbose, draws, tune, inference_method, progressbar, **kwargs
    ):

        # Hide basic compilation messages if lowest verbosity (0)
        with open(os.devnull, "w") as f:
            with redirect_stdout(f) if verbose else nullcontext():
                inference_obj = model_obj.fit(
                    draws=draws,
                    tune=tune,
                    inference_method=inference_method,
                    progressbar=progressbar,
                    **kwargs,
                )
        return inference_obj

    def fit(
        self,
        summary=True,
        draws=1000,
        tune=1000,
        inference_method="nuts_numpyro",
        summarize_posterior_with="mean",
        refit=False,
        progressbar=False,
        verbose=False,
        **kwargs,
    ):
        """
        Perform bayesian estimation via the requestined inference method. Defaults to using NUTS sampler with numpyro backend which is a bit faster than PyMC's `'mcmc'` sampler with early identitcal estimates. The underlying `arviz` inference object can be accessed at `.inference_object`. This method will also summarize posterior distributions following BARG guidelines from [Kruschke (2021, NHB)](https://www.nature.com/articles/s41562-021-01177-7).

        Args:
            summary (bool, optional): print an R style summary and return population coefficients table. Defaults to True.
            draws (int, optional): _description_. Defaults to 1000.
            tune (int, optional): _description_. Defaults to 1000.
            inference_method (str, optional): _description_. Defaults to "nuts_numpyro".
            summarize_posterior_with (str, optional): How to summarize posterior distribution. "mean" uses density-based values while "median" uses quantile-based values. Defaults to "mean".
            progressbar (bool, optional): _description_. Defaults to False.
            verbose (bool, optional): _description_. Defaults to False.


        Returns:
            pd.DataFrame: table of posterior parameter summaries if `summary=True`
        """

        # Only perform sampling if model hasn't been fit or if refit is True
        if (not self.fitted) or refit:

            self.inference_obj = self._fit_bambi_model(
                self.model_obj,
                verbose,
                draws,
                tune,
                inference_method,
                progressbar,
                **kwargs,
            )

            # Set flags
            self.fitted = True
            self.backend = inference_method

            # NOTE: self.summary() can rewrite this attribute by design
            self.posterior_summary_statistic = summarize_posterior_with

        # Calculate posterior and fit summary statistics
        # This can be called multiple times without re-fitting
        return self.summary(summarize_posterior_with, return_summary=summary)

    def _coefs_rename_map(self):
        if self.posterior_summary_statistic == "mean":
            rename_map = {
                "mean": "Estimate",
                "sd": "SD",
                "mcse_mean": "MCSE",
                "r_hat": "Rubin_Gelman",
                "hdi_2.5%": "2.5_hdi",
                "hdi_97.5%": "97.5_hdi",
                "ess_bulk": "ESS_Bulk",
                "ess_tail": "ESS_Tail",
            }
            sort_order = [
                "Estimate",
                "2.5_hdi",
                "97.5_hdi",
                "SD",
                "MCSE",
                "Rubin_Gelman",
                "ESS_Bulk",
                "ESS_Tail",
            ]
        elif self.posterior_summary_statistic == "median":
            rename_map = {
                "median": "Estimate",
                "mad": "MAD",
                "mcse_median": "MCSE",
                "r_hat": "Rubin_Gelman",
                "eti_2.5%": "2.5_eti",
                "eti_97.5%": "97.5_eti",
                "ess_median": "ESS_Median",
                "ess_tail": "ESS_Tail",
            }
            sort_order = [
                "Estimate",
                "2.5_eti",
                "97.5_eti",
                "MAD",
                "MCSE",
                "Rubin_Gelman",
                "ESS_Median",
                "ESS_Tail",
            ]
        return rename_map, sort_order

    def _fits_rename_map(self):

        if self.posterior_summary_statistic == "mean":
            rename_map = {
                "mean": "fits",
                "sd": "fits_SD",
                "hdi_2.5%": "fits_2.5_hdi",
                "hdi_97.5%": "fits_97.5_hdi",
            }
            sort_order = [
                "fits",
                "fits_2.5_hdi",
                "fits_97.5_hdi",
                "fits_SD",
                "Kind",
            ]
        elif self.posterior_summary_statistic == "median":
            rename_map = {
                "median": "fits",
                "mad": "fits_MAD",
                "eti_2.5%": "fits_2.5_eti",
                "eti_97.5%": "fits_97.5_eti",
            }
            sort_order = [
                "fits",
                "fits_2.5_eti",
                "fits_97.5_eti",
                "fits_MAD",
                "Kind",
            ]
        return rename_map, sort_order

    def _priors_rename_map(self):

        if self.prior_summary_statistic == "mean":
            rename_map = {
                "mean": "Estimate",
                "hdi_2.5%": "2.5_hdi",
                "hdi_97.5%": "97.5_hdi",
                "sd": "SD",
            }
            sort_order = ["Estimate", "2.5_hdi", "97.5_hdi", "SD"]
        elif self.prior_summary_statistic == "median":
            rename_map = {
                "median": "Estimate",
                "eti_2.5%": "2.5_eti",
                "eti_97.5%": "97.5_eti",
                "mad": "MAD",
            }
            sort_order = ["Estimate", "2.5_eti", "97.5_eti", "MAD"]
        return rename_map, sort_order

    def _build_posteriors(self):

        rename_map, sort_order = self._coefs_rename_map()

        # Population level parameters
        self.coef_posterior = az.summary(
            self.inference_obj,
            kind="all",
            var_names=["~|", "~_sigma"],
            filter_vars="like",
            hdi_prob=0.95,
            stat_focus=self.posterior_summary_statistic,
        ).rename(columns=rename_map)[sort_order]

        # Cluster RFX
        # NOTE: These are equivalent to calleing ranef() in R, not fixef(), i.e. they are cluster level *deviances* from population parameters. Add them to the coefs table to get parameter estimates per cluster
        rfx = dict()
        # Instead of a single data-frame we store a dictionary of data-frames
        # because we have summary statistics over distributions for each rfx term
        for term in self.terms["group_terms"]:
            summary = az.summary(
                self.inference_obj,
                kind="all",
                var_names=term,
                filter_vars="like",
                hdi_prob=0.95,
                stat_focus=self.posterior_summary_statistic,
            ).rename(columns=rename_map)[sort_order]
            # Filter out row with variance across rfx, as the az.summary includes it
            to_remove = summary.filter(like="_sigma", axis=0).index
            summary = summary[~summary.index.isin(to_remove)]
            rfx[term] = summary

        # Store them
        self.ranef = rfx

        # Cluster RFX Variance
        self.ranef_var = az.summary(
            self.inference_obj,
            kind="all",
            var_names=["_sigma"],
            filter_vars="like",
            hdi_prob=0.95,
            stat_focus=self.posterior_summary_statistic,
        ).rename(columns=rename_map)[sort_order]

        # Cluster FFX
        # Wrap in try because this is not always possible to determine automatically
        # with more complicated rfx terms
        try:
            # Get unique cluster names
            cluster_names = list(
                set([term.split("|")[-1] for term in self.terms["group_terms"]])
            )
            fixef = dict()
            for name in cluster_names:
                fixef[name] = self._calc_fixef(name)
            self.fixef = fixef
        except Exception:
            warn(
                f"Hmm, couldn't automatically summarize cluster-level fixed-effects...\nDon't worry this has nothing to do with model fitting, just automatic summarization!\nSince the `model.fixef` attribute won't be available, you should manually combine `model.coef` and `model.ranef` if you want `model.fixef`. You can try to use the internal function `model._calc_fixef(group_var)` method and set `group_var` to the name of cluster variable you want fixed-effects for (e.g. 'subject' or 'item'). However, this may fail again and you need to resort to manual labour :(",
            )

        # Alias attributes for backwards compatibility
        self.coef = self.coef_posterior
        self.coefs = self.coef
        self.fixefs = self.fixef
        self.ranefs = self.ranef

    def _empty_prev_fit_summary(self):

        self.coef = None
        self.fixef = None
        self.ranef = None
        self.ranef_var = None
        self.fits = None
        self.residuals = None
        self.posterior_predictions = None
        _, sort_order = self._fits_rename_map()
        sort_order += ["residuals"]
        self.data = self.data.drop(columns=sort_order, errors="ignore")
        self.posterior_summary_statistic = None

    def _calc_p_values(self):

        out = dict()
        for term in self.coef.index:

            # Factor term
            if "[" in term:
                factor_term = term.split("[")[0]
                posterior = self.inference_obj.posterior[factor_term]
                for i in range(posterior.shape[2]):
                    p = 1 - (posterior[:, :, i] > 0).mean().item()
                    out[f"{term}"] = p

            # Get posterior distribution
            posterior = self.inference_obj.posterior[term]

            if len(posterior.shape) == 3:
                for i in range(posterior.shape[2]):
                    p = 1 - (posterior[:, :, i] > 0).mean().item()
                    out[f"{term}_{i}"] = p
                pass
            else:
                p = 1 - (posterior > 0).mean().item()
                out[term] = p
        # for term in self.terms["common_terms"]:

        #     # Get posterior distribution
        #     posterior = self.inference_obj.posterior[term]
        #     if len(posterior.shape) == 3:
        #         for i in range(posterior.shape[2]):
        #             p = 1 - (posterior[:, :, i] > 0).mean().item()
        #             out[f"{term}_{i}"] = p
        #         pass
        #     else:
        #         p = 1 - (posterior > 0).mean().item()
        #         out[term] = p

    def _calc_fixef(self, group_var):
        """
        Calculate parameter estimates per `group_var` cluster by combining population and rfx estimates. This is equivalent to fixef() in R. This is a method rather than a model attribute because Bambi doesn't store the combined estimates by default and with more complicated rfx terms it's not always clear how to combine them.

        Args:
            group_var (str): name of the group variable to calculate fixef for; Corresponds to the last part of the rfx in the model formula, e.g. for "(condition|subject)" use "subject" to return per-subject estimates.

        Returns:
            pd.DataFrame: data frame with one row per group and columns for the estimate, 2.5 and 97.5 uncertainty estimates

        """

        group_terms = [t for t in self.terms["group_terms"] if group_var in t]
        if len(group_terms) == 0:
            raise ValueError(f"Group variable {group_var} not found in model")

        lb_col = "2.5_hdi" if self.posterior_summary_statistic == "mean" else "2.5_eti"
        ub_col = (
            "97.5_hdi" if self.posterior_summary_statistic == "mean" else "97.5_eti"
        )

        fixef = dict()
        new_index = None
        for term in self.terms["common_terms"]:
            if term == "Intercept":
                rfx_term = [t for t in group_terms if t.startswith("1|")]
                if len(rfx_term) == 0:
                    continue
                rfx_term = rfx_term[0]
                col_prefix = "Intercept"
            else:
                rfx_term = [t for t in group_terms if t.startswith(term + "|")]
                if len(rfx_term) == 0:
                    continue
                rfx_term = rfx_term[0]
                col_prefix = term

            if new_index is None:
                new_index = [e.split("|")[-1] for e in self.ranef[rfx_term].index]

            point_estimates = (
                self.ranef[rfx_term].loc[:, "Estimate"]
                + self.coef_posterior.loc[term, "Estimate"]
            ).to_numpy()
            lb = (
                self.ranef[rfx_term].loc[:, lb_col]
                + self.coef_posterior.loc[term, "Estimate"]
            ).to_numpy()
            ub = (
                self.ranef[rfx_term].loc[:, ub_col]
                + self.coef_posterior.loc[term, "Estimate"]
            ).to_numpy()
            fixef[f"{col_prefix}_Estimate"] = point_estimates
            fixef[f"{col_prefix}_{lb_col}"] = lb
            fixef[f"{col_prefix}_{ub_col}"] = ub

        return pd.DataFrame(fixef, index=new_index)

    def _build_fits(self):
        """Adds .fits (also accessible as .posterior_predictions) and .residuals
        to model object as well as dataframe. Fits are the posterior predictive distribution of the model. Residuals are the difference between the observed data and the fits.
        """

        # Fits/predictions sampled from posterior
        # With no arguments, this will return the posterior predictive distribution
        # using the same data the model was fit to
        fits = self.predict(summarize_predictions_with=self.posterior_summary_statistic)
        fits = fits.rename(
            columns={
                "Estimate": "fits",
                "SD": "fits_SD",
                "2.5_hdi": "fits_2.5_hdi",
                "97.5_hdi": "fits_97.5_hdi",
            },
        ).drop(columns=["Kind"])

        # Store them
        self.data = pd.concat([self.data, fits.reset_index(drop=True)], axis=1)
        self.fits = self.data["fits"].to_numpy()
        self.posterior_predictions = fits

        # Calculate residuals
        self.residuals = self.data[self.terms["response_term"]] - self.fits
        self.data["residuals"] = self.residuals.copy()

    def _build_priors(self):
        """Adds a .prior_coef attribute to the model object that contains summary statistics of the prior predictive distribution of the model parameters. This is useful for understanding the prior distribution of the model parameters."""

        priors = self.model_obj.prior_predictive()
        rename_map, sort_order = self._priors_rename_map()
        self.coef_prior = az.summary(
            priors,
            kind="stats",
            var_names=self.terms["common_terms"],
            hdi_prob=0.95,
            stat_focus=self.prior_summary_statistic,
        ).rename(columns=rename_map)[sort_order]

    def _build_rfx_design_matrices(self, fm_design_matrix):

        # We need to slice the combined rfx numpy array bambi gives us
        # using stored column slices for each rfx term
        self.design_matrix_rfx = dict()
        rfx_design_matrix = np.array(fm_design_matrix.group)

        for rfx in self.terms["group_terms"]:
            # We can use the stored slice range to get the correct columns
            design_mat = rfx_design_matrix[:, fm_design_matrix.group.slices[rfx]]

            self.design_matrix_rfx[rfx] = pd.DataFrame(
                design_mat, columns=fm_design_matrix.group.terms[rfx].groups
            )

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

    def predict(
        self,
        data=None,
        use_rfx=True,
        hdi_prob=0.95,
        kind="pps",
        summarize_predictions_with="mean",
        **kwargs,
    ):
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
            and ci; Default True
            ci (int; optional): highest density interval (bayesian confidence) interval;
            Default to 95%
            stat_focus (str; optional): the aggregation stat if summarize=True; Default
            'mean'

        Returns:
            pd.DataFrame: data frame of predictions and uncertainties. Also saved to
            model.fits if data is None

        """

        # Only guard against external use
        if not self.fitted:
            raise RuntimeError("Model must be fitted to generate predictions!")

        predictions = self.model_obj.predict(
            idata=self.inference_obj,
            data=data,
            inplace=False,
            include_group_specific=use_rfx,
            kind=kind,
            **kwargs,
        )

        # Predictions are an arviz distribution so aggregate them and filter
        # out the single row containing the sigma of the distribution,
        # since we already have uncertainty per prediction
        output = (
            az.summary(
                predictions,
                kind="stats",
                var_names=[self.terms["response_term"]],
                filter_vars="like",
                hdi_prob=hdi_prob,
                stat_focus=summarize_predictions_with,
            )
            .filter(regex=".*?\[.*?\].*?", axis=0)
            .assign(Kind=kind)
        )

        rename_map, sort_order = self._fits_rename_map()
        output = output.rename(columns=rename_map)[sort_order]

        return output

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
        df = (df.assign(Name=names))[
            [
                "Name",
                "Estimate",
                f"{'2.5_hdi' if self.posterior_summary_statistic == 'mean' else '2.5_eti'}",
                f"{'97.5_hdi' if self.posterior_summary_statistic == 'mean' else '97.5_eti'}",
                "Rubin_Gelman",
            ]
        ]

        return df.round(3)

    def summary(self, summarize_posterior_with="mean", return_summary=True):
        """
        Summarize the posterior and estimates of a fitted model. This adds summary tables for population effects (`self.coef`), random effects (`self.ranef`), and fixed effects (`self.fixef`) to the model object. It also adds posterior predictive fits (`self.fits`) and residuals (`self.residuals`) to the model data. Optionally print a summary of the model fit in the style of R's `summary()` function.

        Args:
            summarize_posterior_with (str, optional): Generate summarize using 'mean' or 'median'. Defaults to "mean".
            return_summary (bool, optional): Print summary and return population paraemeter table (`self.coef`). Defaults to True.

        Returns:
            pd.DataFrame: DataFrame of population parameter estimates rounded to 3 decimal places
        """

        if not self.fitted:
            raise RuntimeError("Model must be fitted to generate summary!")

        if summarize_posterior_with not in ["mean", "median"]:
            raise ValueError(
                f"summarize_posterior_with must be 'mean' or 'median', not {summarize_posterior_with}"
            )

        # Remove previous summary stats if they exist
        self._empty_prev_fit_summary()

        # Update user choice, shared across private methods below
        self.posterior_summary_statistic = summarize_posterior_with

        # Create summary tables for population, fixed, and random parameters
        self._build_posteriors()

        # Add fits (predictions on same data) and residuals to
        # self.data marginalizing over posterior
        self._build_fits()

        # Add posterior p-values
        self._calc_p_values()

        # Print R style summary and return population fixed effects
        if return_summary:
            print(f"Linear mixed model fit by: {self.backend}\n")

            print("Formula: {}\n".format(self.formula))
            print("Family: {}\n".format(self.family))
            print(
                "Number of observations: %s\t Groups: %s\n"
                % (self.data.shape[0], self.grps)
            )
            print(f"Posterior sampling: {self.backend}\n")
            print(f"Posterior summary statistic: {self.posterior_summary_statistic}\n")
            # print("Log-likelihood: %.3f \t AIC: %.3f\n" % (self.logLike, self.AIC))
            print("Random effects:\n")
            print("%s\n" % (self._pprint_ranef_var()))
            if self.coef is None:
                print("No fixed effects estimated\n")
                return
            else:
                print("Fixed effects:\n")
                return self.coef.round(3)

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
