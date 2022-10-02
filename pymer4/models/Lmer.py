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

        # Initialize bambi model object and extract attributes
        self.model_obj = bmb.Model(
            self.formula, data=self.data, family=self.family, **kwargs
        )
        self.model_obj.build()

        # Store separate model terms for ease of reference
        if self.model_obj.intercept_term:
            common_terms = ["Intercept"]
        else:
            common_terms = []
        common_terms += list(self.model_obj.common_terms.keys())
        group_terms = list(self.model_obj.group_specific_terms.keys())
        self.terms = dict(common_terms=common_terms, group_terms=group_terms)

        # Fixed effects population params dm
        self.design_matrix = self.model_obj._design.common.as_dataframe()
        self._get_ngrps()

        # Rfx matrix: num obs x num rfx terms (e.g. intercepts, slopes) per group
        # E.g. with random intercepts and 10 subs this would have 10 columns
        # E.g. with random intercepts+slopes and 10 subs this would have 20 columns
        # E.g. with random intercepts+slopes for 10 subs and random intercepts for 5 items this would have 25 columns
        rfx_name_slices = self.model_obj._design.group.slices
        rfx_mat = pd.DataFrame(np.array(self.model_obj._design.group))
        col_names = np.array(rfx_mat.columns).astype(str)
        for rfx_name, slice_range in rfx_name_slices.items():
            col_names[slice_range] = rfx_name
        rfx_mat.columns = col_names
        self.design_matrix_rfx = rfx_mat

    def __repr__(self):
        out = "{}(fitted = {}, formula = {}, family = {})".format(
            self.__class__.__module__, self.fitted, self.formula, self.family
        )
        return out

    def _make_factors(self, factor_dict, ordered=False):
        """
        Covert specific columns to R-style factors. Default scheme is dummy coding where reference is 1st level provided. Alternative is orthogonal polynomial contrasts. User can also specific custom contrasts.

        Args:
            factor_dict: (dict) dictionary with column names specified as keys and values as a list for dummy/treatment/polynomial contrast; a dict with keys as factor leves and values as desired comparisons in human readable format
            ordered: (bool) whether to interpret factor_dict values as dummy-coded (1st list item is reference level) or as polynomial contrasts (linear contrast specified by ordered of list items); ignored if factor_dict values are not a list

        Returns:
            pandas.core.frame.DataFrame: copy of original data with factorized columns

        Examples:

            Dummy/treatment contrasts with 'A' as the reference level and other contrasts as 'B'-'A' and 'C'-'A'

            >>> _make_factors(factor_dict={'factor': ['A','B','C']})

            Same as above but a linear contrast (and automatically computed quadratic contrast) of A < B < C

            >>> _make_factors(factor_dict={'factor': ['A','B','C']}, ordered=True)

            Custom contrast of 'A' - mean('B', 'C')

            >>> _make_factors(factor_dict={'factor': {'A': 1, 'B': -0.5, 'C': -0.5}})
        """

        errormsg = "factors should be specified as a dictionary with values as one of:\n1) a list with factor levels in the desired order for dummy/treatment/polynomial contrasts\n2) a dict with keys as factor levels and values as desired comparisons in human readable format"
        # We create a copy of data because we need to convert dtypes to categories and then pass them to R. However, resetting categories on the *same* dataframe and passing to R repeatedly (e.g. multiple calls to .fit with different contrasats) does not work as R only uses the 1st category spec. So instead we create a copy and return that copy to get used by .fit
        out = {}
        df = self.data.copy()
        if not isinstance(factor_dict, dict):
            raise TypeError(errormsg)
        for factor, contrasts in factor_dict.items():
            # First convert to a string type because R needs string based categories
            df[factor] = df[factor].apply(str)

            # Treatment/poly contrasts
            if isinstance(contrasts, list):
                # Ensure that all factor levels are accounted for
                if not all([e in contrasts for e in df[factor].unique()]):
                    raise ValueError(
                        "Not all factor levels are specified in the desired contrast"
                    )
                # Define and apply a pandas categorical type in the same order as requested, which will get converted to the right factor levels in R
                cat = CategoricalDtype(contrasts)
                df[factor] = df[factor].astype(cat)

                if ordered:
                    # Polynomial contrasts
                    con_codes = None
                else:
                    # Treatment/dummy contrasts
                    con_codes = None

                out[factor] = con_codes

            # Custom contrasts (human readable)
            elif isinstance(contrasts, dict):
                factor_levels = list(contrasts.keys())
                cons = list(contrasts.values())
                # Ensure that all factor levels are accounted for
                if not all([e in factor_levels for e in df[factor].unique()]):
                    raise ValueError(
                        "Not all factor levels are specified in the desired contrast"
                    )
                # Define and apply categorical type in the same order as requested
                cat = CategoricalDtype(factor_levels)
                df[factor] = df[factor].astype(cat)
                # Compute desired contrasts in R format along with addition k - 1 contrasts not specified
                out[factor] = cons

            else:
                raise TypeError(errormsg)
        self.factors = factor_dict
        self.contrast_codes = out
        return out, df

    def _refit_orthogonal(self):
        """
        Refit a model with factors organized as polynomial contrasts to ensure valid type-3 SS calculations with using `.anova()`. Previous factor specifications are stored in `model.factors_prev_`.
        """

        self.factors_prev_ = copy(self.factors)
        self.contrast_codes_prev_ = copy(self.contrast_codes)
        # Create orthogonal polynomial contrasts for all factors, by creating a list of unique
        # factor levels as self._make_factors will handle the rest
        new_factors = {}
        for factor in self.factors.keys():
            new_factors[factor] = sorted(list(map(str, self.data[factor].unique())))

        self.fit(
            factors=new_factors,
            ordered=True,
            summarize=False,
        )

    def anova(self, force_orthogonal=False):
        """
        Return a type-3 ANOVA table from a fitted model. Like R, this method does not ensure that contrasts are orthogonal to ensure correct type-3 SS computation. However, the force_orthogonal flag can refit the regression model with orthogonal polynomial contrasts automatically guaranteeing valid SS type 3 inferences. Note that this will overwrite factors specified in the last call to `.fit()`

        Args:
            force_orthogonal (bool): whether factors in the model should be recoded using polynomial contrasts to ensure valid type-3 SS calculations. If set to True, previous factor specifications will be saved in `model.factors_prev_`; default False

        Returns:
            pd.DataFrame: Type 3 ANOVA results
        """

        if self.factors:
            # Model can only have factors if it's been fit
            if force_orthogonal:
                self._refit_orthogonal()
        elif not self.fitted:
            raise ValueError("Model must be fit before ANOVA table can be generated!")

        self.anova_results = None
        return self.anova_results

    def _get_ngrps(self):
        """Get the groups information from the model as a dictionary"""

        group_terms = self.model_obj.group_specific_terms.values()
        self.grps = {e.name.split("|")[-1]: len(e.groups) for e in group_terms}

    def fit(
        self,
        rank=False,
        ordered=False,
        rank_group="",
        rank_exclude_cols=[],
        verbose=0,
        **kwargs,
    ):

        inference_method = kwargs.pop("inference_method", "nuts_numpyro")
        draws = kwargs.pop("draws", self.draws)
        tune = kwargs.pop("tune", self.tune)
        summary = kwargs.pop("summary", True)
        summarize = kwargs.pop("summary", True)
        if isinstance(verbose, bool):
            verbose = 2 if verbose else 0

        # Only print progress bars if maximum verbosity (2)
        # Different backends have different kwarg names so we need to build this dict programmatically
        additional_kwargs = dict()
        if inference_method == "nuts_numpyro":
            additional_kwargs["progress_bar"] = True if verbose > 1 else False
            self.backend = "Numpyro/Jax NUTS Sampler"
        else:
            additional_kwargs["progressbar"] = True if verbose > 1 else False
            self.backend = "PyMC MCMC"

        # Hide basic compilation messages if lowest verbosity (0)
        with open(os.devnull, "w") as f:
            with redirect_stdout(f) if verbose == 0 else nullcontext():
                self.inference_obj = self.model_obj.fit(
                    draws=draws,
                    inference_method=inference_method,
                    tune=tune,
                    **additional_kwargs,
                )

        # Set flag now for other internal ops like .predict call
        self.fitted = True

        # Population level effects
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

        # Cluster level deviances from population effect
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
        # Filter out variances manually as the az.summary still includes them
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

        # This adds a column to self.inference_obj.posterior named 'DV_mean'
        self.model_obj.predict(
            self.inference_obj, inplace=True, include_group_specific=True, kind="mean"
        )
        # This adds a new attribute on self.inference_obj called
        # .posterior_predictive that contains a column called 'DV'
        self.model_obj.predict(
            self.inference_obj, inplace=True, include_group_specific=True, kind="pps"
        )
        # Aggregate them by calling predict using the same data the model was estimated
        # with. By default this uses the posterior estimates of the mean response var,
        # instead of the posterior predictive dist. But aggregating them gives the same
        # estimates when calculated on the same data the model was fit on
        self.fits = self.predict(data=None, summarize=True)

        self.data["fits"] = self.fits["Estimate"].copy()

        if summary or summarize:
            self.summary()

        # TODO: add warnings for RG stat > 1.05

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

    def plot_summary(self, kind="trace", params="default", ci=95, **kwargs):

        if not self.fitted:
            raise RuntimeError("Model must be fitted to plot summary!")

        hdi_prob = kwargs.pop("ci", 95) / 100
        kwargs.update({"hdi_prob": hdi_prob})

        if kind == "priors":
            return self._plot_priors(**kwargs)

        if kind in ["ppc", "yhat", "preds", "predictions", "fits"]:
            _ = kwargs.pop("hdi_prob", None)
            return az.plot_ppc(self.inference_obj, **kwargs)

        var_names = _select_az_params(params)

        if kind == "trace":
            if "combined" not in kwargs:
                kwargs.update({"combined": False})
            _ = kwargs.pop("hdi_prob", None)
            plot_func = az.plot_trace
        elif kind in ["forest", "summary", "ridge"]:
            if "combined" not in kwargs:
                kwargs.update({"combined": True})
            plot_func = az.plot_forest
            if kind == "ridge":
                kwargs.update({"kind": "ridgeplot"})
        elif kind in ["posterior", "posteriors"]:
            plot_func = az.plot_posterior
        else:
            raise ValueError(f"${kind} plot not supported")

        return plot_func(
            self.inference_obj,
            var_names=var_names,
            filter_vars="like",
            **kwargs,
        )

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
        pass

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
            if n in self.model_obj.group_specific_terms.keys():
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
        pass
