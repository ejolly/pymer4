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
        fits (numpy.ndarray): model fits/predictions
        model_obj (lmer model): rpy2 lmer model object
        factors (dict): factors used to fit the model if any

    """

    def __init__(self, formula, data, family="gaussian", **bambi_kwargs):

        self.family = family
        implemented_fams = [
            "gaussian",
            "binomial",
            "gamma",
            "inverse_gaussian",
            "poisson",
        ]
        if self.family not in implemented_fams:
            raise ValueError(
                "Family must be one of: gaussian, binomial, gamma, inverse_gaussian or poisson!"
            )
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

        # Initialize bambi model object and extract attributes
        self.model_obj = bmb.Model(
            self.formula, data=self.data, family=self.family, **bambi_kwargs
        )
        # Fixed effects population params
        self.design_matrix = self.model_obj._design.common.as_dataframe()

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
        pass
        self.grps = None

    def fit(
        self,
        rank=False,
        ordered=False,
        rank_group="",
        rank_exclude_cols=[],
        **bambi_kwargs,
    ):

        inference_method = bambi_kwargs.pop("inference_method", "nuts_numpyro")
        draws = bambi_kwargs.pop("draws", 2000)
        tune = bambi_kwargs.pop("tune", 1000)
        self.fits = self.model_obj.fit(
            draws=draws,
            inference_method=inference_method,
            tune=tune,
            progress_bar=False,
        )
        self.coefs = az.summary(
            self.fits,
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
        self.fixefs = az.summary(
            self.fits, kind="all", var_names=["|"], filter_vars="like", hdi_prob=0.95
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
        # Filter out fixefs variances manually as the az.summary still includes them
        to_remove = self.fixefs.filter(like="_sigma", axis=0).index
        self.fixefs = self.fixefs[~self.fixefs.index.isin(to_remove)]

        # Variance of ranfx
        self.ranef_vars = az.summary(
            self.fits,
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
        self.fitted = True
        print(self.ranef_vars)
        # TODO: add warnings for RG stat > 1.05
        return self.coefs

    def plot_priors(self, *args, **kwargs):
        return self.model_obj.plot_priors(*args, **kwargs)

    def diagnostics(self, params="coefs", **kwargs):

        if not self.fitted:
            raise RuntimeError("Model must be fitted to plot summary!")

        if params in ["coefs", "fixefs"]:
            var_names = ["~|", "~_sigma"]
        elif params in ["ranefs", "rfx"]:
            var_names = ["|"]
        elif params in ["ranef_vars", "rfx_vars"]:
            var_names = ["_sigma"]
        elif params in ["rfx-all", "ranef-all", "ranefs-all"]:
            var_names = ["|", "_sigma"]
        else:
            var_names = None
        return az.summary(
            self.fits,
            kind="diagnostics",
            var_names=var_names,
            filter_vars="like",
            **kwargs,
        )

    def plot_summary(self, kind="trace", params="coefs", **kwargs):

        if not self.fitted:
            raise RuntimeError("Model must be fitted to plot summary!")

        if kind == "priors":
            return self.plot_priors(**kwargs)

        if kind in ["ppc", "yhat"]:
            return az.plot_ppc(self.fits, **kwargs)

        if params in ["coefs", "fixefs"]:
            var_names = ["~|", "~_sigma"]
        elif params in ["ranefs", "rfx"]:
            var_names = ["|"]
        elif params in ["ranef_vars", "rfx_vars"]:
            var_names = ["_sigma"]
        elif params in ["rfx-all", "ranef-all", "ranefs-all"]:
            var_names = ["|", "_sigma"]
        else:
            var_names = None

        if kind == "trace":
            plot_func = az.plot_trace
        elif kind in ["forest", "summary"]:
            plot_func = az.plot_forest
        elif kind in ["posterior", "posteriors"]:
            plot_func = az.plot_posterior
        else:
            raise ValueError(f"${kind} plot not supported")

        return plot_func(
            self.fits,
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

    def predict(
        self,
        data,
        use_rfx=True,
        pred_type="response",
        skip_data_checks=False,
        verify_predictions=True,
        verbose=False,
    ):
        """
        Make predictions given new data. Input must be a dataframe that contains the
        same columns as the model.matrix excluding the intercept (i.e. all the predictor
        variables used to fit the model). If using random effects to make predictions,
        input data must also contain a column for the group identifier that were used to
        fit the model random effects terms. Using random effects to make predictions
        only makes sense if predictions are being made about the same groups/clusters.
        If any predictors are categorical, you can skip verifying column names by
        setting `skip_data_checks=True`.

        Args:
            data (pandas.core.frame.DataFrame): input data to make predictions on
            use_rfx (bool): whether to condition on random effects when making
            predictions; Default True
            pred_type (str): whether the prediction should be on the 'response' scale
            (default); or on the 'link' scale of the predictors passed through the link
            function (e.g. log-odds scale in a logit model instead of probability
            values)
            skip_data_checks (bool): whether to skip checks that input data have the
            same columns as the original data the model were trained on. If predicting
            using a model trained with categorical variables it can be helpful to set
            this to False. Default True
            verify_predictions (bool): whether to ensure that the predicted data are not
            identical to original model fits. Only useful to set this to False when
            making predictions on the same data the model was fit on, but its faster to
            access these directly from model.fits or model.data['fits']. Default True
            verbose (bool): whether to print R messages to console

        Returns:
            np.ndarray: prediction values

        """
        pass

    def summary(self):
        """
        Summarize the output of a fitted model.

        Returns:
            pd.DataFrame: R/statsmodels style summary

        """

        if not self.fitted:
            raise RuntimeError("Model must be fitted to generate summary!")

        # TODO: Print backend info here
        print("Linear mixed model fit by maximum likelihood  ['lmerMod']")

        print("Formula: {}\n".format(self.formula))
        print("Family: {}\t Inference: {}\n".format(self.family, self.sig_type))
        print(
            "Number of observations: %s\t Groups: %s\n"
            % (self.data.shape[0], self.grps)
        )
        print("Log-likelihood: %.3f \t AIC: %.3f\n" % (self.logLike, self.AIC))
        print("Random effects:\n")
        print("%s\n" % (self.ranef_var.round(3)))
        if self.ranef_corr is not None:
            print("%s\n" % (self.ranef_corr.round(3)))
        else:
            print("No random effect correlations specified\n")
        if self.coefs is None:
            print("No fixed effects estimated\n")
            return
        else:
            print("Fixed effects:\n")
            return self.coefs.round(3)

    # TODO Provide option to to pass lmerTest.limit = N in order to get non Inf dof when number of observations > 3000. Apparently this is a new default in emmeans. This warning is only visible when verbose=True
    def post_hoc(
        self,
        marginal_vars,
        grouping_vars=None,
        p_adjust="tukey",
        summarize=True,
        verbose=False,
    ):
        """
        Post-hoc pair-wise tests corrected for multiple comparisons (Tukey method) implemented using the emmeans package. This method provide both marginal means/trends along with marginal pairwise differences. More info can be found at: https://cran.r-project.org/web/packages/emmeans/emmeans.pdf

        Args:
            marginal_var (str/list): what variable(s) to compute marginal means/trends for; unique combinations of factor levels of these variable(s) will determine family-wise error correction
            grouping_vars (str/list): what variable(s) to group on. Trends/means/comparisons of other variable(s), will be computed at each level of these variable(s)
            p_adjust (str): multiple comparisons adjustment method. One of: tukey, bonf, fdr, hochberg, hommel, holm, dunnet, mvt (monte-carlo multi-variate T, aka exact tukey/dunnet). Default tukey
            summarize (bool): output effects and contrasts or don't (always stored in model object as model.marginal_estimates and model.marginal_contrasts); default True
            verbose (bool): whether to print R messages to the console

        Returns:
            Multiple:
                - **marginal_estimates** (*pd.Dataframe*): unique factor level effects (e.g. means/coefs)

                - **marginal_contrasts** (*pd.DataFrame*): contrasts between factor levels

        """
        pass

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
