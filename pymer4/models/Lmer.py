"""
Pymer4 Lmer Class
=================

Main class to wrap R's lme4 library
"""

from copy import copy
from rpy2.robjects.packages import importr
import rpy2.robjects as robjects
from rpy2.rinterface_lib import callbacks

from rpy2.robjects import numpy2ri
import rpy2.rinterface as rinterface
import warnings
import traceback
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ..utils import (
    _sig_stars,
    _perm_find,
    _return_t,
    _to_ranks_by_group,
    con2R,
    pandas2R,
)
from pandas.api.types import CategoricalDtype

# Import R libraries we need
base = importr("base")
stats = importr("stats")

numpy2ri.activate()

# Make a reference to the default R console writer from rpy2
consolewrite_warning_backup = callbacks.consolewrite_warnerror
consolewrite_print_backup = callbacks.consolewrite_print


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
        AIC (float): model akaike information criterion
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

    def __init__(self, formula, data, family="gaussian"):

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
                    con_codes = np.array(stats.contr_poly(len(contrasts)))
                else:
                    # Treatment/dummy contrasts
                    con_codes = np.array(stats.contr_treatment(len(contrasts)))

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
                con_codes = con2R(cons)
                out[factor] = con_codes

            else:
                raise TypeError(errormsg)
        self.factors = factor_dict
        self.contrast_codes = out
        return robjects.ListVector(out), df

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
            permute=self._permute,
            conf_int=self._conf_int,
            REML=self._REML,
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

        rstring = """
            function(model){
            df<- anova(model)
            df
            }
        """
        anova = robjects.r(rstring)
        self.anova_results = pd.DataFrame(anova(self.model_obj))
        if self.anova_results.shape[1] == 6:
            self.anova_results.columns = [
                "SS",
                "MS",
                "NumDF",
                "DenomDF",
                "F-stat",
                "P-val",
            ]
            self.anova_results["Sig"] = self.anova_results["P-val"].apply(
                lambda x: _sig_stars(x)
            )
        elif self.anova_results.shape[1] == 4:
            warnings.warn(
                "MODELING FIT WARNING! Check model.warnings!! P-value computation did not occur because lmerTest choked. Possible issue(s): ranefx have too many parameters or too little variance..."
            )
            self.anova_results.columns = ["DF", "SS", "MS", "F-stat"]
        if force_orthogonal:
            print(
                "SS Type III Analysis of Variance Table with Satterthwaite approximated degrees of freedom:\n(NOTE: Model refit with orthogonal polynomial contrasts)"
            )
        else:
            print(
                "SS Type III Analysis of Variance Table with Satterthwaite approximated degrees of freedom:\n(NOTE: Using original model contrasts, orthogonality not guaranteed)"
            )
        return self.anova_results

    def _get_ngrps(self, unsum, base):
        """Get the groups information from the model as a dictionary"""
        # This works for 2 grouping factors
        ns = unsum.rx2("ngrps")
        names = base.names(self.model_obj.slots["flist"])
        self.grps = dict(zip(names, ns))

    def _set_R_stdout(self, verbose):
        """Adjust whether R prints to the console (often as a duplicate) based on the verbose flag of a method call. Reference to rpy2 interface here: https://bit.ly/2MsrufO"""

        if verbose:
            # use the default logging in R
            callbacks.consolewrite_warnerror = consolewrite_warning_backup
        else:
            # Create a list buffer to catch messages and discard them
            buf = []

            def _f(x):
                buf.append(x)

            callbacks.consolewrite_warnerror = _f

    def fit(
        self,
        conf_int="Wald",
        n_boot=500,
        factors=None,
        permute=False,
        ordered=False,
        verbose=False,
        REML=True,
        rank=False,
        rank_group="",
        rank_exclude_cols=[],
        no_warnings=False,
        control="",
        old_optimizer=False,
        **kwargs,
    ):
        """
        Main method for fitting model object. Will modify the model's data attribute to add columns for residuals and fits for convenience. Factors should be specified as a dictionary with values as a list or themselves a dictionary of *human readable* contrasts *not* R-style contrast codes as these will be auto-converted for you. See the factors docstring and examples below. After fitting, the .factors attribute will store a reference to the user-specified dictionary. The .contrast_codes model attributes will store the requested comparisons in converted R format. Note that Lmer estimate naming conventions differs a bit from R: Lmer.coefs = summary(model); Lmer.fixefs = coefs(model); Lmer.ranef = ranef(model)

        Args:
            conf_int (str): which method to compute confidence intervals; 'profile', 'Wald' (default), or 'boot' (parametric bootstrap)
            n_boot (int): number of bootstrap intervals if bootstrapped confidence intervals are requests; default 500
            factors (dict): dictionary with column names specified as keys and values as a list for dummy/treatment/polynomial contrast or a dict with keys as factor leves and values as desired comparisons in human readable format See examples below
            permute (int): if non-zero, computes parameter significance tests by permuting test stastics rather than parametrically. Permutation is done by shuffling observations within clusters to respect random effects structure of data.
            ordered (bool): whether factors should be treated as ordered polynomial contrasts; this will parameterize a model with K-1 orthogonal polynomial regressors beginning with a linear contrast based on the factor order provided; default is False
            summarize/summary (bool): whether to print a model summary after fitting; default is True
            verbose (bool): whether to print when and which model and confidence interval are being fitted
            REML (bool): whether to fit using restricted maximum likelihood estimation instead of maximum likelihood estimation; default True
            rank (bool): covert predictors in model formula to ranks by group prior to estimation. Model object will still contain original data not ranked data; default False
            rank_group (str): column name to group data on prior to rank conversion
            rank_exclude_cols (list/str): columns in model formula to not apply rank conversion to
            no_warnings (bool): turn off auto-printing warnings messages; warnings are always stored in the .warnings attribute; default False
            control (str): string containing options to be passed to (g)lmer control. See https://bit.ly/2OQONTH for options
            old_optimizer (bool): use the old bobyqa optimizer that was the default in lmer4 <= 1.1_20, i.e. prior to 02/04/2019. This is not compatible with the control setting as it's meant to be a quick shorthand (e.g. to reproduce previous model results). However, the same setting can be manually requested using the control option if preferred. (For optimizer change discussions see: https://bit.ly/2MrP9Nq and https://bit.ly/2Vx5jte )

        Returns:
            pd.DataFrame: R/statsmodels style summary

        Examples:
            The following examples demonstrate how to treat variables as categorical factors.

            Dummy-Coding: Treat Col1 as a factor which 3 levels: A, B, C. Use dummy-coding with A as the reference level. Model intercept will be mean of A, and parameters will be B-A, and C-A.

            >>> model.fit(factors = {"Col1": ['A','B','C']})

            Orthogonal Polynomials: Treat Col1 as a factor which 3 levels: A, B, C. Estimate a linear contrast of C > B > A. Model intercept will be grand-mean of all levels, and parameters will be linear contrast, and orthogonal polynomial contrast (auto-computed).

            >>> model.fit(factors = {"Col1": ['A','B','C']}, ordered=True)

            Custom-contrast: Treat Col1 as a factor which 3 levels: A, B, C. Compare A to the mean of B and C. Model intercept will be the grand-mean of all levels, and parameters will be the desired contrast, a well as an automatically determined orthogonal contrast.

            >>> model.fit(factors = {"Col1": {'A': 1, 'B': -.5, 'C': -.5}}))

            Here is an example specifying stricter deviance and paramter values stopping criteria.

            >>> model.fit(control="optCtrl = list(ftol_abs=1e-8, xtol_abs=1e-8)")

            Here is an example specifying a different optimizer in addition to stricter deviance and paramter values stopping criteria.

            >>> model.fit(control="optimizer='Nelder_Mead', optCtrl = list(FtolAbs=1e-8, XtolRel=1e-8)")

            Here is an example using the default optimization in previous versions of lme4 prior to the 2019 update.

            >>> model.fit(old_optimizer=True)

        """

        # Alllow summary or summarize for compatibility
        if "summary" in kwargs and "summarize" in kwargs:
            raise ValueError(
                "You specified both summary and summarize, please prefer summarize"
            )
        summarize = kwargs.pop("summarize", True)
        summarize = kwargs.pop("summary", summarize)
        # Save params for future calls
        self._permute = permute
        self._conf_int = conf_int
        self._REML = REML if self.family == "gaussian" else False
        self._set_R_stdout(verbose)

        if permute is True:
            raise TypeError(
                "permute should 'False' or the number of permutations to perform"
            )

        if old_optimizer:
            if control:
                raise ValueError(
                    "Must specify EITHER control OR old_optimizer not both"
                )
            else:
                control = "optimizer='bobyqa'"
        if factors:
            contrasts, dat = self._make_factors(factors, ordered)
        else:
            contrasts = rinterface.NULL
            dat = self.data

        if rank:
            if not rank_group:
                raise ValueError("rank_group must be provided if rank is True")
            dat = _to_ranks_by_group(
                self.data, rank_group, self.formula, rank_exclude_cols
            )
            if factors and (set(factors.keys()) != set(rank_exclude_cols)):
                w = "Factors and ranks requested, but factors are not excluded from rank conversion. Are you sure you wanted to do this?"
                warnings.warn(w)
                self.warnings.append(w)
        if conf_int == "boot":
            self.sig_type = "bootstrapped"
        else:
            if permute:
                self.sig_type = "permutation" + " (" + str(permute) + ")"
            else:
                self.sig_type = "parametric"

        data = pandas2R(dat)

        if self.family == "gaussian":
            _fam = "gaussian"
            if verbose:
                print(
                    f"Fitting linear model using lmer with {conf_int} confidence intervals...\n"
                )

            lmer = importr("lmerTest")
            lmc = robjects.r(f"lmerControl({control})")
            self.model_obj = lmer.lmer(
                self.formula, data=data, REML=REML, control=lmc, contrasts=contrasts
            )
        else:
            if verbose:
                print(
                    f"Fitting generalized linear model using glmer (family {self.family}) with {conf_int} confidence intervals...\n"
                )
            lmer = importr("lme4")
            if self.family == "inverse_gaussian":
                _fam = "inverse.gaussian"
            elif self.family == "gamma":
                _fam = "Gamma"
            else:
                _fam = self.family
            lmc = robjects.r(f"glmerControl({control})")
            self.model_obj = lmer.glmer(
                self.formula,
                data=data,
                family=_fam,
                control=lmc,
                contrasts=contrasts,
            )

        # Store design matrix and get number of IVs for inference
        design_matrix = stats.model_matrix(self.model_obj)
        # rpy2 > 3.4 returns a numpy array that can be empty but has shape (obs x IVs)
        if isinstance(design_matrix, np.ndarray):
            if design_matrix.shape[1] > 0:
                self.design_matrix = pd.DataFrame(base.data_frame(design_matrix))
                num_IV = self.design_matrix.shape[1]
            else:
                num_IV = 0
        # rpy2 < 3.4 returns an R matrix object with a length
        elif len(design_matrix):
            self.design_matrix = pd.DataFrame(base.data_frame(design_matrix))
            num_IV = self.design_matrix.shape[1]
        else:
            num_IV = 0

        if permute and verbose:
            print("Using {} permutations to determine significance...".format(permute))

        summary = base.summary(self.model_obj)
        unsum = base.unclass(summary)

        # Do scalars first cause they're easier

        # Get group names separately cause rpy2 > 2.9 is weird and doesnt return them above
        try:
            self._get_ngrps(unsum, base)
        except Exception as _:
            print(traceback.format_exc())
            raise Exception(
                "The rpy2, lme4, or lmerTest API appears to have changed again. Please file a bug report at https://github.com/ejolly/pymer4/issues with your R, Python, rpy2, lme4, and lmerTest versions and the OS you're running pymer4 on. Apologies."
            )

        self.AIC = unsum.rx2("AICtab")[0]
        self.logLike = unsum.rx2("logLik")[0]

        # First check for lme4 printed messages (e.g. convergence info is usually here instead of in warnings)
        fit_messages = unsum.rx2("optinfo").rx2("conv").rx2("lme4").rx2("messages")
        # Then check warnings for additional stuff
        fit_warnings = unsum.rx2("optinfo").rx2("warnings")

        try:
            fit_warnings = [fw for fw in fit_warnings]
        except TypeError:
            fit_warnings = []
        try:
            fit_messages = [fm for fm in fit_messages]
        except TypeError:
            fit_messages = []

        fit_messages_warnings = fit_warnings + fit_messages
        if fit_messages_warnings:
            self.warnings.extend(fit_messages_warnings)
            if not no_warnings:
                for warning in self.warnings:
                    if isinstance(warning, list) | isinstance(warning, np.ndarray):
                        for w in warning:
                            print(w + " \n")
                    else:
                        print(warning + " \n")
        else:
            self.warnings = []

        # Coefficients, and inference statistics
        if num_IV != 0:
            if self.family in ["gaussian", "gamma", "inverse_gaussian", "poisson"]:

                rstring = (
                    """
                    function(model){
                    out.coef <- data.frame(unclass(summary(model))$coefficients)
                    out.ci <- data.frame(confint(model,method='"""
                    + conf_int
                    + """',nsim="""
                    + str(n_boot)
                    + """))
                    n <- c(rownames(out.ci))
                    idx <- max(grep('sig',n))
                    out.ci <- out.ci[-seq(1:idx),]
                    out <- cbind(out.coef,out.ci)
                    list(out,rownames(out))
                    }
                """
                )
                estimates_func = robjects.r(rstring)
                out_summary, out_rownames = estimates_func(self.model_obj)
                df = pd.DataFrame(out_summary)
                dfshape = df.shape[1]
                df.index = out_rownames

                # gaussian
                if dfshape == 7:
                    df.columns = [
                        "Estimate",
                        "SE",
                        "DF",
                        "T-stat",
                        "P-val",
                        "2.5_ci",
                        "97.5_ci",
                    ]
                    df = df[
                        ["Estimate", "2.5_ci", "97.5_ci", "SE", "DF", "T-stat", "P-val"]
                    ]

                # gamma, inverse_gaussian
                elif dfshape == 6:
                    if self.family in ["gamma", "inverse_gaussian"]:
                        df.columns = [
                            "Estimate",
                            "SE",
                            "T-stat",
                            "P-val",
                            "2.5_ci",
                            "97.5_ci",
                        ]
                        df = df[
                            ["Estimate", "2.5_ci", "97.5_ci", "SE", "T-stat", "P-val"]
                        ]
                    else:
                        # poisson
                        df.columns = [
                            "Estimate",
                            "SE",
                            "Z-stat",
                            "P-val",
                            "2.5_ci",
                            "97.5_ci",
                        ]
                        df = df[
                            ["Estimate", "2.5_ci", "97.5_ci", "SE", "Z-stat", "P-val"]
                        ]

                # Incase lmerTest chokes it won't return p-values
                elif dfshape == 5 and self.family == "gaussian":
                    if not permute:
                        warnings.warn(
                            "MODELING FIT WARNING! Check model.warnings!! P-value computation did not occur because lmerTest choked. Possible issue(s): ranefx have too many parameters or too little variance..."
                        )
                        df.columns = ["Estimate", "SE", "T-stat", "2.5_ci", "97.5_ci"]
                        df = df[["Estimate", "2.5_ci", "97.5_ci", "SE", "T-stat"]]

            elif self.family == "binomial":

                rstring = (
                    """
                    function(model){
                    out.coef <- data.frame(unclass(summary(model))$coefficients)
                    out.ci <- data.frame(confint(model,method='"""
                    + conf_int
                    + """',nsim="""
                    + str(n_boot)
                    + """))
                    n <- c(rownames(out.ci))
                    idx <- max(grep('sig',n))
                    out.ci <- out.ci[-seq(1:idx),]
                    out <- cbind(out.coef,out.ci)
                    odds <- exp(out.coef[1])
                    colnames(odds) <- "OR"
                    probs <- data.frame(sapply(out.coef[1],plogis))
                    colnames(probs) <- "Prob"
                    odds.ci <- exp(out.ci)
                    colnames(odds.ci) <- c("OR_2.5_ci","OR_97.5_ci")
                    probs.ci <- data.frame(sapply(out.ci,plogis))
                    if(ncol(probs.ci) == 1){
                      probs.ci = t(probs.ci)
                    }
                    colnames(probs.ci) <- c("Prob_2.5_ci","Prob_97.5_ci")
                    out <- cbind(out,odds,odds.ci,probs,probs.ci)
                    list(out,rownames(out))
                    }
                """
                )

                estimates_func = robjects.r(rstring)
                out_summary, out_rownames = estimates_func(self.model_obj)
                df = pd.DataFrame(out_summary)
                df.index = out_rownames
                df.columns = [
                    "Estimate",
                    "SE",
                    "Z-stat",
                    "P-val",
                    "2.5_ci",
                    "97.5_ci",
                    "OR",
                    "OR_2.5_ci",
                    "OR_97.5_ci",
                    "Prob",
                    "Prob_2.5_ci",
                    "Prob_97.5_ci",
                ]
                df = df[
                    [
                        "Estimate",
                        "2.5_ci",
                        "97.5_ci",
                        "SE",
                        "OR",
                        "OR_2.5_ci",
                        "OR_97.5_ci",
                        "Prob",
                        "Prob_2.5_ci",
                        "Prob_97.5_ci",
                        "Z-stat",
                        "P-val",
                    ]
                ]

            if permute:
                perm_dat = dat.copy()
                dv_var = self.formula.split("~")[0].strip()
                grp_vars = list(self.grps.keys())
                perms = []
                for _ in range(permute):
                    perm_dat[dv_var] = perm_dat.groupby(grp_vars)[dv_var].transform(
                        lambda x: x.sample(frac=1)
                    )
                    if self.family == "gaussian":
                        perm_obj = lmer.lmer(self.formula, data=perm_dat, REML=REML)
                    else:
                        perm_obj = lmer.glmer(self.formula, data=perm_dat, family=_fam)
                    perms.append(_return_t(perm_obj))
                perms = np.array(perms)
                pvals = []
                for c in range(df.shape[0]):
                    if self.family in ["gaussian", "gamma", "inverse_gaussian"]:
                        pvals.append(_perm_find(perms[:, c], df["T-stat"][c]))
                    else:
                        pvals.append(_perm_find(perms[:, c], df["Z-stat"][c]))
                df["P-val"] = pvals
                if "DF" in df.columns:
                    df["DF"] = [permute] * df.shape[0]
                    df = df.rename(columns={"DF": "Num_perm", "P-val": "Perm-P-val"})
                else:
                    df["Num_perm"] = [permute] * df.shape[0]
                    df = df.rename(columns={"P-val": "Perm-P-val"})

            if "P-val" in df.columns:
                df = df.assign(Sig=df["P-val"].apply(lambda x: _sig_stars(x)))
            elif "Perm-P-val" in df.columns:
                df = df.assign(Sig=df["Perm-P-val"].apply(lambda x: _sig_stars(x)))

            if (conf_int == "boot") and (permute is None):
                # We're computing parametrically bootstrapped ci's so it doesn't make sense to use approximation for p-values. Instead remove those from the output and make significant inferences based on whether the bootstrapped ci's cross 0.
                df = df.drop(columns=["P-val", "Sig"])
                if "DF" in df.columns:
                    df = df.drop(columns="DF")
                df["Sig"] = df.apply(
                    lambda row: "*" if row["2.5_ci"] * row["97.5_ci"] > 0 else "",
                    axis=1,
                )

            # Because all models except lmm have no DF column make sure Num_perm gets put in the right place
            if permute:
                if self.family != "gaussian":
                    cols = list(df.columns)
                    col_order = cols[:-4] + ["Num_perm"] + cols[-4:-2] + [cols[-1]]
                    df = df[col_order]

            self.coefs = df
            # Make sure the design matrix column names match population coefficients
            self.design_matrix.columns = self.coefs.index[:]
        else:
            self.coefs = None
            if permute or conf_int == "boot":
                print(
                    "**NOTE**: Non-parametric inference only applies to fixed effects and none were estimated\n"
                )

        self.fitted = True

        # Random effect variances and correlations
        varcor_NAs = ["NA", "N", robjects.NA_Character]  # NOQA
        df = pd.DataFrame(base.data_frame(unsum.rx2("varcor")))

        ran_vars = df.query("var2 in @varcor_NAs").drop("var2", axis=1)
        ran_vars.index = ran_vars["grp"]
        ran_vars.drop("grp", axis=1, inplace=True)
        ran_vars.columns = ["Name", "Var", "Std"]
        ran_vars.index.name = None
        ran_vars.replace("NA", "", inplace=True)
        ran_vars = ran_vars.applymap(
            lambda x: np.nan if x == robjects.NA_Character else x
        )
        ran_vars.replace(np.nan, "", inplace=True)

        ran_corrs = df.query("var2 not in @varcor_NAs").drop("vcov", axis=1)
        if ran_corrs.shape[0] != 0:
            ran_corrs.index = ran_corrs["grp"]
            ran_corrs.drop("grp", axis=1, inplace=True)
            ran_corrs.columns = ["IV1", "IV2", "Corr"]
            ran_corrs.index.name = None
            ran_corrs = ran_corrs.applymap(
                lambda x: np.nan if x == robjects.NA_Character else x
            )
            ran_corrs.replace(np.nan, "", inplace=True)
        else:
            ran_corrs = None

        self.ranef_var = ran_vars
        self.ranef_corr = ran_corrs

        # Cluster (e.g subject) level coefficients
        rstring = """
            function(model){
            getIndex <- function(df){
                    orignames <- names(df)
                    df <- transform(df, index=row.names(df))
                    names(df) <- append(orignames, c("index"))
                    df
                    }
            out <- coef(model)
            out <- lapply(out, getIndex)
            out
            }
        """
        fixef_func = robjects.r(rstring)
        fixefs = fixef_func(self.model_obj)
        fixefs = [
            pd.DataFrame(e, index=e.index).drop(columns=["index"]) for e in fixefs
        ]
        if len(fixefs) > 1:
            if self.coefs is not None:
                f_corrected_order = []
                for f in fixefs:
                    f_corrected_order.append(
                        pd.DataFrame(
                            f[
                                list(self.coefs.index)
                                + [
                                    elem
                                    for elem in f.columns
                                    if elem not in self.coefs.index
                                ]
                            ]
                        )
                    )
                self.fixef = f_corrected_order
            else:
                self.fixef = list(fixefs)
        else:
            self.fixef = fixefs[0]
            if self.coefs is not None:
                self.fixef = self.fixef[
                    list(self.coefs.index)
                    + [
                        elem
                        for elem in self.fixef.columns
                        if elem not in self.coefs.index
                    ]
                ]

        # Sort column order to match population coefs
        # This also handles cases in which random slope terms exist in the model without corresponding fixed effects terms, which generates extra columns in this dataframe. By default put those columns *after* the fixed effect columns of interest (i.e. population coefs)

        # Cluster (e.g subject) level random deviations
        rstring = """
            function(model){
            uniquify <- function(df){
            colnames(df) <- make.unique(colnames(df))
            df
            }
            getIndex <- function(df){
            df <- transform(df, index=row.names(df))
            df
            }
            out <- lapply(ranef(model),uniquify)
            out <- lapply(out, getIndex)
            out
            }
        """
        ranef_func = robjects.r(rstring)
        ranefs = ranef_func(self.model_obj)
        if len(ranefs) > 1:
            self.ranef = [
                pd.DataFrame(e, index=e.index).drop(columns=["index"]) for e in ranefs
            ]
        else:
            self.ranef = pd.DataFrame(ranefs[0], index=ranefs[0].index).drop(
                columns=["index"]
            )

        # Model residuals
        rstring = """
            function(model){
            out <- resid(model)
            out
            }
        """
        resid_func = robjects.r(rstring)
        self.residuals = np.array(resid_func(self.model_obj))
        try:
            self.data["residuals"] = copy(self.residuals)
        except ValueError as _:  # NOQA
            print(
                "**NOTE**: Column for 'residuals' not created in model.data, but saved in model.resid only. This is because you have rows with NaNs in your data.\n"
            )

        # Model fits
        rstring = """
            function(model){
            out <- fitted(model)
            out
            }
        """
        fit_func = robjects.r(rstring)
        self.fits = fit_func(self.model_obj)
        try:
            self.data["fits"] = copy(self.fits)
        except ValueError as _:  # NOQA
            print(
                "**NOTE** Column for 'fits' not created in model.data, but saved in model.fits only. This is because you have rows with NaNs in your data.\n"
            )

        if summarize:
            return self.summary()

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

        self._set_R_stdout(verbose)

        if isinstance(num_datasets, float):
            num_datasets = int(num_datasets)
        if not isinstance(num_datasets, int):
            raise ValueError("num_datasets must be an integer")

        if use_rfx:
            re_form = "NULL"
        else:
            re_form = "NA"

        rstring = (
            """
            function(model){
            out <- simulate(model,"""
            + str(num_datasets)
            + """,allow.new.levels=TRUE,re.form="""
            + re_form
            + """)
            out
            }
        """
        )
        simulate_func = robjects.r(rstring)
        sims = simulate_func(self.model_obj)
        return pd.DataFrame(sims)

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
        self._set_R_stdout(verbose)
        if self.design_matrix is None:
            raise ValueError(
                "No fixed effects were estimated so prediction is not possible!"
            )
        if not skip_data_checks:
            required_cols = self.design_matrix.columns[1:]
            if not all([col in data.columns for col in required_cols]):
                raise ValueError(
                    "Column names do not match all fixed effects model terms!\nThis may be a false error if some predictors are categorical, in which case you can bypass this check by setting skip_checks=True."
                )

        if use_rfx:
            if not skip_data_checks:
                required_cols = set(list(required_cols) + list(self.grps.keys()))
                if not all([col in data.columns for col in required_cols]):
                    raise ValueError(
                        "Column names are missing random effects model grouping terms!"
                    )

            re_form = "NULL"
        else:
            re_form = "NA"

        rstring = (
            """
            function(model,new){
            out <- predict(model,new,allow.new.levels=TRUE,re.form="""
            + re_form
            + """,type='"""
            + pred_type
            + """')
            out
            }
        """
        )

        predict_func = robjects.r(rstring)
        preds = predict_func(self.model_obj, pandas2R(data))
        if verify_predictions:
            self._verify_preds(preds, use_rfx)
        return preds

    def _verify_preds(self, preds, use_rfx):
        """
        Verify that the output of .predict given new data is not identitical to the
        model's fits from the data it was trained on. This is necessary because
        `predict()` in R will silently fallback to return the same predicted values when
        given input data with no matching columns, but with `use_rfx=False`

        Args:
            preds (np.array): output of self.predict()
            use_rfx (bool): same input parameter as self.predict

        Raises:
            ValueError: If predictions match model fits from original data
        """
        training_preds = self.predict(
            self.data, use_rfx=use_rfx, skip_data_checks=True, verify_predictions=False
        )
        mess = "(using rfx)" if use_rfx else "(without rfx)"

        if np.allclose(training_preds, preds):
            raise ValueError(
                f"Predictions are identitical to fitted values {mess}!!\nYou can ignore this error if you intended to predict using the same data the model was trained on by setting verify_predictions=False. If you didn't, then its likely that some or all of the column names in your test data don't match the column names from the data the model was trained on and you set skip_data_checks=True."
            )

    def summary(self):
        """
        Summarize the output of a fitted model.

        Returns:
            pd.DataFrame: R/statsmodels style summary

        """

        if not self.fitted:
            raise RuntimeError("Model must be fitted to generate summary!")

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

        Examples:

            Pairwise comparison of means of A at each level of B

            >>> model.post_hoc(marginal_vars='A',grouping_vars='B')

            Pairwise differences of slopes of C between levels of A at each level of B

            >>> model.post_hoc(marginal_vars='C',grouping_vars=['A','B'])

            Pairwise differences of each unique A,B cell

            >>> model.post_hoc(marginal_vars=['A','B'])

        """

        self._set_R_stdout(verbose)

        if not marginal_vars:
            raise ValueError("Must provide marginal_vars")

        if not self.fitted:
            raise RuntimeError("Model must be fitted to generate post-hoc comparisons")

        if not isinstance(marginal_vars, list):
            marginal_vars = [marginal_vars]

        if grouping_vars and not isinstance(grouping_vars, list):
            grouping_vars = [grouping_vars]
            # Conditional vars can only be factor types
            if not all([elem in self.factors.keys() for elem in grouping_vars]):
                raise ValueError(
                    "All grouping_vars must be existing categorical variables (i.e. factors)"
                )

        # Need to figure out if marginal_vars is continuous or not to determine lstrends or emmeans call
        cont, factor = [], []
        for var in marginal_vars:
            if not self.factors or var not in self.factors.keys():
                cont.append(var)
            else:
                factor.append(var)

        if cont:
            if factor:
                raise ValueError(
                    "With more than one marginal variable, all variables must be categorical factors. Mixing continuous and categorical variables is not supported. Try passing additional categorical factors to grouping_vars"
                    ""
                )
            else:
                if len(cont) > 1:
                    raise ValueError(
                        "Marginal variables can only contain one continuous variable"
                    )
                elif len(cont) == 1:
                    if grouping_vars:
                        # Emtrends; there's a bug for trends where options don't get set by default so an empty list is passed to R, see: https://bit.ly/2VJ9QZM
                        cont = cont[0]
                        if len(grouping_vars) > 1:
                            g1 = grouping_vars[0]
                            _conditional = "+".join(grouping_vars[1:])

                            rstring = (
                                """
                                function(model){
                                suppressMessages(library(emmeans))
                                out <- emtrends(model,pairwise ~ """
                                + g1
                                + """|"""
                                + _conditional
                                + """,var='"""
                                + cont
                                + """',adjust='"""
                                + p_adjust
                                + """',options=list(),lmer.df='satterthwaite',lmerTest.limit=9999)
                                out
                                }"""
                            )
                        else:
                            rstring = (
                                """
                                function(model){
                                suppressMessages(library(emmeans))
                                out <- emtrends(model,pairwise ~ """
                                + grouping_vars[0]
                                + """,var='"""
                                + cont
                                + """',adjust='"""
                                + p_adjust
                                + """',options=list(),lmer.df='satterthwaite',lmerTest.limit=9999)
                                out
                                }"""
                            )

                    else:
                        raise ValueError(
                            "grouping_vars are required with a continuous marginal_vars"
                        )
        else:
            if factor:
                _marginal = "+".join(factor)
                if grouping_vars:
                    # emmeans with pipe
                    _conditional = "+".join(grouping_vars)
                    rstring = (
                        """
                        function(model){
                        suppressMessages(library(emmeans))
                        out <- emmeans(model,pairwise ~ """
                        + _marginal
                        + """|"""
                        + _conditional
                        + """, adjust='"""
                        + p_adjust
                        + """',lmer.df='satterthwaite',lmerTest.limit=9999)
                        out
                        }"""
                    )
                else:
                    # emmeans without pipe
                    rstring = (
                        """
                        function(model){
                        suppressMessages(library(emmeans))
                        out <- emmeans(model,pairwise ~ """
                        + _marginal
                        + """,adjust='"""
                        + p_adjust
                        + """',lmer.df='satterthwaite',lmerTest.limit=9999)
                        out
                        }"""
                    )
            else:
                raise ValueError("marginal_vars are not in model!")

        func = robjects.r(rstring)
        res = func(self.model_obj)
        emmeans = importr("emmeans")

        # Marginal estimates
        self.marginal_estimates = pd.DataFrame(base.summary(res)[0])
        # Resort columns
        effect_names = list(self.marginal_estimates.columns[:-4])
        # this column name changes depending on whether we're doing post-hoc trends or means
        effname = effect_names[-1]
        sortme = effect_names[:-1] + ["Estimate", "2.5_ci", "97.5_ci", "SE", "DF"]

        # In emmeans (compared to lsmeans) the CI column names change too depending on how many factor variabls are in the model
        if "asymp.LCL" in self.marginal_estimates.columns:
            self.marginal_estimates = self.marginal_estimates.rename(
                columns={
                    effname: "Estimate",
                    "df": "DF",
                    "asymp.LCL": "2.5_ci",
                    "asymp.UCL": "97.5_ci",
                }
            )[sortme]
        elif "lower.CL" in self.marginal_estimates.columns:
            self.marginal_estimates = self.marginal_estimates.rename(
                columns={
                    effname: "Estimate",
                    "df": "DF",
                    "lower.CL": "2.5_ci",
                    "upper.CL": "97.5_ci",
                }
            )[sortme]
        else:
            raise ValueError(
                f"Cannot figure out what emmeans is naming marginal CI columns. Expected 'lower.CL' or 'asymp.LCL', but columns are {self.marginal_estimates.columns}"
            )

        # Marginal Contrasts
        self.marginal_contrasts = pd.DataFrame(base.summary(res)[1])
        if "t.ratio" in self.marginal_contrasts.columns:
            rename_dict = {
                "t.ratio": "T-stat",
                "p.value": "P-val",
                "estimate": "Estimate",
                "df": "DF",
                "contrast": "Contrast",
            }
            sorted_names = [
                "Estimate",
                "2.5_ci",
                "97.5_ci",
                "SE",
                "DF",
                "T-stat",
                "P-val",
            ]
        elif "z.ratio" in self.marginal_contrasts.columns:
            rename_dict = {
                "z.ratio": "Z-stat",
                "p.value": "P-val",
                "estimate": "Estimate",
                "df": "DF",
                "contrast": "Contrast",
            }
            sorted_names = [
                "Estimate",
                "2.5_ci",
                "97.5_ci",
                "SE",
                "DF",
                "Z-stat",
                "P-val",
            ]
        else:
            raise ValueError(
                f"Cannot figure out what emmeans is naming contrast means columns. Expected 't.ratio' or 'z.ratio', but columns are: {self.marginal_contrasts.columns}"
            )

        self.marginal_contrasts = self.marginal_contrasts.rename(columns=rename_dict)

        # Need to make another call to emmeans to get confidence intervals on contrasts
        confs = pd.DataFrame(base.unclass(emmeans.confint_emmGrid(res))[1])
        confs = confs.iloc[:, -2:]
        # Deal with changing column names again
        if "asymp.LCL" in confs.columns:
            confs = confs.rename(
                columns={"asymp.LCL": "2.5_ci", "asymp.UCL": "97.5_ci"}
            )
        elif "lower.CL" in confs.columns:
            confs = confs.rename(columns={"lower.CL": "2.5_ci", "upper.CL": "97.5_ci"})
        else:
            raise ValueError(
                f"Cannot figure out what emmeans is naming contrast CI columns. Expected 'lower.CL' or 'asymp.LCL', but columns are {self.marginal_estimates.columns}"
            )

        self.marginal_contrasts = pd.concat([self.marginal_contrasts, confs], axis=1)
        # Resort columns
        effect_names = list(self.marginal_contrasts.columns[:-7])
        sortme = effect_names + sorted_names
        self.marginal_contrasts = self.marginal_contrasts[sortme]
        self.marginal_contrasts["Sig"] = self.marginal_contrasts["P-val"].apply(
            _sig_stars
        )

        if (
            p_adjust == "tukey"
            and self.marginal_contrasts.shape[0] >= self.marginal_estimates.shape[0]
        ):
            print(
                "P-values adjusted by tukey method for family of {} estimates".format(
                    self.marginal_contrasts["Contrast"].nunique()
                )
            )
        elif p_adjust != "tukey":
            print(
                "P-values adjusted by {} method for {} comparisons".format(
                    p_adjust, self.marginal_contrasts["Contrast"].nunique()
                )
            )
        if summarize:
            return self.marginal_estimates.round(3), self.marginal_contrasts.round(3)

    def plot_summary(
        self,
        figsize=(12, 6),
        error_bars="ci",
        ranef=True,
        axlim=None,
        plot_intercept=True,
        ranef_alpha=0.5,
        coef_fmt="o",
        orient="v",
        ranef_idx=0,
    ):
        """
        Create a forestplot overlaying estimated coefficients with random effects (i.e. BLUPs). By default display the 95% confidence intervals computed during fitting.

        Args:
            error_bars (str): one of 'ci' or 'se' to change which error bars are plotted; default 'ci'
            ranef (bool): overlay BLUP estimates on figure; default True
            axlim (tuple): lower and upper limit of plot; default min and max of BLUPs
            plot_intercept (bool): plot the intercept estimate; default True
            ranef_alpha (float): opacity of random effect points; default .5
            coef_fmt (str): matplotlib marker style for population coefficients
            ranef_idx (int): if multiple random effects clusters were specified this value indicates which one should be plotted; uses 0-based indexing; default 0 (first)

        Returns:
            plt.axis: matplotlib axis handle
        """

        if not self.fitted:
            raise RuntimeError("Model must be fit before plotting!")
        if orient not in ["h", "v"]:
            raise ValueError("orientation must be 'h' or 'v'")

        if isinstance(self.fixef, list):
            m_ranef = self.fixef[ranef_idx]
        else:
            m_ranef = self.fixef
        m_fixef = self.coefs

        if not plot_intercept:
            m_ranef = m_ranef.drop("(Intercept)", axis=1)
            m_fixef = m_fixef.drop("(Intercept)", axis=0)

        if error_bars == "ci":
            col_lb = (m_fixef["Estimate"] - m_fixef["2.5_ci"]).values
            col_ub = (m_fixef["97.5_ci"] - m_fixef["Estimate"]).values
        elif error_bars == "se":
            col_lb, col_ub = m_fixef["SE"], m_fixef["SE"]

        # For seaborn
        m = pd.melt(m_ranef)

        _, ax = plt.subplots(1, 1, figsize=figsize)

        if ranef:
            alpha_plot = ranef_alpha
        else:
            alpha_plot = 0

        if orient == "v":
            x_strip = "value"
            x_err = m_fixef["Estimate"]
            y_strip = "variable"
            y_err = range(m_fixef.shape[0])
            xerr = [col_lb, col_ub]
            yerr = None
            ax.vlines(
                x=0, ymin=-1, ymax=self.coefs.shape[0], linestyles="--", color="grey"
            )
            if not axlim:
                xlim = (m["value"].min() - 1, m["value"].max() + 1)
            else:
                xlim = axlim
            ylim = None
        else:
            y_strip = "value"
            y_err = m_fixef["Estimate"]
            x_strip = "variable"
            x_err = range(m_fixef.shape[0])
            yerr = [col_lb, col_ub]
            xerr = None
            ax.hlines(
                y=0, xmin=-1, xmax=self.coefs.shape[0], linestyles="--", color="grey"
            )
            if not axlim:
                ylim = (m["value"].min() - 1, m["value"].max() + 1)
            else:
                ylim = axlim
            xlim = None

        sns.stripplot(
            x=x_strip, y=y_strip, data=m, ax=ax, size=6, alpha=alpha_plot, color="grey"
        )

        ax.errorbar(
            x=x_err,
            y=y_err,
            xerr=xerr,
            yerr=yerr,
            fmt=coef_fmt,
            capsize=0,
            elinewidth=4,
            color="black",
            ms=12,
            zorder=9999999999,
        )

        ax.set(ylabel="", xlabel="Estimate", xlim=xlim, ylim=ylim)
        sns.despine(top=True, right=True, left=True)
        return ax

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
        if self.factors:
            raise NotImplementedError(
                "Plotting can currently only handle models with continuous predictors!"
            )
        if isinstance(self.fixef, list) or isinstance(self.ranef, list):
            raise NotImplementedError(
                "Plotting can currently only handle models with 1 random effect grouping variable!"
            )
        if self.design_matrix is None:
            raise ValueError(
                "No fixed effects were estimated so prediction is not possible!"
            )
        if not ax:
            _, ax = plt.subplots(1, 1, figsize=figsize)

        # Get range of unique values for desired parameter
        x_vals = self.design_matrix[param].unique()
        # Sort order to handle bug in matplotlib plotting
        idx = np.argsort(x_vals)

        # Get desired parameter part of the prediction
        fixef_pred = (
            self.coefs.loc["(Intercept)", "Estimate"]
            + self.coefs.loc[param, "Estimate"] * x_vals
        )
        fixef_pred_upper = (
            self.coefs.loc["(Intercept)", "97.5_ci"]
            + self.coefs.loc[param, "97.5_ci"] * x_vals
        )
        fixef_pred_lower = (
            self.coefs.loc["(Intercept)", "2.5_ci"]
            + self.coefs.loc[param, "2.5_ci"] * x_vals
        )

        if grps:
            if all(isinstance(x, int) for x in grps):
                ran_dat = self.fixef.iloc[grps, :]
            elif all(isinstance(x, str) for x in grps):
                ran_dat = self.fixef.loc[grps, :]
            else:
                raise TypeError(
                    "grps must be integer list for integer-indexing (.iloc) of fixed effects, or label list for label-indexing (.loc) of fixed effects"
                )
        else:
            ran_dat = self.fixef

        # Now generate random effects predictions
        for _, row in ran_dat.iterrows():

            ranef_desired = row["(Intercept)"] + row[param] * x_vals
            # ranef_other = np.dot(other_vals_means, row.loc[other_vals])
            pred = ranef_desired  # + ranef_other

            ax.plot(x_vals[idx], pred[idx], "-", linewidth=2)

        if plot_fixef:
            ax.plot(
                x_vals[idx],
                fixef_pred[idx],
                "--",
                color="black",
                linewidth=3,
                zorder=9999999,
            )

        if plot_ci:
            ax.fill_between(
                x_vals[idx],
                fixef_pred_lower[idx],
                fixef_pred_upper[idx],
                facecolor="black",
                alpha=0.25,
                zorder=9999998,
            )

        ax.set(
            ylim=(self.data.fits.min(), self.data.fits.max()),
            xlim=(x_vals.min(), x_vals.max()),
            xlabel=param,
            ylabel=self.formula.split("~")[0].strip(),
        )
        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)
        return ax
