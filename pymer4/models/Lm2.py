"""
Pymer4 Lm2 Class
================

Main class for two-stage regression models
"""

from copy import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from patsy import dmatrices
from joblib import Parallel, delayed
from .Lm import Lm
from ..utils import _sig_stars, _permute_sign, _ols_group, _corr_group, _perm_find


class Lm2(object):

    """
    Model class to perform two-stage OLS regression. Practically, this class fits a separate Lm() model to each cluster/group in the data and performs inference on the coefficients of each model (i.e. 1-sample t-test per coefficient). The results from this second level regression are reported. This is an alternative to using Lmer, as it implicitly allows intercept and slopes to vary by group, however with no prior/smoothing/regularization on the random effects. See https://bit.ly/2SwHhQU and Gelman (2005). This approach maybe less preferable to Lmer if the number of observations per group are few, but the number of groups is large, in which case the 1st-level estimates are much noisier and are not smoothed/regularized as in Lmer. It maybe preferable when a "maximal" rfx Lmer model is not estimable. Formula specification works just like in R based on columns of a dataframe. Formulae are parsed by patsy which makes it easy to utilize specific columns as factors. This is **different** from Lmer. See patsy for more information on the different use cases.

    Args:
        formula (str): Complete lm-style model formula
        data (pd.DataFrame): input data
        family (string): what distribution family (i.e.) link function to use for the generalized model; default is gaussian (linear model)
        group (list/string): the grouping variable to use to run the 1st-level regression; if a list is provided will run multiple levels feeding the coefficients from the previous level into the subsequent level

    Attributes:
        fitted (bool): whether model has been fit
        formula (str): model formula
        data (pandas.core.frame.DataFrame): model copy of input data
        grps (dict): groups and number of observations per groups recognized by lmer
        AIC (float): model akaike information criterion
        logLike (float): model Log-likelihood
        family (string): model family
        warnings (list): warnings output from Python
        fixef (pd.DataFrame): cluster-level parameters
        coefs (pd.DataFrame): model summary table of population parameters
        residuals (numpy.ndarray): model residuals
        fits (numpy.ndarray): model fits/predictions
        se_type (string): how standard errors are computed
        sig_type (string): how inference is performed


    """

    def __init__(self, formula, data, group, family="gaussian"):

        self.family = family
        # implemented_fams = ['gaussian','binomial']
        if self.family != "gaussian":
            raise NotImplementedError(
                "Currently only linear (family ='gaussian') models supported! "
            )
        if isinstance(group, str):
            self.group = group
        else:
            raise TypeError("group must be a string or list")
        self.fitted = False
        self.formula = formula.replace(" ", "")
        self.data = copy(data)
        self.AIC = None
        self.logLike = None
        self.warnings = []
        self.residuals = None
        self.fixef = None
        self.coefs = None
        self.model_obj = None
        self.ci_type = None
        self.se_type = None
        self.sig_type = None
        self.ranked_data = False
        self.iscorrs = False

    def __repr__(self):
        out = "{}(fitted={}, formula={}, family={}, group={})".format(
            self.__class__.__module__,
            self.fitted,
            self.formula,
            self.family,
            self.group,
        )
        return out

    def fit(
        self,
        robust=False,
        conf_int="standard",
        permute=False,
        perm_on="t-stat",
        rank=False,
        verbose=False,
        n_boot=500,
        n_jobs=1,
        n_lags=1,
        to_corrs=False,
        ztrans_corrs=True,
        cluster=None,
        **kwargs,
    ):
        """
        Fit a variety of second-level OLS models; all 1st-level models are standard OLS. By default will fit a model that makes parametric assumptions (under a t-distribution) replicating the output of software like R. 95% confidence intervals (CIs) are also estimated parametrically by default. However, empirical bootstrapping can also be used to compute CIs, which will resample with replacement from the first level regression estimates and uses these CIs to perform inference unless permutation tests are requested. Permutation testing  will perform a one-sample sign-flipped permutation test on the estimates directly (perm_on='coef') or the t-statistic (perm_on='t-stat'). Permutation is a bit different than Lm which always permutes based on the t-stat.

        Heteroscedasticity robust standard errors can also be computed, but these are applied at the second-level, *not* at the first level. See the Lm() documentatation for more information about robust standard errors.

        Args:
            robust (bool/str): whether to use heteroscedasticity robust s.e. and optionally which estimator type to use ('hc0','hc3','hac','cluster'). If robust = True, default robust estimator is 'hc0'; default False
            conf_int (str): whether confidence intervals should be computed through bootstrap ('boot') or assuming a t-distribution ('standard'); default 'standard'
            permute (int): if non-zero, computes parameter significance tests by permuting t-stastics rather than parametrically; works with robust estimators
            perm_on (str): permute based on a null distribution of the 'coef' of first-level estimates or the 't-stat' of first-level estimates; default 't-stat'
            rank (bool): convert all predictors and dependent variable to ranks before estimating model; default False
            to_corrs (bool/string): for each first level model estimate a semi-partial or partial correlations instead of betas and perform inference over these partial correlation coefficients. *note* this is different than Lm(); default False
            ztrans_corrs (bool): whether to fisher-z transform (arcsin) first-level correlations before running second-level model. Ignored if to_corrs is False; default True
            summarize (bool): whether to print a model summary after fitting; default True
            verbose (bool): whether to print which model, standard error, confidence interval, and inference type are being fitted
            n_boot (int): how many bootstrap resamples to use for confidence intervals (ignored unless conf_int='boot')
            n_jobs (int): number of cores for parallelizing bootstrapping or permutations; default 1
            n_lags (int): number of lags for robust estimator type 'hac' (ignored unless robust='hac'); default 1
            cluster (str): column name identifying clusters/groups for robust estimator type 'cluster' (ignored unless robust='cluster')

        Returns:
            DataFrame: R style summary() table

        """

        # Alllow summary or summarize for compatibility
        if "summary" in kwargs and "summarize" in kwargs:
            raise ValueError(
                "You specified both summary and summarize, please prefer summarize"
            )
        summarize = kwargs.pop("summarize", True)
        summarize = kwargs.pop("summary", summarize)

        if robust:
            if isinstance(robust, bool):
                robust = "hc0"
            self.se_type = "robust" + " (" + robust + ")"
            if cluster:
                if cluster not in self.data.columns:
                    raise ValueError(
                        "cluster identifier must be an existing column in data"
                    )
                else:
                    cluster = self.data[cluster]
        else:
            self.se_type = "non-robust"
        self.ci_type = (
            conf_int + " (" + str(n_boot) + ")" if conf_int == "boot" else conf_int
        )
        if isinstance(to_corrs, str):
            if to_corrs not in ["semi", "partial"]:
                raise ValueError("to_corrs must be 'semi' or 'partial'")

        if (conf_int == "boot") and (permute is None):
            self.sig_type = "bootstrapped"
        else:
            if permute:
                if perm_on not in ["coef", "t-stat"]:
                    raise ValueError("perm_on must be 't-stat' or 'coef'")
                self.sig_type = "permutation" + " (" + str(permute) + ")"
                if permute is True:
                    raise TypeError(
                        "permute should 'False' or the number of permutations to perform"
                    )
            else:
                self.sig_type = "parametric"

        # Parallelize regression computation for 1st-level models
        par_for = Parallel(n_jobs=n_jobs, backend="multiprocessing")

        if rank:
            self.ranked_data = True
        else:
            self.ranked_data = False

        if to_corrs:
            # Loop over each group and get semi/partial correlation estimates
            # Reminder len(betas) == len(betas) - 1, from normal OLS, since corr of intercept is not computed
            betas = par_for(
                delayed(_corr_group)(
                    self.data,
                    self.formula,
                    self.group,
                    self.data[self.group].unique()[i],
                    self.ranked_data,
                    to_corrs,
                )
                for i in range(self.data[self.group].nunique())
            )
            if ztrans_corrs:
                betas = np.arctanh(betas)
            else:
                betas = np.array(betas)
        else:
            # Loop over each group and fit a separate regression
            betas = par_for(
                delayed(_ols_group)(
                    self.data,
                    self.formula,
                    self.group,
                    self.data[self.group].unique()[i],
                    self.ranked_data,
                )
                for i in range(self.data[self.group].nunique())
            )
            betas = np.array(betas)

        # Get the model matrix formula from patsy to make it more reliable to set the results dataframe index like Lmer
        _, x = dmatrices(self.formula, self.data, 1, return_type="dataframe")
        # Perform an intercept only regression for each beta
        results = []
        perm_ps = []
        for i in range(betas.shape[1]):
            df = pd.DataFrame({"X": np.ones_like(betas[:, i]), "Y": betas[:, i]})
            lm = Lm("Y ~ 1", data=df)
            lm.fit(
                robust=robust,
                conf_int=conf_int,
                summarize=False,
                n_boot=n_boot,
                n_jobs=n_jobs,
                n_lags=n_lags,
            )
            results.append(lm.coefs)
            if permute:
                # sign-flip permutation test for each beta instead to replace p-values
                if perm_on == "coef":
                    return_stat = "mean"
                else:
                    return_stat = "t-stat"
                seeds = np.random.randint(np.iinfo(np.int32).max, size=permute)
                par_for = Parallel(n_jobs=n_jobs, backend="multiprocessing")
                perm_est = par_for(
                    delayed(_permute_sign)(
                        data=betas[:, i], seed=seeds[j], return_stat=return_stat
                    )
                    for j in range(permute)
                )
                perm_est = np.array(perm_est)
                if perm_on == "coef":
                    perm_ps.append(_perm_find(perm_est, betas[:, i].mean()))
                else:
                    perm_ps.append(_perm_find(perm_est, lm.coefs["T-stat"].values))

        results = pd.concat(results, axis=0)
        ivs = self.formula.split("~")[-1].strip().split("+")
        ivs = [e.strip() for e in ivs]
        if to_corrs:
            intercept_pd = dict()
            for c in results.columns:
                intercept_pd[c] = np.nan
            intercept_pd = pd.DataFrame(intercept_pd, index=[0])
            results = pd.concat([intercept_pd, results], ignore_index=True)
        results.index = x.columns
        self.coefs = results
        if to_corrs:
            self.fixef = pd.DataFrame(betas, columns=ivs)
        else:
            self.fixef = pd.DataFrame(betas, columns=x.columns)
        self.fixef.index = self.data[self.group].unique()
        self.fixef.index.name = self.group
        if permute:
            # get signifance stars
            sig = [_sig_stars(elem) for elem in perm_ps]
            # Replace dof and p-vales with permutation results
            if conf_int != "boot":
                self.coefs = self.coefs.drop(columns=["DF", "P-val"])
            if to_corrs:
                self.coefs["Num_perm"] = [np.nan] + [permute] * (
                    self.coefs.shape[0] - 1
                )
                self.coefs["Sig"] = [np.nan] + sig
                self.coefs["Perm-P-val"] = [np.nan] + perm_ps
            else:
                self.coefs["Num_perm"] = [permute] * self.coefs.shape[0]
                self.coefs["Sig"] = sig
                self.coefs["Perm-P-val"] = perm_ps
            self.coefs = self.coefs[
                [
                    "Estimate",
                    "2.5_ci",
                    "97.5_ci",
                    "SE",
                    "Num_perm",
                    "T-stat",
                    "Perm-P-val",
                    "Sig",
                ]
            ]
        self.fitted = True

        # Need to figure out how best to compute predictions and residuals. Should test how Lmer does it, i.e. BLUPs or fixed effects?
        # Option 1) Use only second-level estimates
        # Option 2) Use only first-level estimates and make separate predictions per group
        # self.resid = res
        # self.data['fits'] = y.squeeze() - res
        # self.data['residuals'] = res

        # Fit statistics
        # self.rsquared = np.nan
        # self.rsquared = np.nan
        # self.rsquared_adj = np.nan
        # self.logLike = np.nan
        # self.AIC = np.nan
        # self.BIC = np.nan
        self.iscorrs = to_corrs

        if summarize:
            return self.summary()

    def summary(self):
        """
        Summarize the output of a fitted model.

        Returns:
            pd.DataFrame: R/statsmodels style summary

        """

        if not self.fitted:
            raise RuntimeError("Model must be fitted to generate summary!")

        print("Formula: {}\n".format(self.formula))
        print("Family: {}\n".format(self.family))
        print(
            "Std-errors: {}\tCIs: {} 95%\tInference: {} \n".format(
                self.se_type, self.ci_type, self.sig_type
            )
        )
        print(
            "Number of observations: %s\t Groups: %s\n"
            % (self.data.shape[0], {str(self.group): self.data[self.group].nunique()})
        )
        print("Fixed effects:\n")
        if self.iscorrs:
            if self.iscorrs == "semi":
                corr = "semi-partial"
            else:
                corr = self.iscorrs
            print("Note: {} correlations reported".format(corr))
        return self.coefs.round(3)

    def plot_summary(
        self,
        figsize=(12, 6),
        error_bars="ci",
        ranef=True,
        axlim=None,
        ranef_alpha=0.5,
        coef_fmt="o",
        orient="v",
        **kwargs,
    ):
        """
        Create a forestplot overlaying estimated coefficients with first-level effects. By default display the 95% confidence intervals computed during fitting.

        Args:
            error_bars (str): one of 'ci' or 'se' to change which error bars are plotted; default 'ci'
            ranef (bool): overlay BLUP estimates on figure; default True
            axlim (tuple): lower and upper limit of plot; default min and max of BLUPs
            ranef_alpha (float): opacity of random effect points; default .5
            coef_fmt (str): matplotlib marker style for population coefficients

        Returns:
            plt.axis: matplotlib axis handle
        """

        if not self.fitted:
            raise RuntimeError("Model must be fit before plotting!")
        if orient not in ["h", "v"]:
            raise ValueError("orientation must be 'h' or 'v'")

        m_ranef = self.fixef
        m_fixef = self.coefs.drop("Intercept", axis=0)

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
