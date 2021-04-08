"""
Pymer4 Lm Class
===============

Main class for linear regression models
"""

from copy import copy
import warnings
import numpy as np
import pandas as pd
from patsy import dmatrices
from scipy.stats import t as t_dist
from joblib import Parallel, delayed
from ..stats import rsquared, rsquared_adj
from ..utils import (
    _sig_stars,
    _chunk_boot_ols_coefs,
    _chunk_perm_ols,
    _ols,
    _perm_find,
    _welch_ingredients,
)


class Lm(object):

    """
    Model class to perform OLS regression. Formula specification works just like in R based on columns of a dataframe. Formulae are parsed by patsy which makes it easy to utilize specifiy columns as factors. This is **different** from Lmer. See patsy for more information on the different use cases.

    Args:
        formula (str): Complete lm-style model formula
        data (pandas.core.frame.DataFrame): input data
        family (string): what distribution family (i.e.) link function to use for the generalized model; default is gaussian (linear model)

    Attributes:
        fitted (bool): whether model has been fit
        formula (str): model formula
        data (pd.DataFrame): model copy of input data
        design_matrix (pd.DataFrame): model design matrix determined by patsy
        AIC (float): model akaike information criterion
        logLike (float): model Log-likelihood
        family (string): model family
        warnings (list): warnings output from Python
        coefs (pd.DataFrame): model summary table of parameters
        residuals (numpy.ndarray): model residuals
        fits (numpy.ndarray): model fits/predictions
        estimator (string): 'OLS' or 'WLS'
        se_type (string): how standard errors are computed
        sig_type (string): how inference is performed

    """

    def __init__(self, formula, data, family="gaussian"):

        self.family = family
        # implemented_fams = ['gaussian','binomial']
        if self.family != "gaussian":
            raise NotImplementedError(
                "Currently only linear (family ='gaussian') models supported! "
            )
        self.fitted = False
        self.formula = formula.replace(" ", "")
        self.data = copy(data)
        self.AIC = None
        self.logLike = None
        self.warnings = []
        self.residuals = None
        self.coefs = None
        self.model_obj = None
        self.ci_type = None
        self.se_type = None
        self.sig_type = None
        self.ranked_data = False
        self.estimator = None
        self.design_matrix = None

    def __repr__(self):
        out = "{}(fitted={}, formula={}, family={})".format(
            self.__class__.__module__,
            self.fitted,
            self.formula,
            self.family,
        )
        return out

    def fit(
        self,
        robust=False,
        conf_int="standard",
        permute=False,
        rank=False,
        verbose=False,
        n_boot=500,
        n_jobs=1,
        n_lags=1,
        cluster=None,
        weights=None,
        wls_dof_correction=True,
        **kwargs
    ):
        """
        Fit a variety of OLS models. By default will fit a model that makes parametric assumptions (under a t-distribution) replicating the output of software like R. 95% confidence intervals (CIs) are also estimated parametrically by default. However, empirical bootstrapping can also be used to compute CIs; this procedure resamples with replacement from the data themselves, not residuals or data generated from fitted parameters and will be used for inference unless permutation tests are requested. Permutation testing will shuffle observations to generate a null distribution of t-statistics to perform inference on each regressor (permuted t-test).

        Alternatively, OLS robust to heteroscedasticity can be fit by computing sandwich standard error estimates (good ref: https://bit.ly/2VRb7jK). This is similar to Stata's robust routine. Of the choices below, 'hc1' or 'hc3' are amongst the more popular.
        Robust estimators include:

        - 'hc0': Huber (1967) original sandwich estimator

        - 'hc1': Hinkley (1977) DOF adjustment to 'hc0' to account for small sample sizes (default)

        - 'hc2': different kind of small-sample adjustment of 'hc0' by leverage values in hat matrix

        - 'hc3': MacKinnon and White (1985) HC3 sandwich estimator; provides more robustness in smaller samples than hc2, Long & Ervin (2000)

        - 'hac': Newey-West (1987) estimator for robustness to heteroscedasticity as well as serial auto-correlation at given lags.

        - 'cluster' : cluster-robust standard errors (see Cameron & Miller 2015 for review). Provides robustness to errors that cluster according to specific groupings (e.g. repeated observations within a person/school/site). This acts as post-modeling "correction" for what a multi-level model explicitly estimates and is popular in the econometrics literature. DOF correction differs slightly from stat/statsmodels which use num_clusters - 1, where as pymer4 uses num_clusters - num_coefs

        Finally, weighted-least-squares (WLS) can be computed as an alternative to to hetereoscedasticity robust standard errors. This can be estimated by providing an array or series of weights (1 / variance of each group) with the same length as the number of observations or a column to use to compute group variances (which can be the same as the predictor column). This is often useful if some predictor(s) is categorical (e.g. dummy-coded) and taking into account unequal group variances is desired (i.e. in the simplest case this would be equivalent to peforming Welch's t-test).


        Args:
            robust (bool/str): whether to use heteroscedasticity robust s.e. and optionally which estimator type to use ('hc0','hc1', 'hc2', hc3','hac','cluster'). If robust = True, default robust estimator is 'hc1'; default False
            conf_int (str): whether confidence intervals should be computed through bootstrap ('boot') or assuming a t-distribution ('standard'); default 'standard'
            permute (int): if non-zero, computes parameter significance tests by permuting t-stastics rather than parametrically; works with robust estimators
            rank (bool): convert all predictors and dependent variable to ranks before estimating model; default False
            summarize (bool): whether to print a model summary after fitting; default True
            verbose (bool): whether to print which model, standard error, confidence interval, and inference type are being fitted
            n_boot (int): how many bootstrap resamples to use for confidence intervals (ignored unless conf_int='boot')
            n_jobs (int): number of cores for parallelizing bootstrapping or permutations; default 1
            n_lags (int): number of lags for robust estimator type 'hac' (ignored unless robust='hac'); default 1
            cluster (str): column name identifying clusters/groups for robust estimator type 'cluster' (ignored unless robust='cluster')
            weights (string/pd.Series/np.ndarray): weights to perform WLS instead of OLS. Pass in a column name in data to use to compute group variances and automatically adjust dof. Otherwise provide an array or series containing 1 / variance of each observation, in which case dof correction will not occur.
            wls_dof_correction (bool): whether to apply Welch-Satterthwaite approximate correction for dof when using weights based on an existing column in the data, ignored otherwise. Set to False to force standard dof calculation

        Returns:
            pd.DataFrame: R/statsmodels style summary


        Examples:

            Simple multiple regression model with parametric assumptions

            >>> model = Lm('DV ~ IV1 + IV2', data=df)
            >>> model.fit()

            Same as above but with robust standard errors

            >>> model.fit(robust='hc1')

            Same as above but with cluster-robust standard errors. The cluster argument should refer to a column in the dataframe.

            >>> model.fit(robust='cluster', cluster='Group')

            Simple regression with categorical predictor, i.e. between groups t-test assuming equal variances

            >>> model = Lm('DV ~ Group', data=df)
            >>> model.fit()

            Same as above but don't assume equal variances and have pymer4 compute the between group variances automatically, i.e. WLS (preferred).

            >>> model.fit(weights='Group')

            Manually compute the variance of each group and use the inverse of that as the weights. In this case WLS is estimated but dof correction won't be applied because it's not trivial to compute.

            >>> weights = 1 / df.groupby("Group")['DV'].transform(np.var,ddof=1)
            model.fit(weights=weights)

        """

        # Alllow summary or summarize for compatibility
        if "summary" in kwargs and "summarize" in kwargs:
            raise ValueError(
                "You specified both summary and summarize, please prefer summarize"
            )
        summarize = kwargs.pop("summarize", True)
        summarize = kwargs.pop("summary", summarize)

        if permute and permute < 500:
            w = "Permutation testing < 500 permutations is not recommended"
            warnings.warn(w)
            self.warnings.append(w)
        elif permute is True:
            raise TypeError(
                "permute should 'False' or the number of permutations to perform"
            )
        if robust:
            if isinstance(robust, bool):
                robust = "hc1"
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

        if self.family == "gaussian":
            if verbose:
                if rank:
                    print_rank = "rank"
                else:
                    print_rank = "linear"
                if not robust:
                    print_robust = "non-robust"
                else:
                    print_robust = "robust " + robust

                if conf_int == "boot":
                    print(
                        "Fitting "
                        + print_rank
                        + " model with "
                        + print_robust
                        + " standard errors and \n"
                        + str(n_boot)
                        + "bootstrapped 95% confidence intervals...\n"
                    )
                else:
                    print(
                        "Fitting "
                        + print_rank
                        + " model with "
                        + print_robust
                        + " standard errors\nand 95% confidence intervals...\n"
                    )

                if permute:
                    print(
                        "Using {} permutations to determine significance...".format(
                            permute
                        )
                    )

        self.ci_type = (
            conf_int + " (" + str(n_boot) + ")" if conf_int == "boot" else conf_int
        )
        if (conf_int == "boot") and (permute is None):
            self.sig_type = "bootstrapped"
        else:
            if permute:
                self.sig_type = "permutation" + " (" + str(permute) + ")"
            else:
                self.sig_type = "parametric"

        # Parse formula using patsy to make design matrix
        if rank:
            self.ranked_data = True
            ddat = self.data.rank()
        else:
            self.ranked_data = False
            ddat = self.data

        # Handle weights if provided
        if isinstance(weights, str):
            if weights not in self.data.columns:
                raise ValueError(
                    "If weights is a string it must be a column that exists in the data"
                )
            else:
                dv = self.formula.split("~")[0]
                weight_groups = self.data.groupby(weights)
                weight_vals = 1 / weight_groups[dv].transform(np.var, ddof=1)
        else:
            weight_vals = weights
        if weights is None:
            self.estimator = "OLS"
        else:
            self.estimator = "WLS"

        y, x = dmatrices(self.formula, ddat, 1, return_type="dataframe")
        self.design_matrix = x

        # Compute standard estimates
        b, se, t, res = _ols(
            x,
            y,
            robust,
            all_stats=True,
            n_lags=n_lags,
            cluster=cluster,
            weights=weight_vals,
        )
        if cluster is not None:
            # Cluster corrected dof (num clusters - num coef)
            # Differs from stats and statsmodels which do num cluster - 1
            # Ref: http://cameron.econ.ucdavis.edu/research/Cameron_Miller_JHR_2015_February.pdf
            df = cluster.nunique() - x.shape[1]
        else:
            df = x.shape[0] - x.shape[1]
            if isinstance(weights, str) and wls_dof_correction:
                if weight_groups.ngroups != 2:
                    w = "Welch-Satterthwait DOF correction only supported for 2 groups in the data"
                    warnings.warn(w)
                    self.warnings.append(w)
                else:
                    welch_ingredients = np.array(
                        self.data.groupby(weights)[dv]
                        .apply(_welch_ingredients)
                        .values.tolist()
                    )
                    df = (
                        np.power(welch_ingredients[:, 0].sum(), 2)
                        / welch_ingredients[:, 1].sum()
                    )

        p = 2 * (1 - t_dist.cdf(np.abs(t), df))
        df = np.array([df] * len(t))
        sig = np.array([_sig_stars(elem) for elem in p])

        if conf_int == "boot":

            # Parallelize bootstrap computation for CIs
            par_for = Parallel(n_jobs=n_jobs, backend="multiprocessing")

            # To make sure that parallel processes don't use the same random-number generator pass in seed (sklearn trick)
            seeds = np.random.randint(np.iinfo(np.int32).max, size=n_boot)

            # Since we're bootstrapping coefficients themselves we don't need the robust info anymore
            boot_betas = par_for(
                delayed(_chunk_boot_ols_coefs)(
                    dat=self.data, formula=self.formula, weights=weights, seed=seeds[i]
                )
                for i in range(n_boot)
            )

            boot_betas = np.array(boot_betas)
            ci_u = np.percentile(boot_betas, 97.5, axis=0)
            ci_l = np.percentile(boot_betas, 2.5, axis=0)

        else:
            # Otherwise we're doing parametric CIs
            ci_u = b + t_dist.ppf(0.975, df) * se
            ci_l = b + t_dist.ppf(0.025, df) * se

        if permute:
            # Permuting will change degrees of freedom to num_iter and p-values
            # Parallelize computation
            # Unfortunate monkey patch that robust estimation hangs with multiple processes; maybe because of function nesting level??
            # _chunk_perm_ols -> _ols -> _robust_estimator
            if robust:
                n_jobs = 1
            par_for = Parallel(n_jobs=n_jobs, backend="multiprocessing")
            seeds = np.random.randint(np.iinfo(np.int32).max, size=permute)
            perm_ts = par_for(
                delayed(_chunk_perm_ols)(
                    x=x,
                    y=y,
                    robust=robust,
                    n_lags=n_lags,
                    cluster=cluster,
                    weights=weights,
                    seed=seeds[i],
                )
                for i in range(permute)
            )
            perm_ts = np.array(perm_ts)

            p = []
            for col, fit_t in zip(range(perm_ts.shape[1]), t):
                p.append(_perm_find(perm_ts[:, col], fit_t))
            p = np.array(p)
            df = np.array([permute] * len(p))
            sig = np.array([_sig_stars(elem) for elem in p])

        # Make output df
        results = np.column_stack([b, ci_l, ci_u, se, df, t, p, sig])
        results = pd.DataFrame(results)
        results.index = x.columns
        results.columns = [
            "Estimate",
            "2.5_ci",
            "97.5_ci",
            "SE",
            "DF",
            "T-stat",
            "P-val",
            "Sig",
        ]
        results[
            ["Estimate", "2.5_ci", "97.5_ci", "SE", "DF", "T-stat", "P-val"]
        ] = results[
            ["Estimate", "2.5_ci", "97.5_ci", "SE", "DF", "T-stat", "P-val"]
        ].apply(
            pd.to_numeric, args=("coerce",)
        )

        if permute:
            results = results.rename(columns={"DF": "Num_perm", "P-val": "Perm-P-val"})

        self.coefs = results
        self.fitted = True
        self.residuals = res
        self.fits = (y.squeeze() - res).values
        self.data["fits"] = (y.squeeze() - res).values
        self.data["residuals"] = res

        # Fit statistics
        if "Intercept" in self.design_matrix.columns:
            center_tss = True
        else:
            center_tss = False
        self.rsquared = rsquared(y.squeeze(), res, center_tss)
        self.rsquared_adj = rsquared_adj(
            self.rsquared, len(res), len(res) - x.shape[1], center_tss
        )
        half_obs = len(res) / 2.0
        ssr = np.dot(res, res.T)
        self.logLike = (-np.log(ssr) * half_obs) - (
            (1 + np.log(np.pi / half_obs)) * half_obs
        )
        self.AIC = 2 * x.shape[1] - 2 * self.logLike
        self.BIC = np.log((len(res))) * x.shape[1] - 2 * self.logLike

        if summarize:
            return self.summary()

    def summary(self):
        """
        Summarize the output of a fitted model.

        Returns:
            pd.DataFrame: R/statsmodels style summary

        """

        if not self.fitted:
            raise RuntimeError("Model must be fit to generate summary!")

        print("Formula: {}\n".format(self.formula))
        print("Family: {}\t Estimator: {}\n".format(self.family, self.estimator))
        print(
            "Std-errors: {}\tCIs: {} 95%\tInference: {} \n".format(
                self.se_type, self.ci_type, self.sig_type
            )
        )
        print(
            "Number of observations: %s\t R^2: %.3f\t R^2_adj: %.3f\n"
            % (self.data.shape[0], self.rsquared, self.rsquared_adj)
        )
        print(
            "Log-likelihood: %.3f \t AIC: %.3f\t BIC: %.3f\n"
            % (self.logLike, self.AIC, self.BIC)
        )
        print("Fixed effects:\n")
        return self.coefs.round(3)

    def to_corrs(self, corr_type="semi", ztrans_corrs=False):
        """
        Transform fitted model coefficients (excluding the intercept) to partial or semi-partial correlations with dependent variable. The is useful for rescaling coefficients to a correlation scale (-1 to 1) and does **not** change how inferences are performed. Semi-partial correlations are computed as the correlation between a DV and each predictor *after* the influence of all other predictors have been regressed out from that predictor. They are interpretable in the same way as the original coefficients. Partial correlations reflect the unique variance a predictor explains in the DV accounting for correlations between predictors *and* what is not explained by other predictors; this value is always >= the semi-partial correlation. They are *not* interpretable in the same way as the original coefficients. Partial correlations are computed as the correlations between a DV and each predictor *after* the influence of all other predictors have been regressed out from that predictor *and* the DV. Good ref: https://bit.ly/2GNwXh5

        Args:
            corr_type (string): 'semi' or 'partial'
            ztrans_partial_corrs (bool): whether to fisher z-transform (arctan) partial correlations before reporting them; default False

        Returns:
            pd.Series: partial or semi-partial correlations

        """

        if not self.fitted:
            raise RuntimeError(
                "Model must be fit before partial correlations can be computed"
            )
        if corr_type not in ["semi", "partial"]:
            raise ValueError("corr_type must be 'semi' or 'partial'")
        from scipy.stats import pearsonr

        corrs = []
        corrs.append(np.nan)  # don't compute for intercept
        for c in self.design_matrix.columns[1:]:
            dv = self.formula.split("~")[0]
            other_preds = [e for e in self.design_matrix.columns[1:] if e != c]
            right_side = "+".join(other_preds)
            y, x = dmatrices(
                c + "~" + right_side, self.data, 1, return_type="dataframe"
            )
            pred_m_resid = _ols(
                x,
                y,
                robust=False,
                n_lags=1,
                cluster=None,
                all_stats=False,
                resid_only=True,
            )
            y, x = dmatrices(
                dv + "~" + right_side, self.data, 1, return_type="dataframe"
            )
            if corr_type == "semi":
                dv_m_resid = y.values.squeeze()
            elif corr_type == "partial":
                dv_m_resid = _ols(
                    x,
                    y,
                    robust=False,
                    n_lags=1,
                    cluster=None,
                    all_stats=False,
                    resid_only=True,
                )
            corrs.append(pearsonr(dv_m_resid, pred_m_resid)[0])
        if ztrans_corrs:
            corrs = np.arctanh(corrs)
        return pd.Series(corrs, index=self.coefs.index)

    def predict(self, data):
        """
        Make predictions given new data. Input must be a dataframe that contains the same columns as the model.matrix excluding the intercept (i.e. all the predictor variables used to fit the model). Will automatically use/ignore intercept to make a prediction if it was/was not part of the original fitted model.

        Args:
            data (pd.DataFrame): input data to make predictions on

        Returns:
            np.ndarray: prediction values

        """

        required_cols = self.design_matrix.columns[1:]
        if not all([col in data.columns for col in required_cols]):
            raise ValueError("Column names do not match all fixed effects model terms!")
        X = data[required_cols]
        coefs = self.coefs.loc[:, "Estimate"].values
        if self.coefs.index[0] == "Intercept":
            preds = np.dot(np.column_stack([np.ones(X.shape[0]), X]), coefs)
        else:
            preds = np.dot(X, coefs[1:])
        return preds
