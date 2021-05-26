"""User-facing statistics functions and tests."""

__all__ = [
    "discrete_inverse_logit",
    "cohens_d",
    "perm_test",
    "boot_func",
    "tost_equivalence",
    "welch_dof",
    "vif",
]

__author__ = ["Eshin Jolly"]
__license__ = ["MIT"]

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import pearsonr, spearmanr, ttest_ind, ttest_rel, ttest_1samp
from functools import partial

from pymer4.utils import (
    _check_random_state,
    _welch_ingredients,
    _mean_diff,
)
from joblib import Parallel, delayed

MAX_INT = np.iinfo(np.int32).max


def vif(df, has_intercept=True, exclude_intercept=True, tol=5.0, check_only=False):
    """
    Compute variance inflation factor amongst columns of a dataframe to be used as a design matrix. Uses the same method as Matlab and R (diagonal elements) of the inverted correlation matrix. Prints a warning if any vifs are >= to tol. If check_only is true it will only return a 1 if any vifs are higher than tol.

    Args:
        df (pandas.DataFrame): dataframe of design matrix output from patsy
        has_intercept (bool): whether the first column of the dataframe is the intercept
        exclude_intercept (bool): exclude intercept from computation and assumed intercept is the first column; default True
        tol (float): tolerance check to print warning if any vifs exceed this value
        check_only (bool): restrict return to a dictionary of vifs that exceed tol only rather than all; default False

    Returns:
        dict: dictionary with keys as column names and values as vifs
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("input needs to be a pandas dataframe")
    if has_intercept:
        if df.iloc[:, 0].sum() != df.shape[0]:
            raise ValueError(
                "has_intercept is true, but first column does not appear to be a constant"
            )
        if exclude_intercept:
            corr_mat = df.iloc[:, 1:].corr()
            keys = np.array(df.iloc[:, 1:].columns)
    else:
        corr_mat = df.corr()
        keys = np.array(df.columns)

    try:
        vifs = np.diag(np.linalg.inv(corr_mat), 0)
        mask = np.where(vifs >= tol)
        if mask:
            high_vifs = dict(zip(keys[mask], vifs[mask]))
        else:
            high_vifs = {}
        if check_only:
            return high_vifs
        else:
            return high_vifs, dict(zip(keys, vifs))
    except np.linalg.LinAlgError:
        print("ERROR: Cannot compute vifs! Design Matrix is strictly singular.")


def discrete_inverse_logit(arr):
    """Apply a discretized inverse logit transform to an array of values. Useful for converting normally distributed values to binomial classes.

    Args:
        arr (np.array): 1d numpy array

    Returns:
        np.array: transformed values
    """
    probabilities = expit(arr)
    out = np.random.binomial(1, probabilities)
    return out


def cohens_d(
    x, y=None, paired=False, n_boot=1000, equal_var=False, value=0, n_jobs=1, seed=None
):
    """
    Compute Cohen's d for one or two samples (paired or independent). For paired samples Cohen's Dz is computed (ref: https://bit.ly/2J54P61). If x and y are not the same size this will use the same pooled SD calculation in Welch's ttest to account for unequal variances. Unequal variance calculation will almost always produce a *smaller* estimate than the standard formula, except as the variance of the group with fewer observations increases. In that case, this estimate can be *larger* than the standard formula. This can be turned off with the equal_var=True argument. Percentile boot-strapped confidence intervals can also be returned

    Args:
        x (list-like): array or list of observations from first group
        y (list-like): array or list of observations from second group or second set of observations from the same group; optional
        paired (bool): whether to treat x any y (if provided) as paired or independent
        n_boot (int): number of bootstrap samples to run; set to 0 to skip computing
        equal_var (bool): should we pool standard deviation as in Welch's t-test
        value (float): a value to see if the effect size is bigger than; `eff size - value` will be computed; default 0
        n_jobs (int): number of parallel cores to use for bootstraping; default 1
        seed (int or None): numerical seed for reproducibility of bootstrapping

    Returns:
        Multiple:

            - **effect_size** (*float*): cohen's d

            - **ci** (*np.array*): lower and upper bounds of 95% bootstrapped confidence intervals; optional

    """

    if y is None:
        eff = x.mean() / x.std(ddof=1)
    else:
        if paired:
            # Cohen's Dz
            if (y is None) or (len(x) != len(y)):
                raise ValueError(
                    "with paired=True, both and x and y must be provided and must have the same number of observations"
                )
            numerator = np.subtract(x, y).mean() - value
            denominator = x.std(ddof=1) - y.std(ddof=1)
            eff = numerator / denominator

        else:
            # Cohen's D
            m1, s1, ss1, m2, s2, ss2 = (
                x.mean(),
                x.var(ddof=1),
                x.size,
                y.mean(),
                y.var(ddof=1),
                y.size,
            )

            if equal_var:
                pooled_sd = np.sqrt(np.mean([s1, s2]))
            else:
                pooled_sd = np.sqrt(
                    (((ss1 - 1) * s1 + ((ss2 - 1) * s2))) / (ss1 + ss2 - 2)
                )

            numerator = m1 - m2 - value
            eff = numerator / pooled_sd

    if n_boot:
        random_state = _check_random_state(seed)
        seeds = random_state.randint(MAX_INT, size=n_boot)
        par_for = Parallel(n_jobs=n_jobs, backend="multiprocessing")
        boots = par_for(
            delayed(_cohens_d)(x, y, paired, equal_var, value, random_state=seeds[i])
            for i in range(n_boot)
        )
        ci_u = np.percentile(boots, 97.5, axis=0)
        ci_l = np.percentile(boots, 2.5, axis=0)
        return eff, (ci_l, ci_u)
    else:
        return eff


def _cohens_d(x, y, paired, equal_var, value, random_state):
    """For use in parallel cohens_d"""
    random_state = _check_random_state(random_state)
    if paired:
        idx = np.random.choice(np.arange(len(x)), size=x.size, replace=True)
        x, y = x[idx], y[idx]
    else:
        x = random_state.choice(x, size=x.size, replace=True)
        if y is not None:
            y = random_state.choice(y, size=y.size, replace=True)
    return cohens_d(x, y, 0, equal_var, value)


def perm_test(
    x,
    y=None,
    stat="tstat",
    n_perm=1000,
    equal_var=False,
    tails=2,
    return_dist=False,
    n_jobs=1,
    seed=None,
):
    """
    General purpose permutation test between two samples. Can handle a wide varierty of permutation tests including ttest, paired ttest, mean diff test, cohens d, pearson r, spearman r.

    Args:
        x (list-like): array or list of observations from first group
        y (list-like): array or list of observations from second group
        stat (string): one of ['tstat', 'tstat-paired', 'mean', 'cohensd', 'pearsonr', 'spearmanr']; 'mean' will just compute permutations on the difference between the mean of x and mean of y. Useful if statistics are precomputed (e.g. x and y contain correlation values, or t-stats).
        n_perm (int): number of permutations; set to 0 to return parametric results
        equal_var (bool): should assume equal variances for tstat and cohensd
        tails (int): perform one or two-tailed p-value computations; default 2
        return_dists (bool): return permutation distribution
        n_jobs (int): number of parallel cores to use for bootstraping; default 1
        seed (int): for reproducing results

    Returns:
        Multiple:

            - **original_stat** (*float*): the original statistic

            - **perm_p_val** (*float*): the permuted p-value

            - **perm_dist** (*np.array*): array of permuted statistic; optional

    """

    if ((y is None) or isinstance(y, (float, int))) and (
        stat in ["pearsonr", "spearmanr"]
    ):
        raise ValueError("y must be provided for 'pearsonr' or 'spearmanr'")

    if stat == "tstat":
        if isinstance(y, (list, np.ndarray)):
            func = partial(ttest_ind, equal_var=equal_var)
        else:
            if y is None:
                y = 0
            func = partial(ttest_1samp)
        multi_return = True
    elif stat == "tstat-paired":
        func = ttest_rel
        multi_return = True
        if len(x) != len(y):
            raise ValueError("x and y must be the same length")
    elif stat == "mean":

        def func(x, y):
            if y is not None:
                if isinstance(y, (list, np.ndarray)):
                    return x.mean() - y.mean()
                elif isinstance(y, (float, int)):
                    raise NotImplementedError(
                        "One-sample mean test with a scalar y is not currently supported"
                    )
            else:
                return x.mean()

        multi_return = False
    elif stat == "cohensd":
        func = partial(cohens_d, equal_var=equal_var, n_boot=0)
        multi_return = False
    elif stat == "pearsonr":
        func = pearsonr
        multi_return = True
        if len(x) != len(y):
            raise ValueError("x and y must be the same length")
    elif stat == "spearmanr":
        func = spearmanr
        multi_return = True
        if len(x) != len(y):
            raise ValueError("x and y must be the same length")
    else:
        raise ValueError(
            "stat must be in ['tstat', 'tstat-paired', 'mean', 'cohensd', 'pearsonr', 'spearmanr']"
        )

    # Get original statistic
    original_stat = func(x, y)
    if multi_return:
        original_stat = original_stat[0]

    # Permute
    if n_perm == 0:
        return func(x, y)
    else:
        random_state = _check_random_state(seed)
        seeds = random_state.randint(MAX_INT, size=n_perm)
        par_for = Parallel(n_jobs=n_jobs, backend="multiprocessing")
        perms = par_for(
            delayed(_perm_test)(x, y, stat, equal_var, random_state=seeds[i])
            for i in range(n_perm)
        )
        if multi_return:
            perms = [elem[0] for elem in perms]

        denom = float(len(perms)) + 1

        if tails == 2:
            numer = np.sum(np.abs(perms) >= np.abs(original_stat)) + 1
        elif tails == 1:
            if original_stat >= 0:
                numer = np.sum(perms >= original_stat) + 1
            else:
                numer = np.sum(perms <= original_stat) + 1
        else:
            raise ValueError("tails must be 1 or 2")
        p = numer / denom
        if return_dist:
            return original_stat, p, perms
        else:
            return original_stat, p


def _perm_test(x, y, stat, equal_var, random_state):
    """For use in parallel perm_test"""
    random_state = _check_random_state(random_state)
    if stat in ["pearsonr", "spearmanr"]:
        y = random_state.permutation(y)
    elif stat in ["tstat", "cohensd", "mean"]:
        if y is None:
            x = x * random_state.choice([1, -1], len(x))
        elif isinstance(y, (float, int)):
            x -= y
            x = x * random_state.choice([1, -1], len(x))
        else:
            shuffled_combined = random_state.permutation(np.hstack([x, y]))
            x, y = shuffled_combined[: x.size], shuffled_combined[x.size :]
    elif (stat == "tstat-paired") or (y is None):
        x = x * random_state.choice([1, -1], len(x))

    return perm_test(x, y, stat, equal_var=equal_var, n_perm=0)


def boot_func(
    x, y=None, func=None, func_args={}, paired=False, n_boot=500, n_jobs=1, seed=None
):
    """
    Bootstrap an arbitrary function by resampling from x and y independently or jointly.

    Args:
        x (list-like): list of values for first group
        y (list-like): list of values for second group; optional
        function (callable): function that accepts x or y
        paired (bool): whether to resample x and y jointly or independently
        n_boot (int): number of bootstrap iterations
        n_jobs (int): number of parallel cores; default 1

    Returns:
        Multiple:

            - **original_stat** (*float*): function result with given data

            - **ci** (*np.array*): lower and upper bounds of 95% confidence intervals

    """

    if not callable(func):
        raise TypeError("func must be a valid callable function")

    orig_result = func(x, y, **func_args)
    if n_boot:
        random_state = _check_random_state(seed)
        seeds = random_state.randint(MAX_INT, size=n_boot)
        par_for = Parallel(n_jobs=n_jobs, backend="multiprocessing")
        boots = par_for(
            delayed(_boot_func)(
                x, y, func, func_args, paired, **func_args, random_state=seeds[i]
            )
            for i in range(n_boot)
        )
        ci_u = np.percentile(boots, 97.5, axis=0)
        ci_l = np.percentile(boots, 2.5, axis=0)
        return orig_result, (ci_l, ci_u)
    else:
        return orig_result


def _boot_func(x, y, func, func_args, paired, random_state):
    """For use in parallel boot_func"""
    random_state = _check_random_state(random_state)
    if paired:
        idx = np.random.choice(np.arange(len(x)), size=x.size, replace=True)
        x, y = x[idx], y[idx]
    else:
        x = random_state.choice(x, size=x.size, replace=True)
        if y is not None:
            y = random_state.choice(y, size=y.size, replace=True)
    return boot_func(x, y, func, func_args, paired=paired, n_boot=0)


def tost_equivalence(
    x,
    y,
    lower,
    upper,
    paired=False,
    equal_var=False,
    n_perm=1000,
    n_boot=5000,
    plot=False,
    seed=None,
):
    """
    Function to perform equivalence testing using TOST: two-one-sided-tests (Lakens et al, 2018). This works by defining a lower and upper bound of an "equivalence" range for the mean difference between x and y. This is a user-defined range that one might not feel is a particularly meangingful mean difference; conceptually similar to the Bayesian "region of practical equivalence (rope)." Specifically this uses, two one-sided t-tests against and lower and upper seperately to find out whether lower < mean diff < higher. n_perm only controls the permutation for the original two-sided test.

    Args:
        x (list-like): array or list of observations from first group
        y (list-like): array or list of observations from second group
        lower (float): lower bound of equivalence region
        upper (float): upper bound of equivalence region
        equal_var (bool): should assume equal variances for t-stat and effect size calcs
        n_perm (int): number of times to permute groups; set to 0 to turn off
        n_boot (int): number of bootstrap samples for confidence intervals
        plot (bool): return an equivalence plot depicting where the mean difference and 95% CIs fall relative to the equivalence range
        return_dists (bool): optionally return the permuted distributions

    Returns:
        dict: a dictionary of TOST results

    """

    from scipy.stats import t as t_dist
    import matplotlib.pyplot as plt
    import seaborn as sns

    def _calc_stats(x, y, val, equal_var):
        n1, m1, v1, n2, m2, v2 = (
            x.size,
            x.mean(),
            x.var(ddof=1),
            y.size,
            y.mean(),
            y.var(ddof=1),
        )

        numerator = m1 - m2 - val
        if equal_var:
            # From scipy
            df = n1 + n2 - 2.0
            svar = ((n1 - 1) * v1 + (n2 - 1) * v2) / df
            denom = np.sqrt(svar * (1.0 / n1 + 1.0 / n2))
        else:
            vn1 = v1 / n1
            vn2 = v2 / n2
            with np.errstate(divide="ignore", invalid="ignore"):
                df = (vn1 + vn2) ** 2 / (vn1 ** 2 / (n1 - 1) + vn2 ** 2 / (n2 - 1))
            denom = np.sqrt(vn1 + vn2)
        return numerator / denom, df

    # Just get df calculation from sub-function
    _, df = _calc_stats(x, y, 0, equal_var)
    tstat_orig, pval_orig = perm_test(
        x, y, stat="tstat", n_perm=n_perm, equal_var=equal_var, seed=seed
    )
    tstat_lower, df = _calc_stats(x, y, lower, equal_var)
    tstat_upper, df = _calc_stats(x, y, upper, equal_var)
    mdiff = x.mean() - y.mean()

    # Parametric assumptions for each one-sided test
    pval_lower = t_dist.sf(tstat_lower, df)
    pval_upper = t_dist.cdf(tstat_upper, df)

    # Attempted one-side permutation tests, but computing p-values is non-trivial due to how signs can flip based on the equivalence range and test statistic. Revisit at some point in the future.
    # else:
    #     perm_t_origs, perm_t_lowers, perm_t_uppers = [], [], []
    #     for i in range(n_perm):
    #         shuffled_combined = np.random.permutation(np.hstack([x, y]))
    #         new_x, new_y = shuffled_combined[:
    #                                          x.size], shuffled_combined[x.size:]
    #         perm_t_orig, _ = _calc_stats(new_x, new_y, 0, equal_var)
    #         perm_t_lower, _ = _calc_stats(new_x, new_y, upper, equal_var)
    #         perm_t_upper, _ = _calc_stats(new_x, new_y, lower, equal_var)
    #         perm_t_origs.append(perm_t_orig)
    #         perm_t_lowers.append(perm_t_lower)
    #         perm_t_uppers.append(perm_t_upper)

    #     if lower < mdiff < upper:
    #         # upper = orange line is right of orange dist
    #         pval_upper = np.mean(tstat_upper >= perm_t_upper)
    #         # lower = blue line is left of blue dist
    #         pval_lower = np.mean(tstat_lower <= perm_t_lower)
    #     else:
    #         # upper = prob orange line, right of blue dist
    #         pval_upper = np.mean(tstat_upper >= perm_t_lower)
    #         # lower = prob blue line, left of orange dist
    #         pval_lower = np.mean(tstat_lower <= perm_t_upper)

    #     p_orig = np.mean(np.abs(perm_t_origs) >= np.abs(tstat_orig))
    #     res = []  # just to be consistent with above
    #     res.append(tstat_orig)
    #     res.append(p_orig)

    result = {}
    result["original"] = {"m": mdiff, "t": tstat_orig, "p": pval_orig}
    result["lower"] = {"m": lower, "t": tstat_lower, "p": pval_lower}
    result["upper"] = {"m": upper, "t": tstat_upper, "p": pval_upper}

    # Effect size bootstrapped
    d, (dlb, dub) = cohens_d(x, y, n_boot=n_boot, equal_var=equal_var, seed=seed)
    result["cohens_d"] = {"m": d, "CI_lb": dlb, "CI_ub": dub}

    # Some results text for interpretation
    if pval_lower < 0.05 and pval_upper < 0.05:
        result["In_Equivalence_Range"] = True
    else:
        result["In_Equivalence_Range"] = False
    if pval_orig < 0.05:
        result["Means_Are_Different"] = True
    else:
        result["Means_Are_Different"] = False

    if plot:
        # Get mean diff
        m, (lb, ub) = boot_func(x, y, _mean_diff, n_boot=n_boot)
        _, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.plot(m, 0, "o", markersize=18, color="black")
        ax.hlines(y=0, xmin=m, xmax=ub, linestyle="-", linewidth=6)
        ax.hlines(y=0, xmin=lb, xmax=m, linestyle="-", linewidth=6)
        ax.vlines(x=lower, ymin=-1, ymax=1, linestyles="--", linewidth=2)
        ax.vlines(x=upper, ymin=-1, ymax=1, linestyles="--", linewidth=2)
        ax.vlines(x=0, ymin=-1, ymax=1, linestyles="--", linewidth=2, alpha=0.5)
        min_plot = np.min([lb, lower])
        min_plot -= np.abs(min_plot / 2)
        max_plot = np.max([ub, upper])
        max_plot += np.abs(max_plot / 2)
        _ = ax.set(xlim=(min_plot, max_plot), xlabel="Mean Difference", yticks=[])
        ax.text(
            0,
            1,
            f"Equivalence bounds: [{lower}  {upper}]\nMean diff: {np.round(m,3)} [{np.round(lb,3)}  {np.round(ub,3)}]",
            horizontalalignment="center",
            fontsize=14,
        )
        sns.despine()

    return result


def welch_dof(x, y):
    """
    Compute adjusted dof via Welch-Satterthwaite equation

    Args:
        x (np.ndarray): 1d numpy array
        y (np.ndarray): 1d numpy array

    Returns:
        float: degrees of freedom
    """

    if isinstance(x, np.ndarray) and isinstance(y, np.ndarray):

        x_numerator, x_denominator = _welch_ingredients(x)
        y_numerator, y_denominator = _welch_ingredients(y)

        return np.power(x_numerator + y_numerator, 2) / (x_denominator + y_denominator)
    else:
        raise TypeError("Both x and y must be 1d numpy arrays")


# def lrt(models):
#     """
#     WARNING EXPERIMENTAL!
#     Compute a likelihood ratio test between models. This produces similar but not identical results to R's anova() function when comparing models. Will automatically determine the the model order based on comparing all models to the one that has the fewest parameters.

#     Todo:
#     0) Figure out discrepancy with R result
#     1) Generalize function to perform LRT, or vuong test
#     2) Offer nested and non-nested vuong test, as well as AIC/BIC correction
#     3) Given a single model expand out to all separate term tests
#     """

#     raise NotImplementedError("This function is not yet implemented")

#     if not isinstance(models, list):
#         models = [models]
#     if len(models) < 2:
#         raise ValueError("Must have at least 2 models to perform comparison")

#     # Get number of coefs for each model
#     all_params = []
#     for m in models:
#         all_params.append(_get_params(m))

#     # Sort from fewest params to most
#     all_params = np.array(all_params)
#     idx = np.argsort(all_params)
#     all_params = all_params[idx]
#     models = np.array(models)[idx]

#     model_pairs = list(product(models, repeat=2))

#     model_pairs = model_pairs[1 : len(models)]
#     s = []
#     for p in model_pairs:
#         s.append(_lrt(p))
#     out = pd.DataFrame()
#     for i, m in enumerate(models):
#         pval = s[i - 1] if i > 0 else np.nan
#         out = out.append(
#             pd.DataFrame(
#                 {
#                     "model": m.formula,
#                     "DF": m.coefs.loc["Intercept", "DF"],
#                     "AIC": m.AIC,
#                     "BIC": m.BIC,
#                     "log-likelihood": m.logLike,
#                     "P-val": pval,
#                 },
#                 index=[0],
#             ),
#             ignore_index=True,
#         )
#     out["Sig"] = out["P-val"].apply(lambda x: _sig_stars(x))
#     out = out[["model", "log-likelihood", "AIC", "BIC", "DF", "P-val", "Sig"]]
#     return out


def rsquared(y, res, has_constant=True):
    """
    Compute the R^2, coefficient of determination. This statistic is a ratio of "explained variance" to "total variance"

    Args:
        y (np.ndarray): 1d array of dependent variable
        res (np.ndarray): 1d array of residuals
        has_constant (bool): whether the fitted model included a constant (intercept)

    Returns:
        float: coefficient of determination
    """

    y_mean = y.mean()
    rss = np.sum(res ** 2)
    if has_constant:
        tss = np.sum((y - y_mean) ** 2)
    else:
        tss = np.sum(y ** 2)
    return 1 - (rss / tss)


def rsquared_adj(r, nobs, df_res, has_constant=True):
    """
    Compute the adjusted R^2, coefficient of determination.

    Args:
        r (float): rsquared value
        nobs (int): number of observations the model was fit on
        df_res (int): degrees of freedom of the residuals (nobs - number of model params)
        has_constant (bool): whether the fitted model included a constant (intercept)

    Returns:
        float: adjusted coefficient of determination
    """

    if has_constant:
        return 1.0 - (nobs - 1) / df_res * (1.0 - r)
    else:
        return 1.0 - nobs / df_res * (1.0 - r)
