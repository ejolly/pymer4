from __future__ import division
import numpy as np
from pymer4.stats import cohens_d, perm_test, boot_func, tost_equivalence, _mean_diff


def test_cohens_d():
    x = np.random.normal(loc=2, size=100)
    y = np.random.normal(loc=2.2, size=100)
    result = cohens_d(x, y, n_jobs=1, n_boot=500)
    assert len(result) == 2
    result = cohens_d(x, y, paired=True, n_jobs=1, n_boot=500)
    assert len(result) == 2
    y = np.random.normal(loc=2.5, size=15)
    result = cohens_d(x, y, equal_var=False, n_jobs=1, n_boot=500)
    result = cohens_d(x, y=None, equal_var=False, n_jobs=1, n_boot=500)
    assert len(result) == 2


def test_perm_test():
    x = np.random.normal(loc=2, size=10)
    y = np.random.normal(loc=2.5, size=15)
    result = perm_test(x, y, stat="tstat", n_perm=500, n_jobs=1, return_dist=True)
    assert len(result) == 3
    y = np.random.normal(loc=2.5, size=10)
    result = perm_test(
        x, y, stat="tstat-paired", n_perm=500, n_jobs=1, return_dist=False
    )
    assert len(result) == 2
    result = perm_test(x, y, stat="cohensd", n_perm=500, n_jobs=1, return_dist=False)
    assert len(result) == 2
    result = perm_test(x, y, stat="pearsonr", n_perm=500, n_jobs=1, return_dist=False)
    assert len(result) == 2
    result = perm_test(x, y, stat="spearmanr", n_perm=500, n_jobs=1, return_dist=False)
    assert len(result) == 2
    result = perm_test(x, y=None, stat="tstat", n_perm=500, n_jobs=1, return_dist=False)


def test_tost():
    np.random.seed(10)
    lower, upper = -0.1, 0.1
    x, y = np.random.normal(0.145, 0.025, 35), np.random.normal(0.16, 0.05, 17)
    result = tost_equivalence(x, y, lower, upper, n_perm=500)
    assert result["In_Equivalence_Range"] is True
    assert result["Means_Are_Different"] is False
    np.random.seed(1)
    x, y = np.random.normal(0.12, 0.025, 35), np.random.normal(0.14, 0.05, 17)
    result = tost_equivalence(x, y, lower, upper)
    assert result["In_Equivalence_Range"] is True
    assert result["Means_Are_Different"] is True
    x, y = np.random.normal(0.32, 0.025, 35), np.random.normal(0.14, 0.05, 17)
    result = tost_equivalence(x, y, lower, upper)
    assert result["In_Equivalence_Range"] is False
    assert result["Means_Are_Different"] is True


def test_boot_func():
    x = np.random.normal(loc=2, size=10)
    y = np.random.normal(loc=2.5, size=15)
    result = boot_func(x, y, func=_mean_diff)
    assert len(result) == 2
    assert len(result[1]) == 2
