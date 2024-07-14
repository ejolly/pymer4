import numpy as np
import pandas as pd
import pytest
import os
from pymer4.models import Lmer, Lm2
from pymer4.utils import get_resource_path
from pymer4.stats import (
    cohens_d,
    perm_test,
    boot_func,
    tost_equivalence,
    _mean_diff,
    lrt,
)


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


def test_lrt(df):
    # read the data and build 3 nexted models
    model = Lmer("DV ~ IV3 + IV2 + (1|Group)", data=df)
    model.fit(summarize=False)
    model_sub = Lmer("DV ~ IV2 + (1|Group)", data=df)
    model_sub.fit(summarize=False)
    model_sub2 = Lmer("DV ~ 1+ (1|Group)", data=df)
    model_sub2.fit(summarize=False)

    # Can only compare Lmer models
    lm_model = Lm2("DV ~ IV3 + IV2", group="Group", data=df)
    with pytest.raises(TypeError):
        lrt([model, lm_model])

    # lrt test with REML (i.e. WITHOUT refitting the models)
    lrt_reml = lrt([model, model_sub, model_sub2], refit=False)
    # the order in which we give the models should not affect the output
    lrt_reml_scrambled = lrt([model_sub, model, model_sub2], refit=False)
    assert lrt_reml.equals(lrt_reml_scrambled) is True
    # now do an lrt with ML (i.e. refittiing the models)
    lrt_ml = lrt([model, model_sub, model_sub2], refit=True)
    # refitting with ML does change the output compared to REML
    assert lrt_reml.equals(lrt_ml) is False

    # pytest crashes with rpy2 in a conda env with python 3.9, so comment out this live test against
    # R; for the moment we will use the tables generated with this code when running interactively

    # rerun the models directly with rpy2 and compare
    # import rpy2.robjects as robjects
    # import rpy2.robjects.packages as rpackages
    # data = robjects.r('read.table(file ="sample_data.csv", header = T, sep=",")')
    # lme4 = rpackages.importr('lme4')
    # rmodel = lme4.lmer("DV ~ IV3 + IV2 + (1|Group)", data=data)
    # rmodel_sub = lme4.lmer("DV ~ IV2  + (1|Group)", data=data)
    # rmodel_sub2 = lme4.lmer("DV ~ 1  + (1|Group)", data=data)
    # r_lrt_reml = lme4.anova_merMod(rmodel, rmodel_sub, rmodel_sub2, refit=False)
    # r_lrt_ml = lme4.anova_merMod(rmodel, rmodel_sub, rmodel_sub2, refit=True)
    # # subset the pymer4 and r table to match (remove columns with model names and sign)
    # lrt_reml_sub = (lrt_reml.drop(lrt_reml.columns[[0, 9]], axis=1))
    # r_lrt_reml_sub = (pd.DataFrame.from_records(r_lrt_reml)).fillna("")
    # # make column names identical
    # r_lrt_reml_sub.columns = lrt_reml_sub.columns
    # # now assert they are equal

    r_lrt_reml_sub = pd.DataFrame.from_records(
        np.rec.array(
            [
                (
                    0,
                    3.0,
                    4832.10121511,
                    4845.10637787,
                    -2413.05060756,
                    4826.10121511,
                    "",
                    "",
                    "",
                ),
                (
                    1,
                    4.0,
                    4587.9706894,
                    4605.3109064,
                    -2289.9853447,
                    4579.9706894,
                    246.13052571481512,
                    1.0,
                    1.8115312634852814e-55,
                ),
                (
                    2,
                    5.0,
                    4584.81022022,
                    4606.48549148,
                    -2287.40511011,
                    4574.81022022,
                    5.160469174164973,
                    1.0,
                    0.02310665453923671,
                ),
            ],
            dtype=[
                ("index", "<i8"),
                ("npar", "<f8"),
                ("AIC", "<f8"),
                ("BIC", "<f8"),
                ("log-likelihood", "<f8"),
                ("deviance", "<f8"),
                ("Chisq", "O"),
                ("Df", "O"),
                ("P-val", "O"),
            ],
        )
    )

    pd.testing.assert_frame_equal(r_lrt_reml_sub, r_lrt_reml_sub, check_dtype=False)
    # # and now repeat for the ml table
    # lrt_ml_sub = (lrt_ml.drop(lrt_ml.columns[[0, 9]], axis=1))
    # r_lrt_ml_sub = (pd.DataFrame.from_records(r_lrt_ml)).fillna("")
    # r_lrt_ml_sub.columns = lrt_ml_sub.columns

    r_lrt_ml_sub = pd.DataFrame.from_records(
        np.rec.array(
            [
                (
                    0,
                    3.0,
                    4836.69171163,
                    4849.69687439,
                    -2415.34585582,
                    4830.69171163,
                    "",
                    "",
                    "",
                ),
                (
                    1,
                    4.0,
                    4586.95522083,
                    4604.29543783,
                    -2289.47761041,
                    4578.95522083,
                    251.7364908061154,
                    1.0,
                    1.08611087137599e-56,
                ),
                (
                    2,
                    5.0,
                    4586.10807729,
                    4607.78334855,
                    -2288.05403865,
                    4576.10807729,
                    2.8471435322935577,
                    1.0,
                    0.09153644160564649,
                ),
            ],
            dtype=[
                ("index", "<i8"),
                ("npar", "<f8"),
                ("AIC", "<f8"),
                ("BIC", "<f8"),
                ("log-likelihood", "<f8"),
                ("deviance", "<f8"),
                ("Chisq", "O"),
                ("Df", "O"),
                ("P-val", "O"),
            ],
        )
    )

    pd.testing.assert_frame_equal(r_lrt_ml_sub, r_lrt_ml_sub, check_dtype=False)

def test_ranef_as_data_frame(df, ranef_as_dataframe_correct_results):
    model = Lmer("IV1 ~ (1|Group)", data=df)
    model.fit(summarize=False)

    pd.testing.assert_frame_equal(
        ranef_as_dataframe_correct_results, model.ranef_df, check_exact=False, rtol=1e-5
    )