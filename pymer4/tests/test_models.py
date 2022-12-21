from pymer4.models import Lmer, Lm, Lm2
from pymer4.bridge import pandas2R, R2pandas
import pandas as pd
import numpy as np
from scipy.special import logit
from scipy.stats import ttest_ind
import os
import pytest
from rpy2.rinterface_lib.embedded import RRuntimeError
from rpy2.robjects.packages import importr
import rpy2.robjects as ro

stats = importr("stats")
base = importr("base")

np.random.seed(10)

os.environ[
    "KMP_DUPLICATE_LIB_OK"
] = "True"  # Recent versions of rpy2 sometimes cause the python kernel to die when running R code; this handles that


def test_gaussian_lm2(df):

    model = Lm2("DV ~ IV3 + IV2", group="Group", data=df)
    model.fit(summarize=False)
    assert model.coefs.shape == (3, 8)
    estimates = np.array([16.11554138, -1.38425772, 0.59547697])
    assert np.allclose(model.coefs["Estimate"], estimates, atol=0.001)
    assert model.fixef.shape == (47, 3)

    assert model.rsquared is not None
    assert model.rsquared_adj is not None
    assert len(model.rsquared_per_group) == 47
    assert len(model.rsquared_adj_per_group) == 47
    assert len(model.fits) == model.data.shape[0]
    assert len(model.residuals) == model.data.shape[0]
    assert "fits" in model.data.columns
    assert "residuals" in model.data.columns

    # Test bootstrapping and permutation tests
    model.fit(permute=500, conf_int="boot", n_boot=500, summarize=False)
    assert model.ci_type == "boot (500)"
    assert model.sig_type == "permutation (500)"


def test_gaussian_lm(df):

    model = Lm("DV ~ IV1 + IV3", data=df)
    model.fit(summarize=False)

    assert model.coefs.shape == (3, 8)
    estimates = np.array([42.24840439, 0.24114414, -3.34057784])
    assert np.allclose(model.coefs["Estimate"], estimates, atol=0.001)

    # Test robust SE against statsmodels
    standard_se = np.array([6.83783939, 0.30393886, 3.70656475])
    assert np.allclose(model.coefs["SE"], standard_se, atol=0.001)

    hc0_se = np.array([7.16661817, 0.31713064, 3.81918182])
    model.fit(robust="hc0", summarize=False)
    assert np.allclose(model.coefs["SE"], hc0_se, atol=0.001)

    hc1_se = np.array([7.1857547, 0.31797745, 3.82937992])
    # hc1 is the default
    model.fit(robust=True, summarize=False)
    assert np.allclose(model.coefs["SE"], hc1_se, atol=0.001)

    hc2_se = np.array([7.185755, 0.317977, 3.829380])
    model.fit(robust="hc1", summarize=False)
    assert np.allclose(model.coefs["SE"], hc2_se, atol=0.001)

    hc3_se = np.array([7.22466699, 0.31971942, 3.84863701])
    model.fit(robust="hc3", summarize=False)
    assert np.allclose(model.coefs["SE"], hc3_se, atol=0.001)

    hac_lag1_se = np.array([8.20858448, 0.39184764, 3.60205873])
    model.fit(robust="hac", summarize=False)
    assert np.allclose(model.coefs["SE"], hac_lag1_se, atol=0.001)

    # Test bootstrapping
    model.fit(summarize=False, conf_int="boot")
    assert model.ci_type == "boot (500)"

    # Test permutation
    model.fit(summarize=False, permute=500)
    assert model.sig_type == "permutation (500)"

    # Test WLS
    df_two_groups = df.query("IV3 in [0.5, 1.0]").reset_index(drop=True)
    x = df_two_groups.query("IV3 == 0.5").DV.values
    y = df_two_groups.query("IV3 == 1.0").DV.values

    # Fit new a model using a categorical predictor with unequal variances (WLS)
    model = Lm("DV ~ IV3", data=df_two_groups)
    model.fit(summarize=False, weights="IV3")
    assert model.estimator == "WLS"

    # Make sure welch's t-test lines up with scipy
    wls = np.abs(model.coefs.loc["IV3", ["T-stat", "P-val"]].values)
    scit = np.abs(ttest_ind(x, y, equal_var=False))
    assert all([np.allclose(a, b) for a, b in zip(wls, scit)])


def test_gaussian_lmm(df):

    model = Lmer("DV ~ IV3 + IV2 + (IV2|Group) + (1|IV3)", data=df)
    opt_opts = "optimizer='Nelder_Mead', optCtrl = list(FtolAbs=1e-8, XtolRel=1e-8)"
    model.fit(summarize=False, control=opt_opts)

    assert model.coefs.shape == (3, 8)
    estimates = np.array([12.04334602, -1.52947016, 0.67768509])
    assert np.allclose(model.coefs["Estimate"], estimates, atol=0.001)

    assert isinstance(model.fixef, list)
    assert (model.fixef[0].index.astype(int) == df.Group.unique()).all()
    assert (model.fixef[1].index.astype(float) == df.IV3.unique()).all()
    assert model.fixef[0].shape == (47, 3)
    assert model.fixef[1].shape == (3, 3)

    assert isinstance(model.ranef, list)
    assert model.ranef[0].shape == (47, 2)
    assert model.ranef[1].shape == (3, 1)
    assert (model.ranef[1].index == ["0.5", "1", "1.5"]).all()

    assert model.ranef_corr.shape == (1, 3)
    assert model.ranef_var.shape == (4, 3)

    assert np.allclose(model.coefs.loc[:, "Estimate"], model.fixef[0].mean(), atol=0.01)

    # Test predict
    # Little hairy to we test a few different cases. If a dataframe with non-matching
    # column names is passed in, but we only used fixed-effects to make predictions,
    # then R will not complain and will return population level predictions given the
    # model's original data. This is undesirable behavior, so pymer tries to naively
    # check column names in Python first and checks the predictions against the
    # originally fitted values second. This is works fine except when there are
    # categorical predictors which get expanded out to a design matrix internally in R.
    # Unfortunately we can't easily pre-expand this to check against the column names of
    # the model matrix.

    # Test circular prediction which should raise error
    with pytest.raises(ValueError):
        assert np.allclose(model.predict(model.data), model.data.fits)

    # Same thing, but skip the prediction verification; no error
    assert np.allclose(
        model.predict(model.data, verify_predictions=False), model.data.fits
    )

    # Test on data that has no matching columns;
    X = pd.DataFrame(np.random.randn(model.data.shape[0], model.data.shape[1] - 1))

    # Should raise error no matching columns, caught by checks in Python
    with pytest.raises(ValueError):
        model.predict(X)

    # If user skips Python checks, then pymer raises an error if the predictions match
    # the population predictions from the model's original data (which is what predict()
    # in R will do by default).
    with pytest.raises(ValueError):
        model.predict(X, skip_data_checks=True, use_rfx=False)

    # If the user skips check, but tries to predict with rfx then R will complain so we
    # can check for an exception raised from R rather than pymer
    with pytest.raises((RRuntimeError, ValueError)):
        model.predict(X, skip_data_checks=True, use_rfx=True)

    # Finally a user can turn off every kind of check in which case we expect circular predictions
    pop_preds = model.predict(model.data, use_rfx=False, verify_predictions=False)
    assert np.allclose(
        pop_preds,
        model.predict(
            X,
            use_rfx=False,
            skip_data_checks=True,
            verify_predictions=False,
        ),
    )

    # Test prediction with categorical variables
    df["DV_ll"] = df.DV_l.apply(lambda x: "yes" if x == 1 else "no")
    m = Lmer("DV ~ IV3 + DV_ll + (IV2|Group) + (1|IV3)", data=df)
    m.fit(summarize=False)

    # Should fail because column name checks don't understand expanding levels of
    # categorical variable into new design matrix columns, as the checks are in Python
    # but R handles the design matrix conversion
    with pytest.raises(ValueError):
        m.predict(m.data, verify_predictions=False)

    # Should fail because of circular predictions
    with pytest.raises(ValueError):
        m.predict(m.data, skip_data_checks=True)

    # Test simulate
    out = model.simulate(2)
    assert isinstance(out, pd.DataFrame)
    assert out.shape == (model.data.shape[0], 2)

    out = model.simulate(2, use_rfx=True)
    assert isinstance(out, pd.DataFrame)
    assert out.shape == (model.data.shape[0], 2)

    # Test confint
    # Wald confidence interval
    wald_confint = model.confint()

    assert isinstance(wald_confint, pd.DataFrame)
    assert wald_confint.shape == (8, 2)
    # there should be no estimates for the random effects
    assert wald_confint["2.5 %"].isna().sum() == 5
    # bootstrapped confidence intervals
    boot_confint = model.confint(method="boot", nsim=10)
    assert isinstance(boot_confint, pd.DataFrame)
    assert boot_confint.shape == (8, 2)
    # ci for random effects should be estimates by bootstrapping
    assert boot_confint["2.5 %"].isna().sum() == 0

    # Smoketest for old_optimizer
    model.fit(summarize=False, old_optimizer=True)

    # test fixef code for 1 fixed effect
    model = Lmer("DV ~ IV3 + IV2 + (IV2|Group)", data=df)
    model.fit(summarize=False, control=opt_opts)

    assert (model.fixef.index.astype(int) == df.Group.unique()).all()
    assert model.fixef.shape == (47, 3)
    assert np.allclose(model.coefs.loc[:, "Estimate"], model.fixef.mean(), atol=0.01)

    # test fixef code for 0 fixed effects
    model = Lmer("DV ~ (IV2|Group) + (1|IV3)", data=df)
    model.fit(summarize=False, control=opt_opts)

    assert isinstance(model.fixef, list)
    assert (model.fixef[0].index.astype(int) == df.Group.unique()).all()
    assert (model.fixef[1].index.astype(float) == df.IV3.unique()).all()
    assert model.fixef[0].shape == (47, 2)
    assert model.fixef[1].shape == (3, 2)


def test_contrasts(gammas):

    grouped_means = gammas.groupby("ROI")["bold"].mean()
    model = Lmer("bold ~ ROI + (1|subject)", data=gammas)

    custom_contrast = grouped_means["AG"] - np.mean(
        [grouped_means["IPS"], grouped_means["V1"]]
    )
    grand_mean = grouped_means.mean()

    con1 = grouped_means["V1"] - grouped_means["IPS"]
    con2 = grouped_means["AG"] - grouped_means["IPS"]
    intercept = grouped_means["IPS"]

    # Treatment contrasts with non-alphabetic order
    model.fit(factors={"ROI": ["IPS", "V1", "AG"]}, summarize=False)

    assert np.allclose(model.coefs.loc["(Intercept)", "Estimate"], intercept)
    assert np.allclose(model.coefs.iloc[1, 0], con1)
    assert np.allclose(model.coefs.iloc[2, 0], con2)

    # Polynomial contrasts
    model.fit(factors={"ROI": ["IPS", "V1", "AG"]}, ordered=True, summarize=False)

    assert np.allclose(model.coefs.loc["(Intercept)", "Estimate"], grand_mean)
    assert np.allclose(model.coefs.iloc[1, 0], 0.870744)  # From R
    assert np.allclose(model.coefs.iloc[2, 0], 0.609262)  # From R

    # Custom contrasts
    model.fit(factors={"ROI": {"AG": 1, "IPS": -0.5, "V1": -0.5}}, summarize=False)

    assert np.allclose(model.coefs.loc["(Intercept)", "Estimate"], grand_mean)
    assert np.allclose(model.coefs.iloc[1, 0], custom_contrast)


def test_post_hoc(df):
    np.random.seed(1)
    model = Lmer("DV ~ IV1*IV3*DV_l + (IV1|Group)", data=df, family="gaussian")
    model.fit(
        factors={"IV3": ["0.5", "1.0", "1.5"], "DV_l": ["0", "1"]}, summarize=False
    )

    marginal, contrasts = model.post_hoc(marginal_vars="IV3", p_adjust="dunnet")

    assert marginal.shape[0] == 3
    assert contrasts.shape[0] == 3

    marginal, contrasts = model.post_hoc(marginal_vars=["IV3", "DV_l"])
    assert marginal.shape[0] == 6
    assert contrasts.shape[0] == 15


def test_logistic_lm(df):
    model = Lm("DV_l ~ IV1", data=df, family="binomial")
    model.fit(summarize=False)

    # Basic checks
    assert model.coefs.shape == (2, 13)
    assert "OR" in model.coefs.columns and "Prob" in model.coefs.columns

    # Should be able to compare to R glm()
    rdf = pandas2R(df)
    r_model = stats.glm("DV_l ~ IV1", family="binomial", data=rdf)
    get_summary = ro.r(
        """
        function(m){
        out <- data.frame(unclass(summary(m))$coefficients)
        out
        }
        """
    )
    summary = R2pandas(get_summary(r_model))

    # Compare output
    for rcol, pcol in zip(
        ["Estimate", "Std..Error", "z.value", "Pr...z.."],
        ["Estimate", "SE", "Z-stat", "P-val"],
    ):
        assert np.allclose(model.coefs[pcol], summary[rcol])

    # Test prediction
    assert np.allclose(
        model.predict(model.data),
        model.data.fit_probs,
    )
    assert np.allclose(
        model.predict(model.data, pred_type="link"),
        model.data.fits,
    )


def test_logistic_lmm(df):

    model = Lmer("DV_l ~ IV1+ (IV1|Group)", data=df, family="binomial")
    model.fit(summarize=True)
    assert np.allclose(
        model.coefs.loc["(Intercept)", "Prob"], model.data.DV_l.mean(), atol=0.01
    )

    assert model.coefs.shape == (2, 13)
    estimates = np.array([-0.16098421, 0.00296261])
    assert np.allclose(model.coefs["Estimate"], estimates, atol=0.001)

    assert isinstance(model.fixef, pd.core.frame.DataFrame)
    assert model.fixef.shape == (47, 2)

    assert isinstance(model.ranef, pd.core.frame.DataFrame)
    assert model.ranef.shape == (47, 2)

    assert np.allclose(model.coefs.loc[:, "Estimate"], model.fixef.mean(), atol=0.01)

    # Test prediction
    # By default we give back probs
    assert np.allclose(
        model.predict(model.data, use_rfx=True, verify_predictions=False),
        model.data.fits,
    )
    # But can convert to logits like .decision_function in sklearn
    assert np.allclose(
        model.predict(model.data, use_rfx=True, pred_type="link"),
        logit(model.data.fits),
    )

    # Test RFX only
    model = Lmer("DV_l ~ 0 + (IV1|Group)", data=df, family="binomial")
    model.fit(summarize=False)
    assert model.fixef.shape == (47, 2)

    model = Lmer("DV_l ~ 0 + (IV1|Group) + (1|IV3)", data=df, family="binomial")
    model.fit(summarize=False)
    assert isinstance(model.fixef, list)
    assert model.fixef[0].shape == (47, 2)
    assert model.fixef[1].shape == (3, 2)


def test_anova(df):

    np.random.seed(1)
    df["DV_l2"] = np.random.randint(0, 4, df.shape[0])
    model = Lmer("DV ~ IV3*DV_l2 + (IV3|Group)", data=df)
    model.fit(summarize=False)
    out = model.anova()
    assert all(out.index == ["IV3", "DV_l2", "IV3:DV_l2"])
    assert out.shape == (3, 7)
    out = model.anova(force_orthogonal=True)
    assert all(out.index == ["IV3", "DV_l2", "IV3:DV_l2"])
    assert out.shape == (3, 7)


def test_poisson_lmm(df):
    np.random.seed(1)
    df["DV_int"] = np.random.randint(1, 10, df.shape[0])
    m = Lmer("DV_int ~ IV3 + (1|Group)", data=df, family="poisson")
    m.fit(summarize=False)
    assert m.family == "poisson"
    assert m.coefs.shape == (2, 7)
    assert "Z-stat" in m.coefs.columns

    # Test RFX only
    model = Lmer("DV_int ~ 0 + (IV1|Group)", data=df, family="poisson")
    model.fit(summarize=False)
    assert model.fixef.shape == (47, 2)

    model = Lmer("DV_int ~ 0 + (IV1|Group) + (1|IV3)", data=df, family="poisson")
    model.fit(summarize=False)
    assert isinstance(model.fixef, list)
    assert model.fixef[0].shape == (47, 2)
    assert model.fixef[1].shape == (3, 2)


def test_gamma_lmm(df):

    np.random.seed(1)
    df["DV_g"] = np.random.uniform(1, 2, size=df.shape[0])
    m = Lmer("DV_g ~ IV3 + (1|Group)", data=df, family="gamma")
    m.fit(summarize=False)
    assert m.family == "gamma"
    assert m.coefs.shape == (2, 7)

    # Test RFX only; these work but the optimizer in R typically crashes if the model is especially bad fit so commenting out until a better dataset is acquired

    # model = Lmer("DV_g ~ 0 + (IV1|Group)", data=df, family="gamma")
    # model.fit(summarize=False)
    # assert model.fixef.shape == (47, 2)

    model = Lmer("DV_g ~ 0 + (IV1|Group) + (1|IV3)", data=df, family="gamma")
    model.fit(summarize=False)
    assert isinstance(model.fixef, list)
    assert model.fixef[0].shape == (47, 2)
    assert model.fixef[1].shape == (3, 2)


def test_inverse_gaussian_lmm(df):

    np.random.seed(1)
    df["DV_g"] = np.random.uniform(1, 2, size=df.shape[0])
    m = Lmer("DV_g ~ IV3 + (1|Group)", data=df, family="inverse_gaussian")
    m.fit(summarize=False, old_optimizer=True)
    assert m.family == "inverse_gaussian"
    assert m.coefs.shape == (2, 7)

    # Test RFX only; these work but the optimizer in R typically crashes if the model is especially bad fit so commenting out until a better dataset is acquired

    # model = Lmer("DV_g ~ 0 + (IV1|Group)", data=df, family="inverse_gaussian")
    # model.fit(summarize=False)
    # assert model.fixef.shape == (47, 2)

    # model = Lmer("DV_g ~ 0 + (IV1|Group) + (1|IV3)", data=df, family="inverse_gaussian")
    # model.fit(summarize=False)
    # assert isinstance(model.fixef, list)
    # assert model.fixef[0].shape == (47, 2)
    # assert model.fixef[1].shape == (3, 2)


def test_lmer_opt_passing(df):

    model = Lmer("DV ~ IV2 + (IV2|Group)", data=df)
    opt_opts = "optCtrl = list(ftol_abs=1e-8, xtol_abs=1e-8)"
    model.fit(summarize=False, control=opt_opts)
    estimates = np.array([10.301072, 0.682124])
    assert np.allclose(model.coefs["Estimate"], estimates, atol=0.001)
    # On some hardware the optimizer will still fail to converge
    # assert len(model.warnings) == 0

    model = Lmer("DV ~ IV2 + (IV2|Group)", data=df)
    opt_opts = "optCtrl = list(ftol_abs=1e-4, xtol_abs=1e-4)"
    model.fit(summarize=False, control=opt_opts)
    assert len(model.warnings) >= 1


def test_glmer_opt_passing(df):

    np.random.seed(1)
    df["DV_int"] = np.random.randint(1, 10, df.shape[0])
    m = Lmer("DV_int ~ IV3 + (1|Group)", data=df, family="poisson")
    m.fit(
        summarize=False, control="optCtrl = list(FtolAbs=1e-1, FtolRel=1e-1, maxfun=10)"
    )
    assert len(m.warnings) >= 1


tests_ = [
    test_gaussian_lm2,
    test_gaussian_lm,
    test_gaussian_lmm,
    test_post_hoc,
    test_logistic_lmm,
    test_anova,
    test_poisson_lmm,
    # test_gamma_lmm,
    test_inverse_gaussian_lmm,
    test_lmer_opt_passing,
    test_glmer_opt_passing,
]


@pytest.mark.parametrize("model", tests_)
def test_Pool(model, df):
    from multiprocessing import get_context

    # squeeze model functions through Pool pickling
    print("Pool", model.__name__)
    with get_context("spawn").Pool(1) as pool:
        _ = pool.apply(model, [df])
