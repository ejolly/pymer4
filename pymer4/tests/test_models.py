from __future__ import division
from pymer4.models import Lmer, Lm, Lm2
from pymer4.utils import get_resource_path
import pandas as pd
import numpy as np
from scipy.special import logit
from scipy.stats import ttest_ind
import os
import pytest
import re
from bambi import load_data
import matplotlib.pyplot as plt
import arviz as az

np.random.seed(10)


def test_gaussian_lm2():

    df = pd.read_csv(os.path.join(get_resource_path(), "sample_data.csv"))
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


def test_gaussian_lm():

    df = pd.read_csv(os.path.join(get_resource_path(), "sample_data.csv"))
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


def test_gaussian_lmm():

    df = pd.read_csv(os.path.join(get_resource_path(), "sample_data.csv"))

    # Test against lme4 model in tutorial 1 notebook
    model = Lmer("DV ~ IV2 + (IV2|Group)", data=df)
    assert model.inference_obj is None

    model.fit()

    assert model.coefs.loc["Intercept", "2.5_ci"] > 4
    assert model.coefs.loc["Intercept", "97.5_ci"] < 16
    assert model.coefs.loc["IV2", "2.5_ci"] > 0.5
    assert model.coefs.loc["IV2", "97.5_ci"] < 0.9

    # Test shape works with random items
    model = Lmer("DV ~ IV3 + IV2 + (IV2|Group) + (1|IV3)", data=df)

    # Design matrix is build on model init
    assert model.design_matrix is not None
    assert model.design_matrix_rfx is not None
    assert model.design_matrix.shape == (df.shape[0], 3)  # including intercept
    assert model.design_matrix_rfx.shape == (
        df.shape[0],
        df.Group.nunique() * 2 + df.IV3.nunique(),
    )

    model.fit(summary=False)

    assert model.coefs.shape == (3, 6)
    assert model.ranef.shape == (df.Group.nunique() * 2 + df.IV3.nunique(), 6)
    assert model.ranef_var.shape == (4, 6)

    # Fit check against example on bambi website
    data = load_data("sleepstudy")
    model = Lmer("Reaction ~ 1 + Days + (Days | Subject)", data)
    assert model.model_obj is not None
    assert model.fits is None
    assert model.inference_obj is None

    model.fit(summary=False)

    assert model.fits is not None
    assert model.posterior_predictions.equals(model.fits)
    assert model.prior_predictions is not None

    # Check values against bambi docs which use pymc's sampler
    assert model.coefs.loc["Intercept", "2.5_ci"] > 233
    assert model.coefs.loc["Intercept", "97.5_ci"] < 268
    assert model.coefs.loc["Days", "2.5_ci"] > 6.5
    assert model.coefs.loc["Days", "97.5_ci"] < 15

    # Check fits/predictions
    assert hasattr(model.inference_obj, "posterior")
    assert hasattr(model.inference_obj, "posterior_predictive")
    assert isinstance(model.fits, pd.DataFrame)
    assert model.fits.shape == (model.data.shape[0], 4)

    # Test predict
    # Sample from mean of DV distribution using posterior
    preds = model.predict()
    assert preds.iloc[:, :-1].equals(model.fits)

    # PPS and posterior mean give us the same agg stats on the training data
    preds2 = model.predict(kind="pps")
    assert "pps" in preds2.Kind.unique()
    assert preds2.iloc[:, :-1].equals(preds.iloc[:, :-1])

    # No aggregation of samples gives us xarray Inference Object
    preds_obj = model.predict(summarize=False)
    assert isinstance(preds_obj, az.data.inference_data.InferenceData)
    assert model.inference_obj == preds_obj

    # Trace plots
    # Default trace plot with common and group effect sigmas similar to lmer/summary
    # output
    axs = model.plot_summary()
    assert axs.shape == (4, 2)

    # Just common
    axs = model.plot_summary(params="coef")
    assert axs.shape == (2, 2)

    # Just rfx and variances
    axs = model.plot_summary(params="rfx")
    assert axs.shape == (4, 2)

    # Just DV and variances
    axs = model.plot_summary(params="response")
    assert axs.shape == (2, 2)
    plt.close("all")

    # Summary plots
    axs = model.plot_summary(kind="summary")
    axs = model.plot_summary(kind="forest")
    axs = model.plot_summary(kind="ridge")
    plt.close("all")

    # Posterior distribution plots
    ax = model.plot_summary(kind="posterior")
    # With kwargs supported by az.plot_posterior
    ax = model.plot_summary(kind="posterior", rope=[-1, 1])

    # Prior distribution plots
    ax = model.plot_summary(kind="prior")
    assert ax.shape == (2, 3)
    ax = model.plot_summary(kind="prior", params="rfx")
    assert ax.shape == (2,)
    plt.close("all")

    # Prior/Posteriof predictive plot
    _ = model.plot_summary(kind="ppc", dist="prior")
    _ = model.plot_summary(kind="ppc", dist="posterior")

    # Aliases
    model.plot_priors()
    model.plot_posterior(
        params="coefs",
        rope={"Intercept": [{"rope": (200, 300)}], "Days": [{"rope": (5, 10)}]},
    )

    # assert isinstance(model.fixef, list)
    # assert (model.fixef[0].index.astype(int) == df.Group.unique()).all()
    # assert (model.fixef[1].index.astype(float) == df.IV3.unique()).all()
    # assert model.fixef[0].shape == (47, 3)
    # assert model.fixef[1].shape == (3, 3)

    # assert isinstance(model.ranef, list)
    # assert model.ranef[0].shape == (47, 2)
    # assert model.ranef[1].shape == (3, 1)
    # assert (model.ranef[1].index == ["0.5", "1", "1.5"]).all()

    # assert model.ranef_corr.shape == (1, 3)
    # assert model.ranef_var.shape == (4, 3)

    # assert np.allclose(model.coefs.loc[:, "Estimate"], model.fixef[0].mean(), atol=0.01)

    # Test predict

    # Test simulate


@pytest.mark.skip()
def test_contrasts():
    df = pd.read_csv(os.path.join(get_resource_path(), "gammas.csv")).rename(
        columns={"BOLD signal": "bold"}
    )
    grouped_means = df.groupby("ROI")["bold"].mean()
    model = Lmer("bold ~ ROI + (1|subject)", data=df)

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


@pytest.mark.skip()
def test_logistic_lmm():

    df = pd.read_csv(os.path.join(get_resource_path(), "sample_data.csv"))
    model = Lmer("DV_l ~ IV1+ (IV1|Group)", data=df, family="binomial")
    model.fit(summarize=False)

    assert model.coefs.shape == (2, 13)
    estimates = np.array([-0.16098421, 0.00296261])
    assert np.allclose(model.coefs["Estimate"], estimates, atol=0.001)

    assert isinstance(model.fixef, pd.core.frame.DataFrame)
    assert model.fixef.shape == (47, 2)

    assert isinstance(model.ranef, pd.core.frame.DataFrame)
    assert model.ranef.shape == (47, 2)

    assert np.allclose(model.coefs.loc[:, "Estimate"], model.fixef.mean(), atol=0.01)

    # Test prediction
    assert np.allclose(
        model.predict(model.data, use_rfx=True, verify_predictions=False),
        model.data.fits,
    )
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


@pytest.mark.skip()
def test_poisson_lmm():
    np.random.seed(1)
    df = pd.read_csv(os.path.join(get_resource_path(), "sample_data.csv"))
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


@pytest.mark.skip()
def test_gamma_lmm():

    np.random.seed(1)
    df = pd.read_csv(os.path.join(get_resource_path(), "sample_data.csv"))
    df["DV_g"] = np.random.uniform(1, 2, size=df.shape[0])
    m = Lmer("DV_g ~ IV3 + (1|Group)", data=df, family="gamma")
    m.fit(summarize=False)
    assert m.family == "gamma"
    assert m.coefs.shape == (2, 7)

    # Test RFX only; these work but the optimizer in R typically crashes if the model is especially bad fit so commenting out until a better dataset is acquired

    # model = Lmer("DV_g ~ 0 + (IV1|Group)", data=df, family="gamma")
    # model.fit(summarize=False)
    # assert model.fixef.shape == (47, 2)

    # model = Lmer("DV_g ~ 0 + (IV1|Group) + (1|IV3)", data=df, family="gamma")
    # model.fit(summarize=False)
    # assert isinstance(model.fixef, list)
    # assert model.fixef[0].shape == (47, 2)
    # assert model.fixef[1].shape == (3, 2)


@pytest.mark.skip()
def test_inverse_gaussian_lmm():

    np.random.seed(1)
    df = pd.read_csv(os.path.join(get_resource_path(), "sample_data.csv"))
    df["DV_g"] = np.random.uniform(1, 2, size=df.shape[0])
    m = Lmer("DV_g ~ IV3 + (1|Group)", data=df, family="inverse_gaussian")
    m.fit(summarize=False)
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


# all or prune to suit
tests_ = [eval(v) for v in locals() if re.match(r"^test_", str(v))]
tests_ = [
    test_gaussian_lm2,
    test_gaussian_lm,
    test_gaussian_lmm,
    test_logistic_lmm,
    test_poisson_lmm,
    test_gamma_lmm,
    test_inverse_gaussian_lmm,
]


@pytest.mark.skip()
@pytest.mark.parametrize("model", tests_)
def test_Pool(model):
    from multiprocessing import get_context

    # squeeze model functions through Pool pickling
    print("Pool", model.__name__)
    with get_context("spawn").Pool(1) as pool:
        _ = pool.apply(model, [])
