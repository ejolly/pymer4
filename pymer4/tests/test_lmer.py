import numpy as np
from pymer4.models import lmer, lm, compare
from great_tables import GT
from polars import col
from pymer4 import load_dataset
from scipy.stats import pearsonr
import pytest


def test_lmm_basics(sleep):
    from pymer4 import load_dataset
    from pymer4.models import lmer

    sleep = load_dataset("sleep")

    num_rfx = sleep.select("Subject").n_unique()

    lmm = lmer("Reaction ~ 1 + (1 | Subject)", data=sleep)
    lmm.fit()

    # lmms store Blups and deviances as a polars dataframe with rfx level as the first column
    assert lmm.fixef.shape == (num_rfx, 2)
    assert lmm.ranef.shape == (num_rfx, 2)

    # Intercept & Residual
    assert lmm.ranef_var.shape == (2, 5)
    # without bootstrapping CIs are always null
    assert lmm.ranef_var["conf_low"].is_null().sum() == 2
    assert lmm.ranef_var["conf_high"].is_null().sum() == 2
    # We augment with fits
    assert lmm.data.width == 14

    # Now with slopes things get wider
    lmm = lmer("Reaction ~ 1 + (Days | Subject)", data=sleep)
    lmm.fit()
    # lmms store Blups and deviances as a polars dataframe with rfx level as the first column
    assert lmm.fixef.shape == (num_rfx, 3)
    assert lmm.ranef.shape == (num_rfx, 3)
    # Intercept, Slope, Correlation & Residual
    assert lmm.ranef_var.shape == (4, 5)

    # Multiple RFX creates dict of dataframes
    num_days = sleep.select("Days").n_unique()
    lmm = lmer("Reaction ~ Days + (1 | Subject) + (1 | Days)", data=sleep)
    lmm.fit()

    # now we have a list of 2 dataframes
    assert len(lmm.fixef.keys()) == 2
    assert len(lmm.ranef.keys()) == 2
    # subID, intercept (varies), slope (fixed)
    assert lmm.fixef["Subject"].shape == (num_rfx, 3)
    # dayID, intercept (varies), slope (fixed)
    assert lmm.fixef["Days"].shape == (num_days, 3)
    # subID, intercept deviance
    assert lmm.ranef["Subject"].shape == (num_rfx, 2)
    # dayID, intercept deviance
    assert lmm.ranef["Days"].shape == (num_days, 2)
    # 2 intercepts and residual
    assert lmm.ranef_var.shape == (3, 5)

    assert isinstance(lmm.summary(), GT)

    preds = lmm.predict(lmm.data)
    assert np.allclose(preds, lmm.data["fitted"].to_numpy().squeeze())


def test_lmm_conf_methods(sleep):
    # Bootstrapped CIs
    lmm = lmer("Reaction ~ Days + (Days | Subject)", data=sleep)
    lmm.fit()
    # Any non-default CIs will also be calculated for random effects
    assert lmm.ranef_var.shape == (4, 5)
    assert lmm.ranef_var["conf_low"].is_null().sum() == 4
    assert lmm.ranef_var["conf_high"].is_null().sum() == 4

    # Lmer supports: satterthwaite, wald, and boot
    standard_ci = lmm.result_fit.select("conf_low", "conf_high").to_numpy()
    lmm.fit(conf_method="wald")
    wald_ci = lmm.result_fit.select("conf_low", "conf_high").to_numpy()

    lmm.fit(conf_method="boot", nboot=100)
    assert lmm.result_boots.shape == (100, 6)
    assert lmm.ranef_var["conf_low"].is_null().sum() == 0
    assert lmm.ranef_var["conf_high"].is_null().sum() == 0
    # This is equivalent to using the confint function directly
    boot_ci = lmm.result_fit.select("conf_low", "conf_high").to_numpy()

    assert not np.allclose(standard_ci, wald_ci)
    assert not np.allclose(standard_ci, boot_ci)
    assert not np.allclose(wald_ci, boot_ci)

    # Make sure we can handle multiple RFX
    lmm = lmer("Reaction ~ Days + (1 | Subject) + (1 | Days)", data=sleep)
    lmm.fit(conf_method="boot", nboot=100)
    assert lmm.result_boots.shape == (100, 5)
    assert isinstance(lmm.fixef, dict)
    assert isinstance(lmm.ranef, dict)
    assert len(lmm.fixef.keys()) == 2
    assert len(lmm.ranef.keys()) == 2


def test_lmm_logging(sleep, capsys):
    # By default we log and display all R messages
    deg_lmm = lmer("Reaction ~ Days + (1 | Subject) + (1 | Days)", data=sleep)

    # Model is degenerate so we should see a singular fit warning
    deg_lmm.fit()
    captured = capsys.readouterr()
    assert captured.out and "singular" in captured.out

    # Which we can suppress, but still store
    deg_lmm.fit(verbose=False)
    captured = capsys.readouterr()
    assert not captured.out

    # And show on demand
    deg_lmm.show_logs()
    captured = capsys.readouterr()
    assert captured.out and "singular" in captured.out

    # Also applies to non-warnings like bootstrapping messages
    good_lmm = lmer("Reaction ~ Days + (1 | Subject)", data=sleep)
    good_lmm.fit()
    good_lmm.show_logs()

    # Good model no warnings
    assert not capsys.readouterr().out


def test_categorical_lmm(penguins):
    m = lmer("bill_length_mm ~ sex + island + (1 | species)", data=penguins)
    m.set_factors(["sex", "island"])
    m.anova()
    assert isinstance(m.summary_anova(), GT)
    assert m.result_anova.shape == (2, 5)

    # Test emmeans
    assert m.emmeans("sex").shape == (3, 6)
    assert m.emmeans("island").shape == (3, 6)
    assert m.emmeans("sex", by="island").shape == (9, 7)

    # Test contrasts with multiple comparisons correction
    sidak_p = (
        m.emmeans("sex", by="island", contrasts="pairwise").select("p_value").to_numpy()
    )
    tukey_p = (
        m.emmeans("sex", by="island", contrasts="pairwise", p_adjust="tukey")
        .select("p_value")
        .to_numpy()
    )
    bonf_p = (
        m.emmeans("sex", by="island", contrasts="pairwise", p_adjust="bonf")
        .select("p_value")
        .to_numpy()
    )
    unc_p = (
        m.emmeans("sex", by="island", contrasts="pairwise", p_adjust="none")
        .select("p_value")
        .to_numpy()
    )
    assert len(sidak_p) == 9
    assert unc_p.sum() < tukey_p.sum() < sidak_p.sum() < bonf_p.sum()

    # 2 factors and 1 continuous
    df = load_dataset("sample_data")
    model = lmer("DV ~ IV1*IV3*DV_l + (IV1|Group)", data=df)
    model.set_factors({"IV3": ["0.5", "1.0", "1.5"], "DV_l": ["0", "1"]})
    model.set_contrasts({"IV3": "contr.sum", "DV_l": "contr.sum"})
    model.set_transforms({"IV1": "center"})
    model.fit()

    # 3 means
    out = model.emmeans("IV3", p_adjust="sidak")
    assert out.shape == (3, 6)

    # 2 means
    out = model.emmeans("DV_l", p_adjust="sidak")
    assert out.shape == (2, 6)

    # 1 slope, which matches sumary output
    # since we centered and used sum coding
    out = model.emmeans("IV1")
    assert out.shape == (1, 5)  # drop var name?
    trend = out.select("IV1_trend").item()
    beta = model.result_fit.filter(col("term") == "IV1").select("estimate").item()
    assert np.allclose(trend, beta)

    # Slope diffs of IV1 across 2 levels of DV_l
    out = model.emmeans("IV1", "DV_l")
    assert out.shape == (2, 6)
    out = model.emmeans("IV1", "DV_l", contrasts="pairwise")
    assert out.shape == (1, 8)

    # 2 continuous and 1 factor
    model = lmer("DV ~ IV1*IV3*DV_l + (IV1|Group)", data=df)
    model.set_transforms({"IV1": "center", "IV3": "center"})
    model.set_factors({"DV_l": ["0", "1"]})
    model.set_contrasts({"DV_l": "contr.sum"})
    model.fit()

    # Difference between slopes of IV3 at both levels of DV_l
    slope_diffs = (
        model.emmeans("IV3", by="DV_l", contrasts="pairwise").select("estimate").item()
    )
    # Equivalent to the model beta, which is half the diff because of sum-coding
    interaction_beta = (
        model.result_fit.filter(col("term") == "IV3:DV_l1").select("estimate").item()
        * 2
    )
    assert np.allclose(slope_diffs, interaction_beta)
    # Stratify continuous by other continuous at it's mean which is what
    # the marginal slope is since we mean-centered
    model.emmeans("IV1", by="IV3", at={"IV3": model.data["IV3"].mean()}).drop(
        "IV3"
    ).equals(model.emmeans("IV1"))


def test_model_comparison(sleep):
    sleep = load_dataset("sleep")
    intercept_only = lmer("Reaction ~ Days + (1 | Subject)", data=sleep)
    slope_only = lmer("Reaction ~ Days + (0 + Days | Subject)", data=sleep)

    assert not intercept_only.fitted
    assert not slope_only.fitted

    # Unfit models will be auto-fit
    out = compare(intercept_only, slope_only)
    assert isinstance(out, GT)
    assert intercept_only.fitted
    assert slope_only.fitted

    # Multiple calls will not refit
    out = compare(intercept_only, slope_only, as_dataframe=True)

    # Works if models are fit manually before-hand
    intercept_only = lmer("Reaction ~ Days + (1 | Subject)", data=sleep)
    slope_only = lmer("Reaction ~ Days + (0 + Days | Subject)", data=sleep)
    intercept_only.fit()
    slope_only.fit()
    out2 = compare(intercept_only, slope_only, as_dataframe=True)
    assert out.equals(out2)

    # Or if some models are fit and others arent
    intercept_only = lmer("Reaction ~ Days + (1 | Subject)", data=sleep)
    intercept_only.fit()
    slope_only = lmer("Reaction ~ Days + (0 + Days | Subject)", data=sleep)
    out2 = compare(intercept_only, slope_only, as_dataframe=True)

    # We can't compute p-vals if models have same number of params
    # so we replace with nans
    pvals = out2[:, -1].to_numpy()
    assert all(np.isnan(pvals))

    # Compare 3 models
    both = lmer("Reaction ~ Days + (Days | Subject)", data=sleep)
    out3 = compare(intercept_only, slope_only, both, as_dataframe=True)
    assert out3.shape == (3, 8)

    # Compare to OLS via LRT
    ols = lm("Reaction ~ Days", data=sleep)
    assert "F" not in compare(both, ols, as_dataframe=True).columns
    # We can' use LRT with any non lmm models
    assert "F" not in compare(both, ols, as_dataframe=True, test="F").columns

    # But we can when comparing lm/glm models
    ols2 = lm("Reaction ~ 1", data=sleep)
    assert "F" in compare(ols, ols2, as_dataframe=True).columns
    assert "F" not in compare(ols, ols2, as_dataframe=True, test="LRT").columns


def test_predict_simulate(sleep):
    model = lmer("Reaction ~ Days + (Days | Subject)", data=sleep)
    model.fit()
    fits = model.data.select("fitted").to_numpy().squeeze()

    # By default model will condition on random effects
    preds = model.predict(model.data)
    assert np.allclose(fits, preds)
    # But we can tell it not to
    preds_without_rfx = model.predict(model.data, use_rfx=False)
    assert not np.allclose(preds, preds_without_rfx)

    assert pearsonr(fits, preds).statistic > pearsonr(fits, preds_without_rfx).statistic

    # By default we incorporate rfx just like predict method
    # And respect random seeds
    sims = model.simulate(1, seed=1)
    sims2 = model.simulate(1, seed=1)
    assert sims.equals(sims2)

    # But we can ignore them
    sims_no_rfx = model.simulate(1, seed=1, use_rfx=False).to_numpy().squeeze()

    sims = sims.to_numpy().squeeze()
    assert not np.allclose(sims, sims_no_rfx)

    data = model.data.select("Reaction").to_numpy().squeeze()
    assert pearsonr(data, sims).statistic > pearsonr(data, sims_no_rfx).statistic


@pytest.mark.skip(reason="lmer control options are WIP")
def test_control_options(sleep):
    from pymer4.tidystats.lmerTest import lmer_control

    model = lmer("Reaction ~ Days + (Days | Subject)", data=sleep)
    control = lmer_control(
        optimizer="Nelder_Mead", optCtrl={"FtolAbs": 1e-8, "XtolRel": 1e-8}
    )
    model.fit(control=control)
