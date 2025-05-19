import numpy as np
from pymer4.models import glmer
from great_tables import GT
import polars.selectors as cs


def test_glmer_basics(titanic):
    """Test basic glmer functionality with a binomial model."""
    from pymer4.models import glmer
    from pymer4 import load_dataset

    titanic = load_dataset("titanic")
    model = glmer("survived ~ fare + (1 | pclass)", data=titanic, family="binomial")
    model.fit()
    assert model.result_fit.shape == (2, 8)

    # Check that we have the right attributes
    assert model.family == "binomial"
    assert model.ranef is not None
    assert model.ranef_var is not None
    assert model.result_fit is not None

    # Basic assertions for a binomial model
    num_rfx = titanic.select("pclass").n_unique()
    assert model.ranef.shape == (num_rfx, 2)  # Intercept + by group

    # Intercept var
    assert model.ranef_var.shape == (1, 5)

    # Test different summary displays
    assert isinstance(model.summary(), GT)

    # Predictions are on response scale by default, i.e. probabilities for logistic lmm
    preds = model.predict(model.data)
    assert 0 <= preds.max() <= 1
    assert 0 <= preds.min() <= 1
    assert np.allclose(preds, model.data["fitted"].to_numpy().squeeze())

    preds = model.predict(model.data, type_predict="link")
    assert preds.min() < 0 and preds.max() > 1
    assert not np.allclose(preds, model.data["fitted"].to_numpy().squeeze())


def test_glmer_categorical(titanic):
    """Test glmer with categorical predictors, ANOVA, and emmeans."""

    model = glmer("survived ~ sex + (1 | pclass)", data=titanic, family="binomial")
    model.set_factors({"sex": ["male", "female"]})
    model.fit()

    # ANOVA
    model.anova()
    assert isinstance(model.summary_anova(), GT)

    # Test emmeans
    # type='response' is the default
    # so the probability scale for logit models
    emmeans_result_prob = model.emmeans("sex")
    assert emmeans_result_prob.shape[0] == 2  # 2 sex levels

    # Contrasts on the link scale
    emmeans_contrasts = model.emmeans(
        "sex", contrasts={"female - male": [-1, 1]}, type="link"
    )
    assert emmeans_contrasts.shape[0] == 1

    # Same as parameter estimate for sex
    assert np.allclose(
        emmeans_contrasts["estimate"].item(), model.result_fit[-1, "estimate"]
    )

    # Get marginal comparison on the probability scale
    # which is odds ratio for logit models
    emmeans_contrasts_odds_ratio = model.emmeans(
        "sex", contrasts={"female - male": [-1, 1]}
    )
    assert "odds_ratio" in emmeans_contrasts_odds_ratio.columns


def test_glmer_boot(titanic):
    """Test bootstrapped confidence intervals for glmer models."""
    # Fit model with default parametric CIs
    model = glmer("survived ~ fare + (1 | pclass)", data=titanic, family="binomial")
    model.fit()
    standard_ci = model.result_fit.select("conf_low", "conf_high").to_numpy()

    # Glmer default CIs are Wald
    model.fit(conf_method="wald")
    wald_ci = model.result_fit.select("conf_low", "conf_high").to_numpy()
    assert np.allclose(standard_ci, wald_ci)

    # Fit with bootstrapped CIs
    model.fit(conf_method="boot", nboot=100)
    boot_ci = model.result_fit.select("conf_low", "conf_high").to_numpy()

    # CIs should be different
    assert not np.allclose(standard_ci, boot_ci)

    # Test different bootstrapped CI types
    model.fit(conf_method="boot", nboot=100, ci_type="basic")
    boot_ci_basic = model.result_fit.select("conf_low", "conf_high").to_numpy()
    assert not np.allclose(boot_ci, boot_ci_basic)


def test_glmer_link_functions(titanic):
    """Test different link functions for glmer models."""
    # Default link for binomial is logit
    model_logit = glmer(
        "survived ~ fare + (1 | pclass)", data=titanic, family="binomial"
    )
    model_logit.fit()

    # Explicitly set logit link
    model_logit_explicit = glmer(
        "survived ~ fare + (1 | pclass)", data=titanic, family="binomial", link="logit"
    )
    model_logit_explicit.fit()

    # Should be the same
    assert np.allclose(
        model_logit.fixef.select(cs.numeric()),
        model_logit_explicit.fixef.select(cs.numeric()),
    )

    # Test probit link
    model_probit = glmer(
        "survived ~ fare + (1 | pclass)", data=titanic, family="binomial", link="probit"
    )
    model_probit.fit()

    # Should be different from logit
    assert not np.allclose(
        model_logit.fixef.select(cs.numeric()), model_probit.fixef.select(cs.numeric())
    )

    # Test cloglog link
    model_cloglog = glmer(
        "survived ~ fare + (1 | pclass)",
        data=titanic,
        family="binomial",
        link="cloglog",
    )
    model_cloglog.fit()

    # Should be different from logit and probit
    assert not np.allclose(
        model_logit.fixef.select(cs.numeric()), model_cloglog.fixef.select(cs.numeric())
    )
    assert not np.allclose(
        model_probit.fixef.select(cs.numeric()),
        model_cloglog.fixef.select(cs.numeric()),
    )

    # Test poisson with different links
    model_poisson = glmer(
        "survived ~ fare + (1 | pclass)", data=titanic, family="poisson"
    )
    model_poisson.fit()

    model_poisson_log = glmer(
        "survived ~ fare + (1 | pclass)", data=titanic, family="poisson", link="log"
    )
    model_poisson_log.fit()

    # Default for poisson is log, so these should be the same
    assert np.allclose(
        model_poisson.fixef.select(cs.numeric()),
        model_poisson_log.fixef.select(cs.numeric()),
    )

    # Test poisson with identity link
    model_poisson_identity = glmer(
        "survived ~ fare + (1 | pclass)",
        data=titanic,
        family="poisson",
        link="identity",
    )
    model_poisson_identity.fit()

    # Should be different from log link
    assert not np.allclose(
        model_poisson.fixef.select(cs.numeric()),
        model_poisson_identity.fixef.select(cs.numeric()),
    )


def test_glmer_predict_simulate(titanic):
    """Test predict and simulate methods for glmer models."""
    model = glmer(
        "survived ~ fare + sex + (1 | pclass)", data=titanic, family="binomial"
    )
    model.set_factors("sex")
    model.fit()

    # Test simulations
    sims = model.simulate(nsim=5)
    assert sims.shape[0] == titanic.shape[0]
    assert sims.shape[1] == 5  # 5 simulation columns

    # For binomial models, simulations should be 0/1
    assert set(np.unique(sims.to_numpy().flatten())).issubset({0, 1})

    # Simulate ignoring class-level random effects
    sims_no_rfx = model.simulate(use_rfx=False).to_numpy().squeeze()

    sims = sims[:, 0].to_numpy()
    assert not np.allclose(sims, sims_no_rfx)
