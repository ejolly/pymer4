import numpy as np
from pymer4.models import lm, glm, compare
from great_tables import GT
from polars import DataFrame


def test_logistic_regression(titanic):
    model = glm("survived ~ fare", data=titanic, family="binomial")
    model.fit()

    # Model calculates coefs on link scale by default like R
    params, result_fit = model.params, model.result_fit
    assert isinstance(params, DataFrame)
    assert isinstance(result_fit, DataFrame)

    # But we can get odds scale instead
    model.fit(exponentiate=True)
    assert not params.equals(model.params)
    assert not result_fit.equals(model.result_fit)

    coefs = model.summary()
    assert isinstance(coefs, GT)

    # Predictions are on the response scale by default, i.e. probabilities for logistic regression
    preds = model.predict(model.data)
    assert 0 <= preds.max() <= 1
    assert 0 <= preds.min() <= 1
    assert np.allclose(preds, model.data["fitted"].to_numpy().squeeze())

    # Link scale, i.e. log-odds for logistic regression
    preds = model.predict(model.data, type_predict="link")
    assert preds.min() < 0 and preds.max() > 1
    assert not np.allclose(preds, model.data["fitted"].to_numpy().squeeze())

    # Bootstrapped CI
    prev_result_fit = model.result_fit
    model.fit(conf_method="boot", nboot=100, save_boots=False)
    assert not prev_result_fit.select("conf_low", "conf_high").equals(
        model.result_fit.select("conf_low", "conf_high")
    )
    assert model.result_boots is None
    model.fit(conf_method="boot", nboot=100, save_boots=True)
    assert model.result_boots.height == 100
    assert isinstance(model.summary(), GT)

    # Anova
    model = glm("survived ~ sex", data=titanic, family="binomial")
    model.set_factors("sex")
    model.fit()
    model.anova()
    out = model.summary_anova()
    assert isinstance(model.result_anova, DataFrame)
    assert isinstance(out, GT)

    # Compare GLMs using LRT instead of F-test which is the default
    c = glm("survived ~ 1", data=titanic, family="binomial")
    a = glm("survived ~ fare", data=titanic, family="binomial")
    out = compare(c, a, as_dataframe=True, test="LRT")
    assert "Pr(>Chi)" in out.columns


def test_link_functions(titanic):
    # By default glm uses gaussian family with identity link
    gaussian = glm("survived ~ fare", data=titanic)
    gaussian.fit()

    # Which is the same as ols
    # OLS estimator = MLE for gaussian family
    ols = lm("survived ~ fare", data=titanic)
    ols.fit()
    # Compare the coefficient values from the DataFrames
    gaussian_estimates = gaussian.params.get_column("estimate")
    ols_estimates = ols.params.get_column("estimate")
    assert np.allclose(gaussian_estimates, ols_estimates)

    # Logistic regression with 2 different link functions
    log_binom = glm("survived ~ fare", data=titanic, family="binomial")
    log_binom.fit()
    log_probit = glm("survived ~ fare", data=titanic, family="binomial", link="probit")
    log_probit.fit()
    # Compare the coefficient values from the DataFrames
    log_binom_estimates = log_binom.params.get_column("estimate")
    log_probit_estimates = log_probit.params.get_column("estimate")
    assert not np.allclose(log_binom_estimates, log_probit_estimates)
    assert log_binom.link != log_probit.link

    # Logistic regression with poisson family
    poisson = glm("survived ~ fare", data=titanic, family="poisson")
    poisson.fit()
    poisson_identity = glm(
        "survived ~ fare", data=titanic, family="poisson", link="identity"
    )
    poisson_identity.fit()
    # Compare the coefficient values from the DataFrames
    poisson_estimates = poisson.params.get_column("estimate")
    poisson_identity_estimates = poisson_identity.params.get_column("estimate")
    assert not np.allclose(poisson_estimates, poisson_identity_estimates)
    assert poisson.link != poisson_identity.link
