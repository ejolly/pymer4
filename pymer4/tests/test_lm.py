import numpy as np
from pymer4.models import lm, compare
from great_tables import GT
import polars as pl
from polars import DataFrame, String, Enum, Float64, Int64, col, selectors as cs
import pytest


def test_model_basics(credit):
    # Model stores data
    m = lm("Balance ~ Income", data=credit)
    assert m.data.width == 11

    # Fitting model adds output of augment()
    m.fit()
    assert m.data.width == 17

    # Summary is a Great Table
    assert isinstance(m.summary(), GT)

    # But coef is a DataFrame
    assert isinstance(m.result_fit, DataFrame)

    # We can get the summary table during fit instead
    assert isinstance(m.fit(summary=True), GT)

    # Parameters are now a polars DataFrame
    assert isinstance(m.params, DataFrame)

    # Anova
    m.anova()
    assert isinstance(m.result_anova, DataFrame)
    assert isinstance(m.summary_anova(), GT)

    # Bootstrapped CI summary
    prev_result_fit = m.result_fit
    m.fit(conf_method="boot")
    assert not prev_result_fit.select("conf_low", "conf_high").equals(
        m.result_fit.select("conf_low", "conf_high")
    )

    # We can optionally return bootstrapped coefficients as a DataFrame
    m.fit(conf_method="boot", save_boots=True)
    assert m.result_boots.height == 1000

    # We can mean-center the data
    unscaled_params = m.params
    m.set_transforms({"Income": "center"})
    assert "Income_orig" in m.data.columns
    assert m.transformed == {"Income": "center"}
    m.fit()
    scaled_params = m.params
    # Intercepts should change
    assert not np.allclose(
        scaled_params.select("estimate")[0, 0],
        unscaled_params.select("estimate")[0, 0],
    )
    # But slope wont
    assert np.allclose(
        scaled_params.select("estimate")[1, 0],
        unscaled_params.select("estimate")[1, 0],
    )

    # And unscale it
    m.unset_transforms("Income")
    assert "Income_orig" not in m.data.columns
    m.fit()
    new_params = m.params
    # Check the values are restored after unscaling
    assert np.allclose(
        new_params.get_column("estimate").to_list(),
        unscaled_params.get_column("estimate").to_list(),
    )

    # Scale multiple columns
    m.set_transforms({"Income": "center", "Limit": "zscore"})
    assert "Income_orig" in m.data.columns
    assert "Limit_orig" in m.data.columns

    income_orig = m.data.select("Income_orig").to_numpy().squeeze()
    income_center = m.data.select("Income").to_numpy().squeeze()
    limit_orig = m.data.select("Limit_orig").to_numpy().squeeze()
    limit_zscore = m.data.select("Limit").to_numpy().squeeze()

    assert not np.allclose(income_orig, income_center)
    assert not np.allclose(limit_orig, limit_zscore)

    # Selectively unscale
    m.unset_transforms("Income")
    assert m.transformed == {"Limit": "zscore"}

    current_income = m.data.select("Income").to_numpy().squeeze()
    assert np.allclose(current_income, income_orig)
    assert "Income_orig" not in m.data.columns
    assert "Limit_orig" in m.data.columns

    # Rescale an existing transform which undoes it first
    m.set_transforms({"Limit": "rank"})
    limit_rank = m.data.select("Limit").to_numpy().squeeze()
    current_limit_orig = m.data.select("Limit_orig").to_numpy().squeeze()

    assert not np.allclose(limit_zscore, limit_rank)
    assert not np.allclose(limit_orig, limit_rank)
    assert not np.allclose(limit_orig, limit_zscore)
    assert np.allclose(limit_orig, current_limit_orig)


def test_model_comparison(credit):
    null_model = lm("Balance ~ 1", data=credit)
    full_model = lm("Balance ~ Income", data=credit)

    # Model comparison will automatically call .fit()
    out = compare(null_model, full_model)
    assert isinstance(out, GT)

    table = compare(null_model, full_model, as_dataframe=True)
    assert isinstance(table, DataFrame)


def test_factors_and_default_R_contrasts(chickweight):
    # For testing purposes we'll cast an integer to float and recover it
    df = chickweight.with_columns(col("Time").cast(float))

    # By default numeric Python types are treated as numeric R types
    # and thus continuous predictors
    # time is float and diet is int
    model = lm("weight ~ Time * Diet", data=df)
    assert model.data["Time"].dtype == Float64
    assert model.data["Diet"].dtype == Int64

    model.fit()
    assert model.result_fit.height == 4

    model.anova()
    cont_anova = model.result_anova
    cont_design = model.design_matrix

    # But we can request factors which sets contrasts to treatment by default
    # and are converted to R factors
    model.set_factors(["Diet", "Time"])
    assert model.data["Time"].dtype == Enum
    assert model.data["Diet"].dtype == Enum
    assert model.factors is not None
    assert model.contrasts == {"Diet": "contr.treatment", "Time": "contr.treatment"}

    # And now estimates a different number of parameters
    model.fit()
    assert model.result_fit.height == 48

    model.anova()
    fac_anova = model.result_anova
    # Which is reflected in the model's design matrix
    assert not fac_anova.equals(cont_anova)
    assert not model.design_matrix.equals(cont_design)
    dummy_design = model.design_matrix

    # Change contrasts from dummy/treatment to poly
    model.set_contrasts({"Diet": "contr.poly", "Time": "contr.poly"})
    model.fit()
    assert not model.design_matrix.equals(dummy_design)

    # And back to dummy/treatment to verify it's the default
    model.set_contrasts({"Diet": "contr.treatment", "Time": "contr.treatment"})
    model.fit()
    assert model.design_matrix.equals(dummy_design)

    # Unset factors preserves the original numeric types
    model.unset_factors()
    assert model.data["Time"].dtype == Float64
    assert model.data["Diet"].dtype == Int64


def test_factor_levels_and_planned_contrasts(poker):
    good_mean = poker.filter(col("hand") == "good").select("balance").mean().item()
    bad_mean = poker.filter(col("hand") == "bad").select("balance").mean().item()
    neutral_mean = (
        poker.filter(col("hand") == "neutral").select("balance").mean().item()
    )

    # Let's say we want 'hand' to be treated as a factor with 3 levels
    # Fitting this model will have unpredictable behavior
    # if levels of hand are numbers R will treat it continuously
    # if levels of hand are strings R will treat it categorically
    model = lm("balance ~ hand", data=poker)
    assert isinstance(model.data["hand"].dtype, String)

    # Instead explicitly set the factors which will convert the dtypes
    # to polars enums which are categoricals and guaranteed to be cast to R
    # factors by rpy2
    model.set_factors("hand")
    assert isinstance(model.data["hand"].dtype, Enum)
    # and the default factor levels are alphabetical
    default_order = ["bad", "good", "neutral"]

    assert model.factors == {"hand": default_order}

    model.fit()
    # Emmeans will also use the specified level order
    assert model.emmeans("hand")["hand"].to_list() == default_order

    # By default b0 = bad, b1 = good-bad, b2 = neutral-bad
    assert np.allclose(model.params.select("estimate")[0, 0], bad_mean)
    assert np.allclose(model.params.select("estimate")[1, 0], good_mean - bad_mean)
    assert np.allclose(model.params.select("estimate")[2, 0], neutral_mean - bad_mean)

    # We can also re-order factor levels by using a dictionary instead
    # If we don't unset factors first we'll get an error
    # with pytest.raises(ValueError):
    #     model.set_factors({"hand": ["good", "bad", "neutral"]})

    # model.unset_factors()
    model.set_factors({"hand": ["good", "bad", "neutral"]})
    model.fit()
    assert model.factors == {"hand": ["good", "bad", "neutral"]}
    assert model.emmeans("hand")["hand"].to_list() == ["good", "bad", "neutral"]
    assert np.allclose(model.params.select("estimate")[1, 0], bad_mean - good_mean)

    # Changing the default contrast type will also respect the level order
    # bad < neutral < good
    # model.unset_factors()
    model.set_factors({"hand": ["bad", "neutral", "good"]})
    model.set_contrasts({"hand": "contr.poly"})
    model.fit()

    # Check against the linear combination of means
    contrast_weights = model.design_matrix.select("hand_L").to_numpy().squeeze()
    lin_poly = np.dot(contrast_weights, [bad_mean, neutral_mean, good_mean])
    assert np.allclose(model.params.select("estimate")[1, 0], lin_poly)

    # We can also specify custom contrasts in terms of cell means and they'll
    # be convert to R contrast codes for us
    # model.unset_factors()
    model.set_factors({"hand": ["bad", "neutral", "good"]})
    model.set_contrasts({"hand": [-1, 0, 1]}, normalize=True)
    model.fit()
    assert np.allclose(model.params.select("estimate")[1, 0], lin_poly)

    # If we don't normalize then we'll just use the raw contrast values
    model.set_contrasts({"hand": [-1, 0, 1]}, normalize=False)
    model.fit()
    raw_lin_con = np.dot([-1, 0, 1], [bad_mean, neutral_mean, good_mean])
    assert np.allclose(model.params.select("estimate")[1, 0], raw_lin_con)

    # We can also set multiple, potentially non-orthgonal contrasts
    model.set_contrasts({"hand": np.array([[-1, 0, 1], [-1, 1, 0]])}, normalize=False)
    model.fit()
    assert np.allclose(model.params.select("estimate")[1, 0], good_mean - bad_mean)
    assert np.allclose(model.params.select("estimate")[2, 0], neutral_mean - bad_mean)


def test_unbalanced_anova(poker):
    df_unbalanced = poker[10:, :]
    m = lm("balance ~ hand+skill", data=df_unbalanced)
    m.set_factors(["hand", "skill"])

    m.fit()
    # Unbalanced designs will differ between type I and type III SS
    m.anova()
    typ3 = m.result_anova
    m.anova(auto_ss_3=False)
    typ1 = m.result_anova
    assert not typ3.equals(typ1)

    # And the order of terms will matter for type I SS
    m_flip = lm("balance ~ skill+hand", data=df_unbalanced)
    m_flip.set_factors(["skill", "hand"])
    m_flip.anova()
    typ3_flip = m_flip.result_anova
    m_flip.anova(auto_ss_3=False)
    typ1_flip = m_flip.result_anova
    assert not typ3_flip.equals(typ1_flip)

    # However order doesn't matter for type III SS
    assert np.allclose(
        typ3[:, 1:].to_numpy(), typ3_flip.sort(by="model term")[:, 1:].to_numpy()
    )


def test_categorical_emmeans(poker, sleep):
    # hand: [bad, good, neutral]
    # skill: [average, expert]
    m = lm("balance ~ hand*skill", data=poker)
    m.set_factors(["hand", "skill"])
    m.fit()

    # We can get estimated marginal means aka "cell means" for each combination of factors
    emms = m.emmeans("hand", by="skill").sort(by=["hand", "skill"])
    assert emms.shape == (6, 7)

    # Compare them to the actual cell means
    cell_means = (
        poker.group_by(["hand", "skill"])
        .mean()
        .drop("limit")
        .sort(by=["hand", "skill"])
    )
    assert np.allclose(emms["emmean"].to_numpy(), cell_means["balance"].to_numpy())

    # We can also get all pairwise contrasts between levels of a factor
    # [bad-good, bad-neutral, good-neutral] within [average, expert]
    contrasts = m.emmeans("hand", by="skill", contrasts="pairwise")
    assert contrasts.shape == (6, 9)

    # average - expert within [bad, good, neutral]
    contrasts = m.emmeans("skill", by="hand", contrasts="pairwise")
    assert contrasts.shape == (3, 9)

    # We can also run custom contrasts
    # Levels correspond to order of factors levels which is alphabetical by default
    # bad: -1, good: 0, neutral: 1
    custom = m.emmeans("hand", contrasts={"bad_neutral_good_linear": [-1, 1, 0]})
    contrast = custom["estimate"].item()

    # Check against manually computing contrast using marginal means
    emms = m.emmeans("hand")
    assert np.allclose(contrast, np.dot(emms["emmean"].to_numpy(), [-1, 1, 0]))

    # Same but use cell means
    means = (
        poker.group_by("hand")
        .mean()
        .sort(by="hand")
        .select("balance")
        .to_numpy()
        .squeeze()
    )
    assert np.allclose(contrast, np.dot(means, [-1, 1, 0]))

    # Contrast with many levels
    model = lm("Reaction ~ Days", data=sleep)
    model.set_factors("Days")
    model.fit()
    assert model.result_fit.height == 10

    # Custom contrast == 1st beta
    contrast = model.emmeans(
        "Days", contrasts={"first_vs_second": [-1, 1, 0, 0, 0, 0, 0, 0, 0, 0]}
    )
    np.allclose(model.result_fit[1, "estimate"], contrast["estimate"].item())


def test_mixed_emmeans(credit):
    # Cell means marginalizing over continuous variable
    # Because groups are not balanced and we have an interaction we do not expect these to be the same as the predicted/estimated marginal means!
    yes_mean = credit.filter(col("Student") == "Yes").select("Balance").mean().item()
    no_mean = credit.filter(col("Student") == "No").select("Balance").mean().item()

    # Uncentered continuous predictor + treatment coded 2-level factor
    # b0 = mean for Student=No when Income=0
    # b1 = Income slope for Student=No
    # b2 = difference between Student=Yes - Student=No when Income=0
    # b3 = difference in Income slope for Student=Yes - Student=No
    model = lm("Balance ~ Income*Student", data=credit)
    model.set_factors("Student")
    model.fit()

    # Factor marginal estimates:
    # emmeans holds continuous covariates at their means
    marginal_means = model.emmeans("Student")
    marginal_mean_diff = model.emmeans("Student", contrasts="pairwise")
    no_mean = marginal_means.filter(col("Student") == "No").select("emmean").item()
    yes_mean = marginal_means.filter(col("Student") == "Yes").select("emmean").item()
    yes_minus_no = yes_mean - no_mean
    no_minus_yes = no_mean - yes_mean
    assert marginal_means.shape == (2, 6)

    # Continuous marginal slopes:
    # emmeans uses evenly weighted sum coding for factors
    marginal_slopes = model.emmeans("Income", by="Student")
    no_slope = (
        marginal_slopes.filter(col("Student") == "No").select("Income_trend").item()
    )
    yes_slope = (
        marginal_slopes.filter(col("Student") == "Yes").select("Income_trend").item()
    )
    yes_slope_minus_no_slope = yes_slope - no_slope
    no_slope_minus_yes_slope = no_slope - yes_slope
    average_slope = np.mean([yes_slope, no_slope])

    # Marginal var must be factors OR continuous by not both
    with pytest.raises(TypeError):
        model.emmeans(marginal_var=["Student", "Income"])

    # Check against default lm parameter estimates:
    # Model parameters will not match emmeans
    # because beta estimates assume Income = 0
    assert model.params.select("estimate")[0, 0] != no_mean
    assert model.params.select("estimate")[2, 0] != yes_minus_no
    # And the continuous term is the slope just for Student=No
    assert np.allclose(model.params.select("estimate")[1, 0], no_slope)
    # But the interaction is correct: difference between slopes across factor levels
    assert np.allclose(model.params.select("estimate")[3, 0], yes_slope_minus_no_slope)

    # If we mean center then our categorical parameters match
    model.set_transforms({"Income": "center"})
    model.fit()
    assert np.allclose(model.params.select("estimate")[0, 0], no_mean)
    assert np.allclose(model.params.select("estimate")[2, 0], yes_minus_no)

    # Which is the same as computing a contrast of the marginal means
    # because emmeans holds covariates at their mean.
    # In this case the order is flipped from the beta estimate
    estimate = marginal_mean_diff.select("estimate").item()
    assert np.allclose(model.params.select("estimate")[2, 0], -estimate)
    assert np.allclose(estimate, no_minus_yes)

    # Specifying 'by' doesn't change anything for marginal mean contrasts
    # We'll simply get another column showing the value of the covariate
    # is being held at
    pairwise_mean_diff_2 = model.emmeans(
        marginal_var="Student", by="Income", contrasts="pairwise"
    )
    assert np.allclose(
        pairwise_mean_diff_2.select(cs.exclude("Income", "contrast")).to_numpy(),
        marginal_mean_diff.select(cs.exclude("contrast")).to_numpy(),
    )

    # However the slope estimate for Income is still only for Student=No
    assert np.allclose(model.params.select("estimate")[1, 0], no_slope)
    # And the interaction is unaffected by centering
    assert np.allclose(model.params.select("estimate")[3, 0], yes_slope_minus_no_slope)

    # To fix the slope, we need to code our factor using sum coding
    # which estimates a slope "in between" factor levels, i.e. their average
    model.set_contrasts({"Student": "contr.sum"})
    model.fit()
    assert np.allclose(model.params.select("estimate")[1, 0], average_slope)

    # however this will change the scale of the interaction and categorical parameters
    # contr.sum will do No: 1; Yes: -1 by default
    assert np.allclose(model.params.select("estimate")[2, 0], no_minus_yes / 2)
    assert np.allclose(
        model.params.select("estimate")[3, 0], no_slope_minus_yes_slope / 2
    )

    # We can set them manually to check
    # When we set them manually, we use *human-readable format*
    # Which is auto-converted to R contrast codes via matrix inversion
    # This will reproduce R's result
    model.set_contrasts({"Student": [0.5, -0.5]})
    model.fit()
    assert np.allclose(model.params.select("estimate")[2, 0], no_minus_yes / 2)
    assert np.allclose(
        model.params.select("estimate")[3, 0], no_slope_minus_yes_slope / 2
    )

    # But we can just do this, which gives us the contrast we actually want
    model.set_contrasts({"Student": [1, -1]})
    model.fit()
    assert np.allclose(model.params.select("estimate")[2, 0], no_minus_yes)
    assert np.allclose(model.params.select("estimate")[3, 0], no_slope_minus_yes_slope)

    # We can also compute slope contrasts, i.e. the interaction by comparing marginal
    # estimates
    contrast = model.emmeans(marginal_var="Income", by="Student", contrasts="pairwise")
    contrast_estimate = contrast.select("estimate").item()
    # Should match the diff in marginal estimates we calculated manually
    assert np.allclose(contrast_estimate, no_slope_minus_yes_slope)
    # And the interaction term
    assert np.allclose(contrast_estimate, model.params.select("estimate")[3, 0])

    # We can also generate more specific marginal predictions
    # using emmeans reference grid
    # Uncenter, reset to treatment contrasts, and re-fit model
    model.unset_transforms()
    model.set_factors("Student")
    model.fit()
    # Grab the mean
    income_mean = model.data.select("Income").mean().item()

    # Make a marginal prediction per factor level at the mean
    preds = model.empredict({"Income": income_mean})

    # Which are the same as the marginal means we get from emmeans
    assert np.allclose(
        preds.select("prediction").to_numpy().squeeze(), [no_mean, yes_mean]
    )

    # The default uncentered, dummy-coded parameter estimate for student
    # is the difference between levels when Income = 0
    # let's verify that
    beta = model.params.select("estimate")[2, 0]
    # Compute a contrast at a specific value of continuous predictor
    marginal_diff = model.emmeans(
        "Student", contrasts={"yes_minus_no": [-1, 1]}, at={"Income": 0}
    )
    marginal_diff = marginal_diff.select("estimate").item()
    assert np.allclose(beta, marginal_diff)

    # We can very the intercept in the same way
    marginal_pred = (
        model.empredict(at={"Income": 0, "Student": "No"}).select("prediction").item()
    )
    intercept = model.params.select("estimate")[0, 0]
    assert np.allclose(intercept, marginal_pred)


def test_predict_simulate(mtcars):
    m = lm("mpg ~ cyl + hp", data=mtcars)
    m.fit()

    # Predictions require a dataframe and return a numpy array
    pred = m.predict(mtcars)
    assert isinstance(pred, np.ndarray)
    assert np.allclose(pred, m.data["fitted"].to_numpy())

    # Simulations always return a dataframe with width equal to the number of simulations
    sim = m.simulate(2)
    assert sim.shape == (m.data.height, 2)
    assert isinstance(m.simulate(1), DataFrame)


def test_wls(credit):
    from scipy.stats import ttest_ind

    student = credit.filter(col("Student") == "Yes").select("Balance")
    non_student = credit.filter(col("Student") == "No").select("Balance")
    results = ttest_ind(student, non_student, equal_var=False)

    # Create weight column that's inverse of variance of each factor level
    credit = credit.with_columns(
        pl.when(col("Student") == "No")
        .then(pl.lit(1 / non_student.var(ddof=1).item()))
        .otherwise(pl.lit(1 / student.var(ddof=1).item()))
        .alias("student_weights")
    )

    # Test by referring to weights as a string
    wls = lm("Balance ~ Student", weights="student_weights", data=credit)
    wls.set_factors("Student")
    wls.fit()

    ols = lm("Balance ~ Student", data=credit)
    ols.set_factors("Student")
    ols.fit()

    assert np.allclose(results.statistic[0], wls.result_fit[-1, "t_stat"])
    assert not ols.params.equals(wls.params)
