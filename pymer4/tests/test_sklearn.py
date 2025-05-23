import numpy as np
import polars as pl
from pymer4.models import skmer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from pymer4 import load_dataset
from scipy.stats import pearsonr
import pytest

penguins = load_dataset("penguins")
sleep = load_dataset("sleep")
mtcars = load_dataset("mtcars")
titanic = load_dataset("titanic")
corr = lambda x, y: pearsonr(x, y)[0]


def test_skmer_lm(penguins):
    # Initialize just with formula
    # default model class is lm()
    ols = skmer("flipper_length_mm ~ bill_length_mm + bill_depth_mm + body_mass_g")
    # We expect many tests to fail because the model X shape is parameterized by its formula
    # results = check_estimator(ols, on_fail=None)

    # Prepare data sklearn style
    penguins = penguins.drop_nulls()
    X = penguins[["bill_length_mm", "bill_depth_mm", "body_mass_g"]].to_numpy()
    y = penguins["flipper_length_mm"].to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Test fit
    ols.fit(X_train, y_train)
    assert hasattr(ols, "coef_")
    assert ols.coef_.shape[0] == 4  # intercept + 3 features
    assert hasattr(ols, "n_features_in_")
    assert ols.n_features_in_ == 3

    # Test predict
    y_pred_train = ols.predict(X_train)
    y_pred_test = ols.predict(X_test)
    score_train = r2_score(y_train, y_pred_train)
    score_test = r2_score(y_test, y_pred_test)
    assert isinstance(y_pred_train, np.ndarray)
    assert isinstance(y_pred_test, np.ndarray)
    # Both scores should be reasonable (model is fitting well)
    assert score_train > 0.7
    assert score_test > 0.7

    # Test get_params
    params = ols.get_params()
    assert (
        params["formula"]
        == "flipper_length_mm ~ bill_length_mm + bill_depth_mm + body_mass_g"
    )
    assert params["model_class"] == "lm"

    # Test set_params
    ols.set_params(formula="flipper_length_mm ~ bill_length_mm + bill_depth_mm")
    assert ols.formula == "flipper_length_mm ~ bill_length_mm + bill_depth_mm"

    # Categorical features encoded as treatment contrasts
    # in R, so we don't need to use sklearn's OneHotEncoder
    ols = skmer("flipper_length_mm ~ species")

    # Prepare data sklearn style
    X = penguins[["species"]].to_numpy()
    y = penguins["flipper_length_mm"].to_numpy()
    mask = ~np.isnan(y)
    X = X[mask]
    y = y[mask]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Test fit
    ols.fit(X_train, y_train)


def test_skmer_lmer(sleep):
    # LMM
    lmm = skmer(
        "Reaction ~ Days + (Days | Subject)",
        model_class="lmer",
    )

    # Prepare data
    sleep = sleep.drop_nulls()
    X = sleep[["Days"]].to_numpy()
    y = sleep["Reaction"].to_numpy()
    group = sleep["Subject"].to_numpy()

    # For mixed models, append group as last column of X
    X_with_group = np.column_stack([X, group])

    lmm.fit(X_with_group, y)
    assert len(lmm.coef_) == 2

    # RFX coef_
    assert isinstance(lmm.coef_rfx_, pl.DataFrame)
    assert lmm.coef_rfx_.shape == (18, 3)
    assert lmm.n_features_in_ == 1

    # Predict with/without random effects
    y_fixed = lmm.predict(X)
    y_rfx = lmm.predict(X_with_group)
    assert not np.allclose(y_fixed, y_rfx)
    score = r2_score(y, y_fixed)
    assert 0 <= score <= 1
    score_rfx = r2_score(y, y_rfx)
    assert score_rfx > score


def test_skmer_glm(titanic):
    # Create binary outcome for GLM testing
    glm = skmer(
        "survived ~ sex + pclass + age + fare", model_class="glm", family="binomial"
    )

    # Prepare data
    titanic = titanic.drop_nulls()
    X = titanic[["sex", "pclass", "age", "fare"]].to_numpy()
    y = titanic["survived"].to_numpy()

    glm.fit(X, y)

    # Predictions should be probabilities between 0 and 1
    y_pred = glm.predict(X[:10])
    assert y_pred.shape[0] == 10
    assert np.all((y_pred >= 0) & (y_pred <= 1))


def test_skmer_sklearn_integration(penguins, mtcars, sleep):
    # Prepare data
    penguins = penguins.drop_nulls()
    X = penguins[["bill_length_mm", "bill_depth_mm", "body_mass_g"]].to_numpy()
    y = penguins["flipper_length_mm"].to_numpy()

    # Test in sklearn pipeline
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                skmer(
                    "flipper_length_mm ~ bill_length_mm + bill_depth_mm + body_mass_g"
                ),
            ),
        ]
    )

    pipeline.fit(X, y)
    y_pred = pipeline.predict(X[:10])
    assert y_pred.shape[0] == 10

    # Test cross-validation
    mtcars = mtcars.drop_nulls()
    X = mtcars[["disp", "hp"]].to_numpy()
    y = mtcars["mpg"].to_numpy()

    scores = cross_val_score(
        skmer("mpg ~ disp + hp"),
        X,
        y,
        cv=5,
        scoring="r2",
    )
    linreg_scores = cross_val_score(
        LinearRegression(),
        X,
        y,
        cv=5,
        scoring="r2",
    )
    assert np.allclose(scores, linreg_scores)
    assert len(scores) == 5

    # cross_validation with lmer
    sleep = sleep.drop_nulls()
    X = sleep[["Days"]].to_numpy()
    y = sleep["Reaction"].to_numpy()
    group = sleep["Subject"].to_numpy()

    # For cross-validation with mixed models, append group as last column of X
    X_with_group = np.column_stack([X, group])

    # Test cross-validation with mixed models works
    lmm_is_scores = cross_val_score(
        skmer("Reaction ~ Days + (Days | Subject)"),
        X_with_group,
        y,
        groups=group,  # Still needed for GroupKFold splitting
        cv=LeaveOneGroupOut(),
        scoring="r2",
    )
    assert len(lmm_is_scores) == len(np.unique(group))
    scores = np.maximum(lmm_is_scores, 0)
    assert scores.mean() > 0.2
