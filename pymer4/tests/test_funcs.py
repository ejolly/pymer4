from numpy import ndarray
from polars import DataFrame
from polars.datatypes import String, Enum
from rpy2.robjects.vectors import FactorVector, StrVector
import pymer4.tidystats as ts
from pymer4.config import test_install
from pymer4.models import lmer, lm, glm, glmer
from rpy2.robjects.packages import importr

lib_stats = importr("stats")


def test_config():
    test_install()


def test_categorical_conversion(poker):
    # By default str -> R string/character
    rframe = ts.polars2R(poker)
    assert isinstance(poker["hand"].dtype, String)
    assert isinstance(rframe.rx2("hand"), StrVector)

    # enum (polars categorical) -> factor
    poker = ts.make_factors(poker, ["hand"])
    rframe = ts.polars2R(poker)
    assert isinstance(poker["hand"].dtype, Enum)
    assert isinstance(rframe.rx2("hand"), FactorVector)
    # we didn't convert this to a factor
    assert isinstance(rframe.rx2("limit"), StrVector)


def test_to_dict(poker):
    """Test converting unsupported types to Python dicts. Mostly applies to R models and summaries."""

    model = ts.lm("balance ~ hand", data=poker)
    model_dict = ts.to_dict(model)
    assert isinstance(model_dict, dict)

    summary = ts.summary(model)
    summary_dict = ts.to_dict(summary)
    assert isinstance(summary_dict, dict)


def test_anova_emmeans(poker):
    # Basic lm with categorical variables
    model = ts.lm("balance ~ hand*skill*limit", data=poker)
    result = ts.tidy(model)
    assert isinstance(result, DataFrame)

    # Anova
    anova_table = ts.joint_tests(model)
    assert isinstance(anova_table, DataFrame)
    assert anova_table.shape == (7, 5)

    # Marginal means
    means = ts.emmeans(model, specs="hand")
    assert isinstance(means, DataFrame)
    assert means.shape == (3, 6)

    # Multiple factors
    means2 = ts.emmeans(model, specs=["hand", "skill"])
    assert isinstance(means2, DataFrame)
    assert means2.shape == (6, 7)

    # Equivalent
    means2by = ts.emmeans(model, specs="hand", by="skill")
    assert isinstance(means2by, DataFrame)
    assert means2by.shape == (6, 7)
    assert means2.equals(means2by)

    # Pairwise contrasts
    means = ts.emmeans(model, specs="hand", contrasts="pairwise")
    assert isinstance(means, DataFrame)
    assert means.shape == (3, 6)

    # Custom contrasts
    means = ts.emmeans(
        model, specs="hand", contrasts={"lin_bad_neutral_good": [-1, 1, 0]}
    )
    assert isinstance(means, DataFrame)
    assert means.shape == (1, 6)


def test_model_contrast_setting(poker):
    # Dummy/treatment default
    treatment = ts.lm("balance ~ hand", data=poker)
    mat_treat = ts.model_matrix(treatment)
    assert "handgood" in mat_treat.columns and "handneutral" in mat_treat.columns

    # Set explicitly
    m = ts.lm("balance ~ hand", data=poker, contrasts={"hand": "contr.treatment"})
    mat = ts.model_matrix(m)
    assert mat_treat.equals(mat)

    # Or with explicit factor conversion on the Python side
    m = ts.lm("balance ~ hand", data=ts.make_factors(poker, ["hand"]))
    mat = ts.model_matrix(m)
    assert mat_treat.equals(mat)

    # Sum
    sum_code = ts.lm("balance ~ hand", data=poker, contrasts={"hand": "contr.sum"})
    mat_sum = ts.model_matrix(sum_code).unique()
    assert "hand1" in mat_sum.columns and "hand2" in mat_sum.columns
    assert not mat_treat.equals(mat_sum)

    # Poly
    poly_code = ts.lm("balance ~ hand", data=poker, contrasts={"hand": "contr.poly"})
    mat_poly = ts.model_matrix(poly_code).unique()
    # Note: our decorators auto-convert names with "." to "_"
    assert "hand_L" in mat_poly.columns and "hand_Q" in mat_poly.columns
    assert not mat_treat.equals(mat_poly) and not mat_sum.equals(mat_poly)


def test_predict_simulate(sleep):
    ols = ts.lm("Reaction ~ Days", data=sleep)

    preds = ts.predict(ols)
    assert isinstance(preds, ndarray)
    assert len(preds) == sleep.height

    preds = ts.predict(ols, sleep[:10])
    assert len(preds) == 10

    sims = ts.simulate(ols)
    assert isinstance(sims, DataFrame)
    assert sims.height == sleep.height

    sims = ts.simulate(ols, nsim=3)
    assert sims.height == sleep.height
    assert sims.width == 3


def test_bootMer(sleep):
    formula = "Reaction ~ Days + (1 | Subject)"
    model = ts.lmer(formula, data=sleep)
    cis, boots = ts.bootMer(model, nsim=500, ncpus=1)
    assert cis.height == 4
    assert boots.shape == (500, 4)


def test_glm_family(titanic):
    formula = "survived ~ fare"

    # Logit link default
    model = ts.glm(formula, data=titanic, family="binomial")
    coef_1 = ts.tidy(model)

    # Manually set family and link
    family = getattr(lib_stats, "binomial")
    r_family_link = family()
    model = ts.glm(formula, data=titanic, family=r_family_link)
    coef_2 = ts.tidy(model)
    assert coef_1.equals(coef_2)

    probit_family_link = family(link="probit")
    model = ts.glm(formula, data=titanic, family=probit_family_link)
    coef_3 = ts.tidy(model)
    assert not coef_1.equals(coef_3)


def test_easy_boot(sleep):
    model = lmer("Reaction ~ Days + (Days | Subject)", data=sleep)
    model.fit()
    boots = ts.bootstrap_model(model)
    assert boots.shape == (1000, 2)


def test_model_params(sleep):
    ols = lm("Reaction ~ Days", data=sleep)
    ols.fit()
    lmm = lmer("Reaction ~ Days + (Days | Subject)", data=sleep)
    lmm.fit()
    mle = glm("Reaction ~ Days", data=sleep, family="gaussian")
    mle.fit()
    log_lmm = glmer(
        "Reaction ~ Days + (Days | Subject)", data=sleep, family="gamma", link="log"
    )
    log_lmm.fit()

    ols_params = ts.model_params(ols)
    mle_params = ts.model_params(mle)
    assert ols_params.shape == mle_params.shape
    lmm_params = ts.model_params(lmm)
    log_lmm_params = ts.model_params(log_lmm, ci_method="wald")
    assert lmm_params.shape == log_lmm_params.shape
