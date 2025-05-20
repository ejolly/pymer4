from pymer4.models import glm, lmer
from pymer4.io import load_model, save_model


def test_saveload_lmer_model(sleep, tmp_path):
    model = lmer("Reaction ~ Days + (Days|Subject)", data=sleep)
    model.fit()

    # tmp_path = Path(".")
    output_file = tmp_path / "model.joblib"
    rds_file = tmp_path / "model.rds"

    # Lmer models
    save_model(model, output_file)
    assert output_file.exists()
    assert rds_file.exists()

    m = load_model(output_file)
    assert m.result_fit.equals(model.result_fit)


def test_saveload_lm_model(sleep, tmp_path):
    # Lm models
    model = glm("Reaction ~ Days", data=sleep)
    model.fit()

    # tmp_path = Path(".")
    output_file = tmp_path / "model.joblib"
    rds_file = tmp_path / "model.rds"

    save_model(model, output_file)
    assert output_file.exists()
    assert rds_file.exists()

    m = load_model(output_file)
    assert m.result_fit.equals(model.result_fit)
