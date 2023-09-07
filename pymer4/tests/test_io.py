from pymer4.models import Lm, Lm2, Lmer
from pymer4.io import load_model, save_model

def test_saveload_lmer(df, tmp_path):

    model = Lmer("DV ~ IV3 + IV2 + (IV2|Group) + (1|IV3)", data=df)
    model.fit(summarize=False)
    output_file = tmp_path / 'model.joblib'
    rds_file = tmp_path / 'model.rds'

    save_model(model, output_file)
    assert output_file.exists()
    assert rds_file.exists()

    m = load_model(output_file)
    assert m.coefs.equals(model.coefs)
    assert m.data.equals(model.data)


def test_saveload_lm(df, tmp_path):

    model = Lm("DV ~ IV1 + IV3", data=df)
    model.fit(summarize=False)
    output_file = tmp_path / 'model.joblib'

    save_model(model, output_file)
    assert output_file.exists()

    m = load_model(output_file)
    assert m.coefs.equals(model.coefs)
    assert m.data.equals(model.data)


def test_saveload_lm2(df, tmp_path):

    model = Lm2("DV ~ IV3 + IV2", group="Group", data=df)
    model.fit(summarize=False)
    output_file = tmp_path / 'model.joblib'

    save_model(model, output_file)
    assert output_file.exists()

    m = load_model(output_file)
    assert m.coefs.equals(model.coefs)
    assert m.data.equals(model.data)