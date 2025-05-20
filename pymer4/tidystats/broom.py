from rpy2.robjects.packages import importr
from .bridge import ensure_py_output, to_dict, ensure_r_input
import rpy2.robjects as ro

__all__ = ["tidy", "glance", "augment"]

lib_broom = importr("broom")
lib_broom_mixed = importr("broom.mixed")


@ensure_py_output
@ensure_r_input
def tidy(model, **kwargs):
    """Summarize information about model components. Uses `broom.mixed::tidy.merMod <https://www.rdocumentation.org/packages/broom.mixed/versions/0.2.9.6/topics/lme4_tidiers>`_ for linear-mixed-models, `broom::tidy.lm <https://www.rdocumentation.org/packages/broom/versions/0.7.0/topics/tidy.lm>`_ for linear models, and `broom::tidy.glm <https://www.rdocumentation.org/packages/broom/versions/0.7.0/topics/tidy.glm>`_ for generalized linear models."""
    if isinstance(model, ro.methods.RS4):
        func = lib_broom_mixed.tidy_merMod
    elif isinstance(model, ro.vectors.ListVector):
        method = to_dict(model).get("method", None)
        if method and method[0] == "glm.fit":
            func = lib_broom.tidy_glm
        else:
            func = lib_broom.tidy_lm

    return func(model, **kwargs)


@ensure_py_output
def glance(model, /, **kwargs):
    """Report information about the entire model. Uses `broom.mixed:::glance.merMod <https://www.rdocumentation.org/packages/broom.mixed/versions/0.2.9.6/topics/lme4_tidiers>`_ for linear-mixed-models, `broom::glance.lm <https://www.rdocumentation.org/packages/broom/versions/0.7.0/topics/glance.lm>`_ for linear models, and `broom::glance.glm <https://www.rdocumentation.org/packages/broom/versions/0.7.0/topics/glance.glm>`_ for generalized linear models."""
    if isinstance(model, ro.methods.RS4):
        func = lib_broom_mixed.glance_merMod
    elif isinstance(model, ro.vectors.ListVector):
        method = to_dict(model).get("method", None)
        if method and method[0] == "glm.fit":
            func = lib_broom.glance_glm
        else:
            func = lib_broom.glance_lm

    return func(model, **kwargs)


@ensure_py_output
def augment(model, /, **kwargs):
    """Add information as observations to dataset. Uses `broom.mixed:::augment.merMod <https://www.rdocumentation.org/packages/broom.mixed/versions/0.2.9.6/topics/lme4_tidiers>`_ for linear-mixed-models, `broom::augment.lm <https://www.rdocumentation.org/packages/broom/versions/0.7.0/topics/augment.lm>`_ for linear models, and `broom::augment.glm <https://www.rdocumentation.org/packages/broom/versions/0.7.0/topics/augment.glm>`_ for generalized linear models."""
    if isinstance(model, ro.methods.RS4):
        func = lib_broom_mixed.augment_merMod
    if isinstance(model, ro.vectors.ListVector):
        method = to_dict(model).get("method", None)
        if method and method[0] == "glm.fit":
            func = lib_broom.augment_glm
        else:
            func = lib_broom.augment_lm

    return func(model, **kwargs)
