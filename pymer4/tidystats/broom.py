from rpy2.robjects.packages import importr
from .bridge import ensure_py_output, to_dict
import rpy2.robjects as ro

__all__ = ["tidy", "glance", "augment"]

lib_broom = importr("broom")
lib_broom_mixed = importr("broom.mixed")


@ensure_py_output
def tidy(model, /, **kwargs):
    """Summarize information about model components"""
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
    """Report information about the entire model"""
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
    """Add information as observations to dataset"""
    if isinstance(model, ro.methods.RS4):
        func = lib_broom_mixed.augment_merMod
    if isinstance(model, ro.vectors.ListVector):
        method = to_dict(model).get("method", None)
        if method and method[0] == "glm.fit":
            func = lib_broom.augment_glm
        else:
            func = lib_broom.augment_lm

    return func(model, **kwargs)
