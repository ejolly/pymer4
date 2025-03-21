from rpy2.robjects.packages import importr
from ._bridge import ensure_py_output, sanitize_polars_columns, drop_rownames
import rpy2.robjects as ro

__all__ = ["tidy", "glance", "augment"]

lib_broom = importr("broom")
lib_broom_mixed = importr("broom.mixed")


@sanitize_polars_columns
@ensure_py_output
def tidy(model):
    """Summarize information about model components"""
    if isinstance(model, ro.methods.RS4):
        func = lib_broom_mixed.tidy_merMod
    if isinstance(model, ro.vectors.ListVector):
        func = lib_broom.tidy_lm

    return func(model)


@sanitize_polars_columns
@ensure_py_output
def glance(model):
    """Report information about the entire model"""
    if isinstance(model, ro.methods.RS4):
        func = lib_broom_mixed.glance_merMod
    if isinstance(model, ro.vectors.ListVector):
        func = lib_broom.glance_lm

    return func(model)


@drop_rownames
@sanitize_polars_columns
@ensure_py_output
def augment(model):
    """Add information as observations to dataset"""
    if isinstance(model, ro.methods.RS4):
        func = lib_broom_mixed.augment_merMod
    if isinstance(model, ro.vectors.ListVector):
        func = lib_broom.augment_lm

    return func(model)
