from ._Pipe import pipeable
from .._broom import tidy as tidy_, augment as augment_, glance as glance_


@pipeable(None)
def tidy(model, **kwargs):
    # Your existing augment implementation
    return tidy_(model, **kwargs)


@pipeable(None)
def augment(model, **kwargs):
    # Your existing augment implementation
    return augment_(model, **kwargs)


@pipeable(None)
def glance(model, **kwargs):
    # Your existing augment implementation
    return glance_(model, **kwargs)
