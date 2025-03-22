from ._Pipe import pipeable
from .._emmeans import joint_tests as joint_tests_, emmeans as emmeans_

__all__ = ["joint_tests", "emmeans"]


@pipeable(None)
def joint_tests(model, **kwargs):
    return joint_tests_(model, **kwargs)


@pipeable(None)
def emmeans(model, **kwargs):
    return emmeans_(model, **kwargs)
