from pytest import fixture
from pymer4 import load_dataset


@fixture(scope="module")
def sleepstudy():
    return load_dataset("sleepstudy")
