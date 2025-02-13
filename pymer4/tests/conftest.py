from pytest import fixture
from pymer4 import load_dataset


@fixture(scope="module")
def sample_data():
    return load_dataset("sample_data")


@fixture(scope="module")
def sleep():
    return load_dataset("sleep")


@fixture(scope="module")
def credit():
    return load_dataset("credit")


@fixture(scope="module")
def titanic():
    return load_dataset("titanic")


@fixture(scope="module")
def poker():
    return load_dataset("poker")


@fixture(scope="module")
def chickweight():
    return load_dataset("chickweight")


@fixture(scope="module")
def mtcars():
    return load_dataset("mtcars")


@fixture(scope="module")
def penguins():
    return load_dataset("penguins")
