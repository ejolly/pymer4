import polars as pl
from importlib import resources

__all__ = ["load_dataset"]


def load_dataset(name):
    """Loads csv file included with package as a polars DataFrame"""

    valid_names = [
        "gammas",
        "mtcars",
        "sample_data",
        "sleep",
        "titanic",
        "titanic_train",
        "titanic_test",
        "poker",
    ]

    if name not in valid_names:
        raise ValueError(f"Dataset name must be one of: {valid_names}")

    with resources.files("pymer4").joinpath(f"resources/{name}.csv") as f:
        return pl.read_csv(f)
