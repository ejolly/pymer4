__all__ = ["save_model", "load_model", "load_dataset"]

from bambi import load_data as load_dataset


def save_model(model, filepath, compression="zlib", **kwargs):

    raise NotImplementedError("Model i/o not yet supported")


def load_model(filepath):

    raise NotImplementedError("Model i/o not yet supported")
