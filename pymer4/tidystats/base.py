from rpy2.robjects.packages import importr

__all__ = ["summary"]

lib_base = importr("base")


def summary(arg):
    """Produce a summary of the results. Currently unused.

    Args:
        arg (object): The object to summarize

    Returns:
        object: An R-type of the summarized object
    """
    return lib_base.summary(arg)
