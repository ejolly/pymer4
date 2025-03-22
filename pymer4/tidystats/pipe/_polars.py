from polars import DataFrame
from ._Pipe import pipeable


@pipeable(DataFrame)
def head(df, *args, **kwargs):
    return df.head(*args, **kwargs)


@pipeable(DataFrame)
def tail(df, *args, **kwargs):
    return df.tail(*args, **kwargs)


@pipeable(DataFrame)
def select(df, *cols, **kwargs):
    return df.select(*cols, **kwargs)


@pipeable(DataFrame)
def with_columns(df, *args, **kwargs):
    return df.with_columns(*args, **kwargs)


@pipeable(DataFrame)
def mutate(df, *args, **kwargs):
    """Alias for .with_columns"""
    return df.with_columns(*args, **kwargs)


@pipeable(DataFrame)
def filter(df, *exprs, **kwargs):
    return df.filter(*exprs, **kwargs)


@pipeable(DataFrame)
def sort(df, *args, **kwargs):
    return df.sort(*args, **kwargs)


@pipeable(DataFrame)
def arrange(df, *args, **kwargs):
    """Alias for .sort"""
    return df.sort(*args, **kwargs)


@pipeable(DataFrame)
def unique(df, *args, **kwargs):
    return df.unique(*args, **kwargs)


@pipeable(DataFrame)
def distinct(df, *args, **kwargs):
    """Alias for .unique"""
    return df.unique(*args, **kwargs)


@pipeable(DataFrame)
def group_by(df, *args, **kwargs):
    return df.group_by(*args, **kwargs, maintain_order=True)


@pipeable(DataFrame)
def summarize(df, *args, **kwargs):
    return df.agg(*args, **kwargs)
