from polars import col, Enum, String, concat, Float64, Int64
import polars as pl
import numpy as np
from string import ascii_uppercase

__all__ = ["join_on_common_cols", "make_factors", "unmake_factors"]


def get_str_numeric_type(x):
    """Get the type of a string that represents a number

    Args:
        x (str): The string to check

    Returns:
        type: The type of the string
    """
    if "." in x:
        try:
            float(x)
            return float
        except ValueError:
            return str
    else:
        try:
            int(x)
            return int
        except ValueError:
            return str


def join_on_common_cols(df1, df2):
    """Join two polars DataFrames on common columns"""
    common_cols = list(set(df1.columns).intersection(set(df2.columns)))
    result_df = concat([df1, df2.drop(common_cols)], how="horizontal")
    return result_df


def make_factors(
    df, factors_and_levels: str | dict | list, return_factor_dict: bool = False
):
    """Convert specified polars columns to categorical types 'enums' which are correctly converted to R factors

    Args:
        df (DataFrame): The DataFrame to convert
        factors_and_levels (str | dict | list): The column(s) to convert to factors and their levels
        return_factor_dict (bool, optional): Whether to return the factor dictionary. Defaults to False.

    Returns:
        DataFrame: The DataFrame with the specified columns converted to factors
    """

    if isinstance(factors_and_levels, str):
        factors_and_levels = [factors_and_levels]

    if isinstance(factors_and_levels, list):
        factors_and_levels = {
            factor: df[factor].unique().cast(String).sort().to_list()
            for factor in factors_and_levels
        }

    compound_expression = []
    for factor, levels in factors_and_levels.items():
        expression = col(factor).cast(String).cast(Enum(levels))
        compound_expression.append(expression)
    if return_factor_dict:
        return df.with_columns(*compound_expression), factors_and_levels
    else:
        return df.with_columns(*compound_expression)


def unmake_factors(df, factors: dict | None):
    """Convert specified polars columns from categorical types 'enums' to float types

    Args:
        df (DataFrame): The DataFrame to convert
        factors (dict | None, optional): The factor dictionary to use for conversion. Defaults to None.

    Returns:
        DataFrame: The DataFrame with the specified columns converted from factors to their original types
    """
    if factors is None:
        return df

    # Infer original types from factor dict
    compound_expression = []
    for factor, levels in factors.items():
        if get_str_numeric_type(levels[0]) is float:
            compound_expression.append(col(factor).cast(Float64))
        elif get_str_numeric_type(levels[0]) is int:
            compound_expression.append(col(factor).cast(Int64))
        else:
            compound_expression.append(col(factor).cast(String))
    return df.with_columns(*compound_expression)


@pl.api.register_expr_namespace("stats")
class StatsExpr:
    """Polars expression namespace for statistical functions"""

    def __init__(self, expr: pl.Expr) -> None:
        self._expr = expr

    def rnorm(self, n: int, mean: float = 0, std: float = 1):
        """Generate random numbers from a normal distribution"""
        return pl.lit(np.random.normal(mean, std, n))

    def rbinom(self, n: int, size: int, prob: float):
        """Generate random numbers from a binomial distribution"""
        return pl.lit(np.random.binomial(size, prob, n))

    def runiform(self, n: int, min: float = 0, max: float = 1):
        """Generate random numbers from a uniform distribution"""
        return pl.lit(np.random.uniform(min, max, n))

    def rgroup(self, n: int, ngroups: int, replace: bool = True):
        """Generate random numbers from a group distribution"""
        g = list(ascii_uppercase[:ngroups])
        return pl.lit(np.random.choice(g, n, replace=replace))

    def rpoisson(self, n: int, lamb: float):
        """Generate random numbers from a Poisson distribution"""
        return pl.lit(np.random.poisson(lamb, n))

    def rbeta(self, n: int, _alpha: float, _beta: float):
        """Generate random numbers from a beta distribution"""
        return pl.lit(np.random.beta(_alpha, _beta, n))

    def rgamma(self, n: int, shape: float, scale: float = 1):
        """Generate random numbers from a gamma distribution"""
        return pl.lit(np.random.gamma(shape, scale, n))

    def rchisq(self, n: int, df: float):
        """Generate random numbers from a chi-squared distribution"""
        return pl.lit(np.random.chisquare(df, n))
