from polars import col, Enum, String, concat, Float64, Int64

__all__ = ["join_on_common_cols", "make_factors", "unmake_factors"]


def get_str_numeric_type(x):
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
    common_cols = list(set(df1.columns).intersection(set(df2.columns)))
    result_df = concat([df1, df2.drop(common_cols)], how="horizontal")
    return result_df


def make_factors(
    df, factors_and_levels: str | dict | list, return_factor_dict: bool = False
):
    """Convert specified polars columns to categorical types 'enums' which are correctly converted to R factors"""

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
    """Convert specified polars columns from categorical types 'enums' to float types"""

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
