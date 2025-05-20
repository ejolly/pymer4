from polars import Expr, col

__all__ = ["rank", "center", "scale", "zscore", "logit2odds", "logit2prob"]


def rank(expr: Expr | str, method="average", descending=False) -> Expr:
    """Rank the values in the expression

    Args:
        expr (Expr | str): The expression to rank
        method (str, optional): The method to use for ranking. Defaults to "average".
        descending (bool, optional): Whether to rank in descending order. Defaults to False.

    """
    if isinstance(expr, str):
        expr = col(expr)
    return expr.rank(method, descending=descending)


def center(expr: Expr | str) -> Expr:
    """Mean center the values in the expression

    Args:
        expr (Expr | str): The expression to center

    """
    if isinstance(expr, str):
        expr = col(expr)
    return expr - expr.mean()


def scale(expr: Expr | str) -> Expr:
    """Scale the values in the expression by their standard deviation

    Args:
        expr (Expr | str): The expression to scale

    """
    if isinstance(expr, str):
        expr = col(expr)
    return expr / expr.std()


def zscore(expr: Expr | str) -> Expr:
    """Z-score the values in the expression

    Args:
        expr (Expr | str): The expression to z-score

    """
    if isinstance(expr, str):
        expr = col(expr)
    return (expr - expr.mean()) / expr.std()


def logit2odds(expr: Expr | str) -> Expr:
    """Convert logits to log-odds

    Args:
        expr (Expr | str): The expression to convert

    """
    if isinstance(expr, str):
        expr = col(expr)
    return expr.exp()


def logit2prob(expr: Expr | str) -> Expr:
    """Convert logits to probabilities

    Args:
        expr (Expr | str): The expression to convert

    """
    if isinstance(expr, str):
        expr = col(expr)
    return expr.exp() / (1 + expr.exp())
