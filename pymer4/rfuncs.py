import rpy2.robjects as ro
from .tidystats.bridge import ensure_r_input

__all__ = ["get_summary", "make_rfunc"]


def get_summary(r_model):
    """Get an R-style summary print out from a model as a multi-line string"""
    make_summary = make_rfunc("""
    function(model) {
    output <- capture.output(summary(model))
    return(output)
    }
    """)
    out = make_summary(r_model)
    cleaned = []
    skip_lines = True
    for line in out:
        line_str = str(line)
        if line_str.startswith("Coef") or "criterion at convergence" in line_str:
            skip_lines = False
        if not skip_lines:
            cleaned.append(line_str)
    return "\n".join(cleaned)


def make_rfunc(r_code):
    """Make a function from an R code string useable with a python object.

    Example:

        >>> # Make a function that returns the coefficients of a model
        >>> coef = make_rfunc(\"""
        >>> function(model) {
        >>> output <- coef(model)
        >>> return(output)
        >>> }
        >>> \""")
        >>>
        >>> # Use the function by passing in a model's .r_model attribute
        >>> ols = lm(mpg ~ hp + wt, data=mtcars)
        >>> ols.fit()
        >>> coefs = coef(ols.r_model)
    """

    func = ro.r(r_code)
    return ensure_r_input(func)
