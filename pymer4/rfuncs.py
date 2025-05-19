import rpy2.robjects as ro

__all__ = ["get_summary"]


def get_summary(r_model):
    """Get an R-style summary print out from a model as a multi-line string"""
    func = ro.r("""
    function(model) {
    output <- capture.output(summary(model))
    return(output)
    }
    """)
    out = func(r_model)
    cleaned = []
    skip_lines = True
    for line in out:
        line_str = str(line)
        if line_str.startswith("Coef") or "criterion at convergence" in line_str:
            skip_lines = False
        if not skip_lines:
            cleaned.append(line_str)
    return "\n".join(cleaned)
