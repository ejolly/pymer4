# Pymer4

[![image](https://github.com/ejolly/pymer4/actions/workflows/Tests.yml/badge.svg)](https://github.com/ejolly/pymer4/actions/workflows/Tests.yml) [![image](https://github.com/ejolly/pymer4/actions/workflows/Build.yml/badge.svg)](https://github.com/ejolly/pymer4/actions/workflows/Build.yml) [![image](https://badge.fury.io/py/pymer4.svg)](https://badge.fury.io/py/pymer4) [![image](https://anaconda.org/ejolly/pymer4/badges/version.svg)](https://anaconda.org/ejolly/pymer4) [![image](https://anaconda.org/ejolly/pymer4/badges/platforms.svg)](https://anaconda.org/ejolly/pymer4) [![image](https://pepy.tech/badge/pymer4)](https://pepy.tech/project/pymer4) [![image](http://joss.theoj.org/papers/10.21105/joss.00862/status.svg)](https://doi.org/10.21105/joss.00862) [![image](https://zenodo.org/badge/90598701.svg)](https://zenodo.org/record/1523205) ![image](https://img.shields.io/badge/python-3.8%20%7C%203.9%20%7C%203.10%20%7C%203.11-blue) [![image](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](https://github.com/ejolly/pymer4/issues)

`pymer4` is a statistics library for estimating various regression and
multi-level models in Python. Love
[lme4](https://cran.r-project.org/web/packages/lme4/index.html) in R,
but prefer to work in the scientific Python ecosystem? This package has
got you covered!

`pymer4` provides a clean interface that hides the back-and-forth code
required when moving between R and Python. In other words, you can work
completely in Python, never having to deal with R, but get (most) of
lme4\'s goodness. This is accomplished using
[rpy2](https://rpy2.github.io/doc/latest/html/index.html/) to interface
between langauges.

Additionally `pymer4` can fit various additional regression models with
some bells, such as robust standard errors, and two-stage regression
(summary statistics) models. See the features page for more information.

**TL;DR** This package is your new *simple* Pythonic drop-in replacement
for `lm()` or `glmer()` in R.

For an example of what\'s possible check out the tutorials or [this blog
post](https://eshinjolly.com/2019/02/18/rep_measures/) comparing
different modeling strategies for clustered/repeated-measures data.

# Publications

`pymer4` has been used to analyze data is several publications including
but not limited to:

-   Jolly, E., Sadhukha, S., & Chang, L.J. (in press). Custom-molded
    headcases have limited efficacy in reducing head motion during
    naturalistic fMRI expreiments. *NeuroImage*.
-   Sharon, G., Cruz, N. J., Kang, D. W., et al. (2019). Human gut
    microbiota from autism spectrum disorder promote behavioral symptoms
    in mice. *Cell*, 177(6), 1600-1618.
-   Urbach, T. P., DeLong, K. A., Chan, W. H., & Kutas, M. (2020). An
    exploratory data analysis of word form prediction during
    word-by-word reading. *Proceedings of the National Academy of
    Sciences*, 117(34), 20483-20494.
-   Chen, P. H. A., Cheong, J. H., Jolly, E., Elhence, H., Wager, T. D.,
    & Chang, L. J. (2019). Socially transmitted placebo effects. *Nature
    Human Behaviour*, 3(12), 1295-1305.

# Citing

If you use `pymer4` in your own work, please cite:

Jolly, (2018). Pymer4: Connecting R and Python for Linear Mixed
Modeling. *Journal of Open Source Software*, 3(31), 862,
<https://doi.org/10.21105/joss.00862>
