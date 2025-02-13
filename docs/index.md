# Pymer4: Generalized Linear & Multi-level Models in Python

[![image](https://github.com/ejolly/pymer4/actions/workflows/Tests.yml/badge.svg)](https://github.com/ejolly/pymer4/actions/workflows/Tests.yml) [![image](https://github.com/ejolly/pymer4/actions/workflows/Build.yml/badge.svg)](https://github.com/ejolly/pymer4/actions/workflows/Build.yml) [![image](https://badge.fury.io/py/pymer4.svg)](https://badge.fury.io/py/pymer4) [![image](https://anaconda.org/ejolly/pymer4/badges/version.svg)](https://anaconda.org/ejolly/pymer4) [![image](https://anaconda.org/ejolly/pymer4/badges/platforms.svg)](https://anaconda.org/ejolly/pymer4) [![image](https://pepy.tech/badge/pymer4)](https://pepy.tech/project/pymer4) [![image](http://joss.theoj.org/papers/10.21105/joss.00862/status.svg)](https://doi.org/10.21105/joss.00862) [![image](https://zenodo.org/badge/90598701.svg)](https://zenodo.org/record/1523205) ![image](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue) [![image](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](https://github.com/ejolly/pymer4/issues)

:::{admonition} `pymer4`
:class: tip
A simple Python replacement for R's `lm()`, `glm()`, `lmer()` and `glmer()`
:::

`pymer4` is a statistics library for estimating various regression models, multi-level models, and generalized-linear-mixed models in Python. Jealous of R's lovely model syntax by prefer to work in the scientific Python ecoysystem? This package has got you covered! `pymer4` provides a clean interface that hides the back-and-forth code required when moving between R and Python. This is accomplished using [rpy2](https://rpy2.github.io/doc/latest/html/index.html/) to interface between langauges.


## Why?

The scientific Python ecosystem has tons of fantastic libraries for data-analysis and statistical modeling such as `statsmodels`, `pingouin`, `scikit-learn`, and `bambi` for bayesian models to name a few. However, Python still sorely lacks a *unified formula-based modeling interface* that rivals what's available in R (and the [`tidyverse`](https://www.tidyverse.org/)) for frequentist statistics. This makes it frustrating for beginners and advanced Python analysts-alike to jump between different tools in order to accomplish a single task. So, rather than completely reinvent the wheel, `pymer4` aims to bring the best R's robust modeling capabilities to Python for the most common General(ized)-Linear-(Mixed)-Modeling (GLMMs) needs in the social and behavioral sciences. 

At the same time, `pymer4` includes numerous *quality-of-life features* for common tasks you're likely to do when working with models (e.g. automatically calculated fit statistics, residuals, p-values for mixed-models, bootstrapped confidence-intervals, random-effects deviances, etc). By bringing together functionality spread across several popular R tools, we've aimed for *intuitive-usability*. `pymer4` also notably builds on top of the [`polars`](https://docs.pola.rs/py-polars/html/reference/) Dataframe library rather than `pandas`. This keeps code simple, fast, and efficient, while opening the door for enhanced future functionality.

### Notable Features

- Generate nicely-formmated [summary tables](https://posit-dev.github.io/great-tables/)
- Easily estimate of a variety of parametric or non-parametric confidence intervals (e.g. bootstrapped)
- Easily calculate marginal predictions, estimates, and comparisons from any model
- Auto "tidy" model statistics (e.g. residuals, fit statistics, AIC)
- Auto-calculation of logit model converted odds and probabilities
- Auto p-value calculation for multi-level models using [`lmerTest`](https://cran.r-project.org/web/packages/lmerTest/index.html)
- Auto-extraction of random-effects for multi-level models
- Built upon `polars` dataframes
- Extensible design using a "functional-core imperative-shell" pattern


## Publications

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

## Citing

If you use `pymer4` in your own work, please cite:

Jolly, (2018). Pymer4: Connecting R and Python for Linear Mixed
Modeling. *Journal of Open Source Software*, 3(31), 862,
<https://doi.org/10.21105/joss.00862>
