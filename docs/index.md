# Pymer4: Generalized Linear & Multi-level Models in Python

[![image](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](https://github.com/ejolly/pymer4/issues) [![image](https://anaconda.org/ejolly/pymer4/badges/version.svg)](https://anaconda.org/ejolly/pymer4) ![image](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)  
[![image](https://pepy.tech/badge/pymer4)](https://pepy.tech/project/pymer4) [![image](http://joss.theoj.org/papers/10.21105/joss.00862/status.svg)](https://doi.org/10.21105/joss.00862) [![image](https://zenodo.org/badge/90598701.svg)](https://zenodo.org/record/1523205)  

:::{admonition} `pymer4`
:class: tip
A simple Python replacement for R's `lm()`, `glm()`, `lmer()` and `glmer()`
:::

`pymer4` is a statistics library for estimating various regression models, multi-level models, and generalized-linear-mixed models in Python. Jealous of R's lovely model syntax by prefer to work in the scientific Python ecoysystem? This package has got you covered! `pymer4` provides a clean interface that hides the back-and-forth code required when moving between R and Python. This is accomplished using [rpy2](https://rpy2.github.io/doc/latest/html/index.html/) to interface between languages.

## Why?

The scientific Python ecosystem has tons of fantastic libraries for data-analysis and statistical modeling such as `statsmodels`, `pingouin`, `scikit-learn`, and `bambi` for bayesian models to name a few. However, Python still sorely lacks a *unified formula-based modeling interface* that rivals what's available in R. This makes it frustrating for analysts, students, and educators who typically work in Python, and requires jumping between different libraries and languages in order to accomplish a single task. Rather than completely reinventing the wheel, `pymer4` aims to bring the best of R's robust modeling capabilities to Python for the most common General(ized)-Linear-(Mixed)-Modeling (GLMMs) needs in the social and behavioral sciences. 

At the same time, `pymer4` includes numerous *quality-of-life features* for common tasks you're likely to do when working with models (e.g. automatically calculated fit statistics, residuals, p-values for mixed-models, bootstrapped confidence-intervals, random-effects deviances, etc). By bringing together functionality spread across several popular R tools, we've aimed for *intuitive-usability*. `pymer4` also notably builds on top of the [`polars`](https://docs.pola.rs/py-polars/html/reference/) Dataframe library rather than `pandas`. This keeps code simple, fast, and efficient, while opening the door for enhanced future functionality.


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
