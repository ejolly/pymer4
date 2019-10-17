[![Build Status](https://travis-ci.org/ejolly/pymer4.svg?branch=master)](https://travis-ci.org/ejolly/pymer4)
[![PyPI version](https://badge.fury.io/py/pymer4.svg)](https://badge.fury.io/py/pymer4)
![Python Versions](https://img.shields.io/badge/python-3.6%20%7C%203.7-blue)
[![DOI](http://joss.theoj.org/papers/10.21105/joss.00862/status.svg)](https://doi.org/10.21105/joss.00862)
[![DOI](https://zenodo.org/badge/90598701.svg)](https://zenodo.org/record/1523205)

# Pymer4

Pymer4 is a statistics library for estimating various regression and multi-level models in Python. Love [lme4](https://cran.r-project.org/web/packages/lme4/index.html) in R, but prefer to work in the scientific Python ecosystem? This package has got you covered!

`pymer4` provides a clean interface that hides the back-and-forth code required when moving between R and Python. In other words, you can work completely in Python, never having to deal with R, but get (most) of lme4’s goodness. This is accomplished using [rpy2](hhttps://rpy2.github.io/doc/latest/html/index.html/) to interface between langauges.

Additionally `pymer4` can fit various additional regression models with some bells, such as robust standard errors, and two-stage regression (summary statistics) models. See the features page for more information.

**TL;DR** this package is your new simple Pythonic drop-in replacement for `lm()` or `glmer()` in R.

Tutorial examples, API documentation, and installation instructions can be found at the [documentation site](http://eshinjolly.com/pymer4/).
