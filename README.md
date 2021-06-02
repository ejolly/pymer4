[![tests & build](https://github.com/ejolly/pymer4/actions/workflows/CI.yml/badge.svg)](https://github.com/ejolly/pymer4/actions/workflows/CI.yml)
[![PyPI version](https://badge.fury.io/py/pymer4.svg)](https://badge.fury.io/py/pymer4)
[![Anaconda Version](https://anaconda.org/ejolly/pymer4/badges/version.svg)](https://anaconda.org/ejolly/pymer4) 
[![Anaconda Platforms](https://anaconda.org/ejolly/pymer4/badges/platforms.svg)](https://anaconda.org/ejolly/pymer4)
[![Downloads](https://pepy.tech/badge/pymer4)](https://pepy.tech/project/pymer4)
[![DOI](http://joss.theoj.org/papers/10.21105/joss.00862/status.svg)](https://doi.org/10.21105/joss.00862)
[![DOI](https://zenodo.org/badge/90598701.svg)](https://zenodo.org/record/1523205)
![Python Versions](https://img.shields.io/badge/python-3.6%20%7C%203.7%20%7C%203.8-blue)
[![contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](https://github.com/ejolly/pymer4/issues)

# Pymer4

❗️⚠️ **[Contributors wanted](https://github.com/ejolly/pymer4/issues/61)** ⚠️❗️

Pymer4 is a statistics library for estimating various regression and multi-level models in Python. Love [lme4](https://cran.r-project.org/web/packages/lme4/index.html) in R, but prefer to work in the scientific Python ecosystem? This package has got you covered!

`pymer4` provides a clean interface that hides the back-and-forth code required when moving between R and Python. In other words, you can work completely in Python, never having to deal with R, but get (most) of lme4’s goodness. This is accomplished using [rpy2](hhttps://rpy2.github.io/doc/latest/html/index.html/) to interface between langauges.

Additionally `pymer4` can fit various additional regression models with some bells, such as robust standard errors, and two-stage regression (summary statistics) models. See the features page for more information.

**TL;DR** this package is your new simple Pythonic drop-in replacement for `lm()` or `glmer()` in R.

For example:

```python
# Assuming you have a pandas dataframe in tidy/long format
# with DV and IV columns for dependent/outcome vars and
# independent/predictor vars 

model = Lmer('DV ~ IV1 + IV2 + (IV+IV2|Group)', data=dataframe)

# Fit and print an R/statsmodels style summary 
# with t/z-tests, CIs, and p-values
model.fit()

# Access model attributes
model.BIC
model.residuals

# Get fitted parameters
model.coef # population parameters
model.fixef # group/cluster estimates (BLUPs)
model.ranef # group/cluster deviates
```

## Documentation

Check out the [documentation site](http://eshinjolly.com/pymer4/) for detailed tutorial examples, API documentation, and installation instructions!


## Installation

Installing via [Anaconda](https://www.anacnda.com/products/individual) is the preferred installation method. Follow the directions [here](https://eshinjolly.com/pymer4/installation.html).  


## Contributing

Contributions are *always welcome*!  
If you are interested in contributing feel free to check out the [open issues](https://github.com/ejolly/pymer4/issues), [development roadmap on Trello](https://trello.com/b/gGKmeAJ4), or submit pull requests for additional features or bug fixes. If you do make a pull request, please do so by forking the [development branch](https://github.com/ejolly/pymer4/tree/dev) and taking note of the [contribution guidelines](https://eshinjolly.com/pymer4/contributing.html).

## Contributors

[![](https://github.com/turbach.png?size=50)](https://github.com/turbach) 
[![](https://github.com/Shotgunosine.png?size=50)](https://github.com/Shotgunosine)
<a href="https://github.com/arose13"><img src="https://github.com/arose13.png" width="50" height="50" /></a>
