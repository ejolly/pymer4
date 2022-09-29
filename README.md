[![tests & build](https://github.com/ejolly/pymer4/actions/workflows/CI.yml/badge.svg)](https://github.com/ejolly/pymer4/actions/workflows/CI.yml)
[![PyPI version](https://badge.fury.io/py/pymer4.svg)](https://badge.fury.io/py/pymer4)
[![Anaconda Version](https://anaconda.org/ejolly/pymer4/badges/version.svg)](https://anaconda.org/ejolly/pymer4) 
[![Anaconda Platforms](https://anaconda.org/ejolly/pymer4/badges/platforms.svg)](https://anaconda.org/ejolly/pymer4)
[![Downloads](https://pepy.tech/badge/pymer4)](https://pepy.tech/project/pymer4)
[![DOI](http://joss.theoj.org/papers/10.21105/joss.00862/status.svg)](https://doi.org/10.21105/joss.00862)
[![DOI](https://zenodo.org/badge/90598701.svg)](https://zenodo.org/record/1523205)
![Python Versions](https://img.shields.io/badge/python-3.6%20%7C%203.7%20%7C%203.8-blue)
[![contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](https://github.com/ejolly/pymer4/issues)

# Pymer4 with Bambi/PyMC Backend
This branch uses [Bambi](https://bambinos.github.io/bambi/main/index.html) as a backend for `Lmer` models instead of `lme4` in `R`. Bambi is itself layer of abstraction on top of the [PyMC probabilistic inference toolbox](https://www.pymc.io/welcome.html). The appeal of going with Bambi instead of PyMC directly is because it offers a lot of "primitives" that would otherwise be tedious to rewrite, e.g.
- a formula-based modeling building interface
- automatic calculation of "weakly informative" priors scaled to the data <- this alone is a huge benefit for adoption making comparisons and the transition between frequentist/maximum-likelihood  bayesian estimation much easier
- support for categorical predictors
- tight integration with [Xarray](https://docs.xarray.dev/en/stable/) data structures and the [Arviz](https://arviz-devs.github.io/arviz/) plotting library

 The current implementation is a proof-of-concept and defaults to the [numpyro NUTS sampler](https://num.pyro.ai/en/stable/) powered by [jax](https://jax.readthedocs.io/en/latest/) as it seems to be the fastest without relying on variational inference.

# Goal
The goal of this branch is to offer a complete version of `pymer4` with **no R dependencies** and **minimal changes** to the current API. Overtime this would
- dramatically reduce the maintenance burden for `pymer4`
- offer and much simpler installation path for users, e.g. `pip` only installation with no `conda` no longer required
- create new opportunities for integrating `pymer4` with other technologies, e.g.
  - embedding in websites with [Pyodide](https://pyodide.org/en/stable/);  PyMC is already [compatible](https://www.pymc.io/welcome.html) 
  - speeding up inference with [JIT GPU compilation](https://jax.readthedocs.io/en/latest/notebooks/quickstart.html)  
- present the opportunity for additional community maintainer of the current R-backend for `pymer4` 

However, reaching feature parity with the release version of pymer4 (e.g. a variety of automatic contrasts computations, post-hoc tests, etc) will likely take sometime and happen gradually. Until then, releases on the [main branch](https://github.com/ejolly/pymer4) or `pypi` will continue to be powered by `R`. 

## Contributing

Contributions are *always welcome*!  
If you are interested in contributing feel free to check out the [open issues](https://github.com/ejolly/pymer4/issues), [development roadmap on Trello](https://trello.com/b/gGKmeAJ4), or submit pull requests for additional features or bug fixes. If you do make a pull request, please do so by forking the [development branch](https://github.com/ejolly/pymer4/tree/dev) and taking note of the [contribution guidelines](https://eshinjolly.com/pymer4/contributing.html).

## Contributors

[![](https://github.com/turbach.png?size=50)](https://github.com/turbach) 
[![](https://github.com/Shotgunosine.png?size=50)](https://github.com/Shotgunosine)
<a href="https://github.com/arose13"><img src="https://github.com/arose13.png" width="50" height="50" /></a>
