.. pymer4 documentation master file, created by
   sphinx-quickstart on Tue Oct 17 12:08:49 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Pymer4
======
Love multi-level-modeling using `lme4  <https://cran.r-project.org/web/packages/lme4/index.html>`_ in R, but prefer to work in the scientific Python ecosystem? This this package has got you covered! It's a small convenience package wrapping the basic functionality of lme4 for compatibility with python. (*Currently this only include linear and logistic multi-level models*)

This package's main purpose is to provide a clean interface that hides the back-and-forth code required when moving between R and Python. In other words a user can work completely in Python, never having to deal with R, but get (most) of lme4's goodness. Behind the scenes this package simply uses `rpy2 <https://rpy2.readthedocs.io/en/version_2.8.x/>`_ to pass objects between languages, compute what's needed, parse everything, and convert to Python types (e.g. numpy arrays, pandas dataframes, etc).

This package can also fit standard regression models with a few extra bells and whistles compared to R's :code:`lm()` (*Currently this only includes linear models*)

TL;DR This package is your new *simple* Pythonic drop-in replacement for :code:`lm()` or :code:`glmer()` in R.



Features
--------
This package has some extra goodies to make life a bit easier, namely:

- For multi-level models (i.e. :code:`glmer()`):

    - Automatic inclusion of p-values in model output using `lmerTest <https://cran.r-project.org/web/packages/lmerTest/index.html>`_
    - Automatic inclusion of confidence intervals in model output
    - Automatic conversion and calculation of *odds-ratios* and *probabilities* for logit models
    - Easy access to model fits, residuals, and random effects as pandas dataframes
    - Random effects plotting using seaborn

- For standard linear models (i.e. :code:`lm()`)

    - Automatic inclusion of confidence intervals in model output
    - Easy computation of empirically bootstrapped 95% confidence intervals
    - Easy computation of heteroscedasticity or auto-correlation robust 'sandwich estimators' for standard errors (*note: these are not the same as auto-regressive models*)
    - Permutation tests on model parameters


Installation
------------
Installation requires a working installation of *both* Python (currently not compatible with Python 3) and R installed.

1. Method 1 - Install from github (Recommended)

.. code-block:: python

    pip install git+https://github.com/ejolly/pymer4

2. Method 2 - Install from PyPi (Not currently working)

.. code-block:: python

    pip install pymer4

Basic Usage Guide
-----------------
.. toctree::
    :maxdepth: 2

    usage

Categorical Predictors
----------------------
.. toctree::
    :maxdepth: 2

    categorical

Lme4 CheatSheet
---------------
.. toctree::
    :maxdepth: 2

    rfx_cheatsheet

API
-----
.. toctree::
    :maxdepth: 2

    api
