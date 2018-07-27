.. pymer4 documentation master file, created by
   sphinx-quickstart on Tue Oct 17 12:08:49 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Pymer4
======
Love multi-level-modeling using `lme4  <https://cran.r-project.org/web/packages/lme4/index.html>`_ in R, but prefer to work in the scientific Python ecosystem? This package has got you covered! It's a small convenience package wrapping the basic functionality of lme4 for compatibility with python.

This package's main purpose is to provide a clean interface that hides the back-and-forth code required when moving between R and Python. In other words a user can work completely in Python, never having to deal with R, but get (most) of lme4's goodness. Behind the scenes this package simply uses `rpy2 <https://rpy2.readthedocs.io/en/version_2.8.x/>`_ to pass objects between languages, compute what's needed, parse everything, and convert to Python types (e.g. numpy arrays, pandas dataframes, etc).

This package can also fit standard regression models with a few extra bells and whistles compared to R's :code:`lm()` (*Currently this only includes linear models*)

TL;DR This package is your new *simple* Pythonic drop-in replacement for :code:`lm()` or :code:`glmer()` in R.

`Github Repo Source <https://github.com/ejolly/pymer4>`_

Features
--------
This package has some extra goodies to make life a bit easier, namely:

- For multi-level models (i.e. :code:`glmer()`):

    - Automatic inclusion of p-values in model output using `lmerTest <https://cran.r-project.org/web/packages/lmerTest/index.html>`_
    - Automatic inclusion of confidence intervals in model output
    - Automatic conversion and calculation of *odds-ratios* and *probabilities* for logit models
    - Easy access to model fits, residuals, and random effects as pandas dataframes
    - Random effects plotting using seaborn
    - Easy post-hoc tests with multiple-comparisons correction via `lsmeans <https://cran.r-project.org/web/packages/lsmeans/index.html>`_
    - Easy model predictions on new data
    - Easy generation of new data from a fitted model
    - Optional p-value computation via within cluster permutation testing (experimental)

- For standard linear models (i.e. :code:`lm()`)

    - Automatic inclusion of confidence intervals in model output
    - Easy computation of empirically bootstrapped 95% confidence intervals
    - Easy computation of cluster-robust, heteroscedasticity-robust or auto-correlation-robust, 'sandwich estimators' for standard errors (*note: these are not the same as auto-regressive models*)
    - Permutation tests on model parameters

- Data simulation

    - Highly customizable functions for generating data useful for standard regression models and multi-level models

- Data visualization

    - Convenience methods for plotting model estimates, including random-effects terms

Installation
------------
Requires a working installation of *both* Python (2.7 or 3.6) and R (>= 3.2.4).

You will also need the :code:`lme4`, :code:`lmerTest`, and :code:`lsmeans` R packages installed.

*This package will not install R or R packages for you!*

1. Method 1 - Install from PyPi (stable)

.. code-block:: python

    pip install pymer4

2. Method 2 - Install from github (latest)

.. code-block:: python

    pip install git+https://github.com/ejolly/pymer4

Install issues
^^^^^^^^^^^^^^
Some users have issues installing ``pymer4`` on recent versions of macOS. This is due to compiler issues that give ``rpy2`` (a package dependency of ``pymer4``) some issues during install. Here's a fix that should work for that:

1. Install `homebrew <https://brew.sh/>`_ if you don't have it already by running the command at the link (it's a great pacakage manager for macOS). To check if you already have it, do ``which brew`` in your Terminal. If nothing pops up you don't have it.
2. Fix brew permissions: ``sudo chown -R $(whoami) $(brew --prefix)/*`` (this is **necessary** on macOS Sierra or higher (>= macOS 10.12))
3. Update homebrew ``brew update``
4. Install an updated compiler: ``brew install gcc``, or if you have homebrew already, ``brew upgrade gcc``
5. Enable the new compiler for use:

    .. code-block:: bash

        export CC="$(find `brew info gcc | grep usr | sed 's/(.*//' | awk '{printf $1"/bin"}'` -name 'x86*gcc-7')"
        export CFLAGS="-W"

6. If the above results in any error output (it should return nothing) you might need to manually find out where the new compiler is installed. To do so use ``brew info gcc`` and ``cd`` into the directory that begins with ``/usr`` in the output of that command. From there ``cd`` into ``bin`` and look for a file that begins with ``x86`` and ends with ``gcc-7``. Copy the *full path* to that file and run the following:

    .. code-block:: bash

        export CC= pathYouCopiedInQuotes
        export CFLAGS="-W"

7. Finally install ``rpy2`` using the new compiler you just installed: ``pip install rpy2==2.8.5``
8. Now you should be able to ``pip install pymer4``:)

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

Post-hoc Comparisons
--------------------
.. toctree::
    :maxdepth: 2

    post_hoc

Simulating Data
---------------
.. toctree::
    :maxdepth: 2

    simulate

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
