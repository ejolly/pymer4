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

If you use this software please cite as:
Jolly, (2018). Pymer4: Connecting R and Python for Linear Mixed Modeling. *Journal of Open Source Software*, 3(31), 862, https://doi.org/10.21105/joss.00862

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
:code:`pymer4` since version 0.6.0 is only compatible with Python 3. Versions 0.5.0 and lower will work with Python 2, but will not contain any new features. :code:`pymer4` also requires a working R installation with specific packages installed and it will *not* install R or these packages for you. However, you can follow either option below to easily handle these dependencies.

Option 1 (simpler but slower model fitting)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
If you don't have R installed and you use the Anaconda Python distribution simply run the following commands to have Anaconda install R and the required packages for you. This is fairly painless installation, but model fitting will be slower than if you install R and ``pymer4`` separately and configure them (option 2).

1. ``conda install -c conda-forge r r-base r-lmertest r-lsmeans rpy2``
2. ``pip install pymer4``
3. Test the installation to see if it's working by running ``python -c "from pymer4.test_install import test_install; test_install()"``
4. If there are errors follow the guide below

Option 2 (potentially trickier, but faster model fitting)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
This method assumes you already have R installed. If not install first install it from the `R Project website <https://www.r-project.org/>`_. Then complete the following steps:

1. Install the required R packages by running the following command from within R: ``install.packages(c('lme4','lmerTest','lsmeans'))``
2. Install pymer4 by running the following command in a terminal: ``pip install pymer4``
3. Test the installation to see if it's working by running the following command in a terminal: ``python -c "from pymer4.test_install import test_install; test_install()"``
4. If there are errors follow the guide below

Install issues
^^^^^^^^^^^^^^
Some users have issues installing ``pymer4`` on recent versions of macOS. This is due to compiler issues that give ``rpy2`` (a package dependency of ``pymer4``) some issues during install. Here's a fix that should work for that:

1. Install `homebrew <https://brew.sh/>`_ if you don't have it already by running the command at the link (it's a great pacakage manager for macOS). To check if you already have it, do ``which brew`` in your Terminal. If nothing pops up you don't have it.
2. Fix brew permissions: ``sudo chown -R $(whoami) $(brew --prefix)/*`` (this is **necessary** on macOS Sierra or higher (>= macOS 10.12))
3. Update homebrew ``brew update``
4. Install the xz uitility ``brew install xz``
5. At this point you can try to re-install ``pymer4`` and re-test the installation. If it still doesn't work follow the next few steps below
6. Install an updated compiler: ``brew install gcc``, or if you have homebrew already, ``brew upgrade gcc``
7. Enable the new compiler for use:

    .. code-block:: bash

        export CC="$(find `brew info gcc | grep usr | sed 's/(.*//' | awk '{printf $1"/bin"}'` -name 'x86*gcc-?')"
        export CFLAGS="-W"

8. If the above results in any error output (it should return nothing) you might need to manually find out where the new compiler is installed. To do so use ``brew info gcc`` and ``cd`` into the directory that begins with ``/usr`` in the output of that command. From there ``cd`` into ``bin`` and look for a file that begins with ``x86`` and ends with ``gcc-7``. It's possible that the directory ends with ``gcc-8`` or a higher number based on how recently you installed from homebrew. In that case, just use the latest version. Copy the *full path* to that file and run the following:

    .. code-block:: bash

        export CC= pathYouCopiedInQuotes
        export CFLAGS="-W"

9. Finally install ``rpy2`` using the new compiler you just installed: ``conda install -c conda-forge rpy2`` if you followed Option 1 above or ``pip install rpy2`` if you followed Option 2
10. Now you should be able to ``pip install pymer4`` :)

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
