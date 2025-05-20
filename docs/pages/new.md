# What's New

## 0.9.0
This is release is a **major overhaul** of `pymer4` that involved a near complete re-write of internals to faciliate future maintenance and integrate more advanced functionality from additional R libraries. This is a **backwards incompatible** release with numerous dependency and **API breaking changes.** You can check out the [migration guide](./migrating.md) for more details. The `0.8.x` version line is still available on github for community contributions/maintenance if desired.

As of this version, `pymer4` is now *only* installable via `conda` following the instructions [here](./installation.md) using the `ejolly` channel. We expect to move to the `conda-forge` channel soon.

### Summary of changes
- Adoption of `polars` as dataframe "backend"
- New consistent API that supports `lm`, `glm`, `lmer` and `glmer` models
- Full support for factors and marginal-estimates/comparisons for all model types
- Overhauled docs and tutorials with several included datasets for demo and teaching purposes
- Much more extensive testing
- Replaced bespoke code with "battle-tested" implementations of helper and utility functions in different R libaries (e.g. `broom`, `insight`, `parameters`)
- Switched `setup.py` and `requirements.txt/requirements-dev.txt` to `pyproject.toml`
- Switched over to [Pixi.sh](../contributing/developing.md#development-tooling) for development-tooling and Github Actions
- Switched documentation from `sphinx` to `jupyterbook`
- Switched project linting from `black` to `ruff`
- Exclusively create `noarch` builds for conda
  - Currently implemented as a `pixi task` using `conda build`, with plans to switch to [`pixi build` when ready](https://pixi.sh/latest/build/python/)

### Encompassed Fixes
- [#61](https://github.com/ejolly/pymer4/issues/61)
- [#71](https://github.com/ejolly/pymer4/issues/71)
- [#99](https://github.com/ejolly/pymer4/issues/99)
- [#102](https://github.com/ejolly/pymer4/issues/102)
- [#124](https://github.com/ejolly/pymer4/issues/124)
- [#130](https://github.com/ejolly/pymer4/issues/130)
- [#132](https://github.com/ejolly/pymer4/issues/132)
- [#134](https://github.com/ejolly/pymer4/issues/134)
- [#139](https://github.com/ejolly/pymer4/pull/139)

### Planned features
These following features are planned for upcoming versions in `0.9.x` line

- support for `lmerControl` options
- automatic convertion to `scikit-learn` compatible estimators via `.to_sklearn()`
- simulation and power modules

## 0.8.2

### Fixes
- Issue in `LogisticRegression` API name change

## 0.8.1

### Compatibility Updates
- This version includes a `noarch` build that should be installable on arm-based macOS platforms (e.g. M1, M2, etc)
- This version drops support for Python 3.7 and adds support for 3.9-3.11

### Breaking changes 

- This version also uses `joblib` for model saving and loading and drops supported hdf5 files previously handled with the `deepdish` library as it is no longer actively maintained. This means that 0.8.1 will **not** be able to load modelssaved with earlier versions of `pymer4`!

###  Fixes
- [#119](https://github.com/ejolly/pymer4/issues/119)
- [#122](https://github.com/ejolly/pymer4/issues/122)
- [#125](https://github.com/ejolly/pymer4/issues/125)

## 0.8.0

### NOTE

-   there was no 0.7.9 release as there were enough major changes to warrant a new minor release version
-   this version unpins the maximum versions of `rpy2` and `pandas`
-   if there are install issues with the `conda` release accompanying this version you should be able to successfully install into a conda environment using pip with the following: `conda install 'r-lmerTest' 'r-emmeans' rpy2 -c conda-forge` followed by `pip install pymer4`

### Bug fixes
- fixed as issue where `Lmer` with `family='binomial'` was not converting logits into probabilities correctly
- fixes [#79](https://github.com/ejolly/pymer4/issues/79)
- fixes [#88](https://github.com/ejolly/pymer4/issues/88)
- fixes [#113](https://github.com/ejolly/pymer4/issues/113)
- fixes [#114](https://github.com/ejolly/pymer4/issues/114)
- generally more robust conversion of R types to pandas

### New features

- `Lm` models now support `family='binomial'` and uses the `LogisticRegression` class from scikit-learn with no
regularization for estimation. Estimates and errors have been verified against the `glm` implementation in R
- new `lrt` function for estimating likelihood-ratio tests between `Lmer` models thanks to[@dramanica](https://github.com/dramanica). This replicates the functionality of `anova()` in R for `lmer` models. 
- new `.confint()` method for `Lmer` models thanks to [@dramanica](https://github.com/dramanica). This allows
computing confidence intervals on 1 or more paramters of an already fit model including random effects which are not computed by default when calling `.fit()`

## 0.7.8

- Maintenance release that pins `rpy2 >= 3.4.5,< 3.5.1` due to R to Python dataframe conversion issue on recent `rpy2` versions that causes a [recursion error](https://github.com/rpy2/rpy2/issues/866).
- Pending code changes to support `rpy2 >= 3.5.1` are tracked on [this development branch](https://github.com/ejolly/pymer4/tree/dev_rpy2_3.5.1).
- **Upcoming releases will drop support for** `rpy2 < 3.5.X`
- Clearer error message when making circular predictions using `Lmer` models

## 0.7.7

- This version is identical to 0.7.6 but supports `R >= 4.1`
- Installation is also more flexible and includes instructions for using `conda-forge` and optimized libraries (MKL) for Intel CPUs

# 0.7.6

-   

    **Bug fixes:**

    :   -   fixes an issue in which a `Lmer` model fit using categorical
            predictors would be unable to use `.predict` or would return
            fitted values instead of predictions on new data. random
            effect and fixed effect index names were lost thanks to
            Mario Leaonardo Salinas for discovering this issue

# 0.7.5

-   This version is identical to 0.7.4 and simply exists because a
    naming conflict that resulted in a failed released to Anaconda
    cloud. See release notes for 0.7.4 below

# 0.7.4

-   

    **Compatibility updates:**

    :   -   This version drops official support for Python 3.6 and adds
            support for Python 3.9. While 3.6 should still work for the
            most part, development support and testing against this
            version of Python will no longer continue moving forward.

-   

    **New features:**

    :   -   `utils.result_to_table` function nicely formats the
            `model.coefs` output for a fitted model. The docstring also
            contains instructions on using this in conjunction with the
            [gspread-pandas](https://github.com/aiguofer/gspread-pandas)
            library for \"exporting\" model results to a google sheet

# 0.7.3

-   

    **Bug fixes:**

    :   -   fix issue in which random effect and fixed effect index
            names were lost thanks to
            [\@jcheong0428](https://github.com/jcheong0428) and
            [\@Shotgunosine](https://github.com/Shotgunosine) for the
            quick PRs!

# 0.7.2

-   

    **Bug fixes:**

    :   -   fix bug in which `boot_func` would fail iwth `y=None` and
            `paired=False`

-   

    **Compatibility updates:**

    :   -   add support for `rpy2>=3.4.3` which handles model matrices
            differently
        -   pin maximum `pandas<1.2`. This is neccesary until our other
            dependency `deepdish` adds support. See [this
            issue](https://github.com/uchicago-cs/deepdish/issues/45)

# 0.7.1

-   

    **Pymer4 will be on conda as of this release!**

    :   -   install with
            `conda install -c ejolly -c defaults -c conda-forge pymer4`
        -   This should make installation much easier
        -   Big thanks to [Tom
            Urbach](https://turbach.github.io/toms_kutaslab_website/)
            for assisting with this!

-   

    **Bug fixes:**

    :   -   design matrix now handles rfx only models properly
        -   compatibility with the latest version of pandas and rpy2 (as
            of 08/20)
        -   `Lmer.residuals` now save as numpy array rather than
            `R FloatVector`

-   

    **New features:**

    :   -   `stats.tost_equivalence` now takes a `seed` argument for
            reproducibility

-   

    **Result Altering Change:**

    :   -   Custom contrasts in `Lmer` models are now expected to be
            specified in *human readable* format. This should be more
            intuitive for most users and is often what users expect from
            R itself, even though that\'s not what it actually does! R
            expects custom contrasts passed to the `contrasts()`
            function to be the *inverse* of the desired contrasts. See
            [this
            vignette](https://rstudio-pubs-static.s3.amazonaws.com/65059_586f394d8eb84f84b1baaf56ffb6b47f.html)
            for more info.
        -   In `Pymer4`, specifying the following contrasts:
            `model.fit(factors = {"Col1": {'A': 1, 'B': -.5, 'C': -.5}}))`
            will estimate the difference between A and the mean of B and
            C as one would expect. Behind the scenes, `Pymer4` is
            performing the inversion operation automatically for R.

-   Lots of other devops changes to make testing, bug-fixing,
    development, future releases and overall maintenance much easier.
    Much of this work has been off-loaded to automated testing and
    deployment via Travis CI.

# 0.7.0

-   **dropped support for versions of** `rpy2 < 3.0`

-   **Result Altering Change:** `Lm` standard errors are now computed
    using the square-root of the adjusted mean-squared-error
    `(np.sqrt(res.T.dot(res) / (X.shape[0] - X.shape[1])))` rather than
    the standard deviation of the residuals with DOF adjustment
    `(np.std(res, axis=0, ddof=X.shape[1]))`. While these produce the
    same results if an intercept is included in the model, they differ
    slightly when an intercept is not included. Formerly in the
    no-intercept case, results from pymer4 would differ slightly from R
    or statsmodels. This change ensures the results are always identical
    in all cases.

-   **Result Altering Change:** `Lm` rsquared and adjusted rsquared now
    take into account whether an intercept is included in the model
    estimation and adjust accordingly. This is consistent with the
    behavior of R and statsmodels

-   **Result Altering Change:** hc1 is the new default robust estimator
    for `Lm` models, changed from hc0

-   **API change:** all model residuals are now saved in the
    `model.residuals` attribute and were formerly saved in the
    `model.resid` attribute. This is to maintain consistency with
    `model.data` column names.

-   **New feature:** addition of `pymer4.stats` module for various
    parametric and non-parametric statistics functions (e.g. permutation
    testing and bootstrapping)

-   **New feature:** addition of `pymer4.io` module for saving and
    loading models to disk

-   **New feature:** addition of `Lm2` models that can perform
    multi-level modeling by first estimating a separate regression for
    each group and then performing inference on those estimates. Can
    perform inference on first-level semi-partial and partial
    correlation coefficients instead of betas too.

-   **New feature:** All model classes now have the ability to rank
    transform data prior to estimation, see the rank argument of their
    respective `.fit()` methods.

-   

    **New features for Lm models:**

    :   -   `Lm` models can transform coefficients to partial or
            semi-partial correlation coefficients
        -   `Lm` models can also perform weight-least-squares (WLS)
            regression given the weights argument to `.fit()`, with
            optional dof correction via Satterthwaite approximation.
            This is useful for categorical (e.g. group) comparison where
            one does not want to assume equal variance between groups
            (e.g. Welch\'s t-test). This remains an experimental feature
        -   `Lm` models can compute hc1 and hc2 robust standard errors

-   **New documentation look:** the look and feel of the docs site has
    been completely changed which should make getting information much
    more accessible. Additionally, overview pages have now been turned
    into downloadable tutorial jupyter notebooks

-   All methods/functions capable of parallelization now have their
    default `n_jobs` set to 1 (i.e. no default parallelization)

-   Various bug fixes to all models

-   Automated testing on travis now pins specific r and r-package
    versions

-   Switched from lsmeans to emmeans for post-hoc tests because lsmeans
    is deprecated

-   Updated interactions with rpy2 api for compatibility with version 3
    and higher

-   Refactored package layout for easier maintainability

# 0.6.0

-   **Dropped support for Python 2**
-   upgraded `rpy2` dependency version
-   Added conda installation instructions
-   Accepted [JOSS](https://joss.theoj.org/) version

# 0.5.0

-   `Lmer` models now support all generalized linear model family types
    supported by lme4 (e.g. poisson, gamma, etc)
-   `Lmer` models now support ANOVA tables with support for
    auto-orthogonalizing factors using the `.anova()` method
-   Test statistic inference for `Lmer` models can now be performed via
    non-parametric permutation tests that shuffle observations within
    clusters
-   `Lmer.fit(factors={})` arguments now support custom arbitrary
    contrasts
-   New forest plots for visualizing model estimates and confidence
    intervals via the `Lmer.plot_summary()` method
-   More comprehensive documentation with examples of new features
-   Submission to [JOSS](https://joss.theoj.org/)

# 0.4.0

-   Added `.post_hoc()` method to `Lmer` models
-   Added `.simulate()` method to `Lmer` models
-   Several bug fixes for Python 3 compatibility

# 0.3.2

-   addition of `simulate` module

# 0.2.2

-   Official pyipi **release**

# 0.2.1

-   Support for standard linear regression models
-   Models include support for robust standard errors, boot-strapped
    CIs, and permuted inference

# 0.2.0

-   Support for categorical predictors, model predictions, and model
    plots

# 0.1.0

-   Linear and Logit multi-level models
