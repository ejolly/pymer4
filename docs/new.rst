What's New
==========
Historically :code:`pymer4` versioning was a bit all over the place but has settled down since 0.5.0. This page includes the most notable updates between versions but github is the best place to checkout more details and `releases <https://github.com/ejolly/pymer4/releases/>`_.

0.7.7
-----
- This version is identical to 0.7.6 but supports :code:`R >= 4.1`
- Installation is also more flexible and includes instructions for using :code:`conda-forge` and optimized libraries (MKL) for Intel CPUs

0.7.6
-----
- **Bug fixes:**
    - fixes an issue in which a :code:`Lmer` model fit using categorical predictors    would be unable to use :code:`.predict` or would return fitted values instead of    predictions on new data. random effect and fixed effect index names were lost thanks to Mario Leaonardo Salinas for discovering this issue
    
0.7.5
-----
- This version is identical to 0.7.4 and simply exists because a naming conflict that resulted in a failed released to Anaconda cloud. See release notes for 0.7.4 below

0.7.4
-----
- **Compatibility updates:**
    - This version drops official support for Python 3.6 and adds support for Python 3.9. While 3.6 should still work for the most part, development support and testing against this version of Python will no longer continue moving forward.
- **New features:**  
    - :code:`utils.result_to_table` function nicely formats the :code:`model.coefs` output for a fitted model. The docstring also contains instructions on using this in conjunction with the `gspread-pandas <https://github.com/aiguofer/gspread-pandas>`_ library for "exporting" model results to a google sheet

0.7.3
-----
- **Bug fixes:**
    - fix issue in which random effect and fixed effect index names were lost thanks to `@jcheong0428 <https://github.com/jcheong0428>`_ and `@Shotgunosine <https://github.com/Shotgunosine>`_ for the quick PRs!

0.7.2
-----
- **Bug fixes:**  
    - fix bug in which :code:`boot_func` would fail iwth :code:`y=None` and :code:`paired=False`
- **Compatibility updates:**  
    - add support for :code:`rpy2>=3.4.3` which handles model matrices differently
    - pin maximum :code:`pandas<1.2`. This is neccesary until our other dependency :code:`deepdish` adds support. See `this issue <https://github.com/uchicago-cs/deepdish/issues/45>`_

0.7.1
-----
- **Pymer4 will be on conda as of this release!**
    - install with :code:`conda install -c ejolly -c defaults -c conda-forge pymer4`
    - This should make installation much easier
    - Big thanks to `Tom Urbach <https://turbach.github.io/toms_kutaslab_website/>`_ for assisting with this!
- **Bug fixes:**  
    - design matrix now handles rfx only models properly
    - compatibility with the latest version of pandas and rpy2 (as of 08/20)
    - :code:`Lmer.residuals` now save as numpy array rather than :code:`R FloatVector`
- **New features:**  
    - :code:`stats.tost_equivalence` now takes a :code:`seed` argument for reproducibility
- **Result Altering Change:**
    - Custom contrasts in :code:`Lmer` models are now expected to be specified in *human readable* format. This should be more intuitive for most users and is often what users expect from R itself, even though that's not what it actually does! R expects custom contrasts passed to the :code:`contrasts()` function to be the *inverse* of the desired contrasts. See `this vignette <https://rstudio-pubs-static.s3.amazonaws.com/65059_586f394d8eb84f84b1baaf56ffb6b47f.html>`_ for more info. 
    - In :code:`Pymer4`, specifying the following contrasts: :code:`model.fit(factors = {"Col1": {'A': 1, 'B': -.5, 'C': -.5}}))` will estimate the difference between A and the mean of B and C as one would expect. Behind the scenes, :code:`Pymer4` is performing the inversion operation automatically for R. 
- Lots of other devops changes to make testing, bug-fixing, development, future releases and overall maintenance much easier. Much of this work has been off-loaded to automated testing and deployment via Travis CI.


0.7.0
-----
- **dropped support for versions of** :code:`rpy2 < 3.0`
- **Result Altering Change:** :code:`Lm` standard errors are now computed using the square-root of the adjusted mean-squared-error :code:`(np.sqrt(res.T.dot(res) / (X.shape[0] - X.shape[1])))` rather than the standard deviation of the residuals with DOF adjustment :code:`(np.std(res, axis=0, ddof=X.shape[1]))`. While these produce the same results if an intercept is included in the model, they differ slightly when an intercept is not included. Formerly in the no-intercept case, results from pymer4 would differ slightly from R or statsmodels. This change ensures the results are always identical in all cases.
- **Result Altering Change:** :code:`Lm` rsquared and adjusted rsquared now take into account whether an intercept is included in the model estimation and adjust accordingly. This is consistent with the behavior of R and statsmodels
- **Result Altering Change:** hc1 is the new default robust estimator for :code:`Lm` models, changed from hc0
- **API change:** all model residuals are now saved in the :code:`model.residuals` attribute and were formerly saved in the :code:`model.resid` attribute. This is to maintain consistency with :code:`model.data` column names. 
- **New feature:** addition of :code:`pymer4.stats` module for various parametric and non-parametric statistics functions (e.g. permutation testing and bootstrapping)
- **New feature:** addition of :code:`pymer4.io` module for saving and loading models to disk
- **New feature:** addition of :code:`Lm2` models that can perform multi-level modeling by first estimating a separate regression for each group and then performing inference on those estimates. Can perform inference on first-level semi-partial and partial correlation coefficients instead of betas too.
- **New feature:** All model classes now have the ability to rank transform data prior to estimation, see the rank argument of their respective :code:`.fit()` methods.
- **New features for Lm models:** 
    - :code:`Lm` models can transform coefficients to partial or semi-partial correlation coefficients
    - :code:`Lm` models can also perform weight-least-squares (WLS) regression given the weights argument to :code:`.fit()`, with optional dof correction via Satterthwaite approximation. This is useful for categorical (e.g. group) comparison where one does not want to assume equal variance between groups (e.g. Welch's t-test). This remains an experimental feature
    - :code:`Lm` models can compute hc1 and hc2 robust standard errors
- **New documentation look:** the look and feel of the docs site has been completely changed which should make getting information much more accessible. Additionally, overview pages have now been turned into downloadable tutorial jupyter notebooks
- All methods/functions capable of parallelization now have their default :code:`n_jobs` set to 1 (i.e. no default parallelization)
- Various bug fixes to all models 
- Automated testing on travis now pins specific r and r-package versions
- Switched from lsmeans to emmeans for post-hoc tests because lsmeans is deprecated
- Updated interactions with rpy2 api for compatibility with version 3 and higher
- Refactored package layout for easier maintainability 

0.6.0
-----
- **Dropped support for Python 2** 
- upgraded :code:`rpy2` dependency version
- Added conda installation instructions
- Accepted `JOSS <https://joss.theoj.org/>`_ version

0.5.0
-----
- :code:`Lmer` models now support all generalized linear model family types supported by lme4 (e.g. poisson, gamma, etc)
- :code:`Lmer` models now support ANOVA tables with support for auto-orthogonalizing factors using the :code:`.anova()` method
- Test statistic inference for :code:`Lmer` models can now be performed via non-parametric permutation tests that shuffle observations within clusters
- :code:`Lmer.fit(factors={})` arguments now support custom arbitrary contrasts
- New forest plots for visualizing model estimates and confidence intervals via the :code:`Lmer.plot_summary()` method
- More comprehensive documentation with examples of new features
- Submission to `JOSS <https://joss.theoj.org/>`_ 

0.4.0
-----
- Added :code:`.post_hoc()` method to :code:`Lmer` models
- Added :code:`.simulate()` method to :code:`Lmer` models
- Several bug fixes for Python 3 compatibility

0.3.2
-----
- addition of :code:`simulate` module

0.2.2
-----
- Official pyipi **release**

0.2.1
-----
- Support for standard linear regression models
- Models include support for robust standard errors, boot-strapped CIs, and permuted inference

0.2.0
-----
- Support for categorical predictors, model predictions, and model plots

0.1.0
-----
- Linear and Logit multi-level models
