What's New
==========
Historically :code:`pymer4` versioning was a bit all over the place but has settled down since 0.5.0. This page includes the most notable updates between versions but github is the best place to checkout more details and `releases <https://github.com/ejolly/pymer4/releases/>`_.

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
- **New features for :code:`Lm` models:** 
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
