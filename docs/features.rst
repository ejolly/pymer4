Features
========

Overview
--------
All :code:`pymer4` models operate on long-format `pandas dataframes <https://pandas.pydata.org/pandas-docs/version/0.25/index.html/>`_. These dataframes should contain columns for a dependent variable, independent variable(s), and optionally a column for a group/cluster identifiers. 

Currently, :code:`pymer4` contains 3 different model classes:

- :code:`Lm` for ordinary-least-squares and weighted-least-squares regression optionally with robust standard errors
- :code:`Lmer` for multi-level models estimated using :code:`glmer()` in R. 
- :code:`Lm2` for two-stage ordinary-least-squares in which a separate :code:`Lm` model is fit to every group/cluster and inference is performed on the coefficients across all groups/clusters. This is also known as the "summary statistics approach" and is an alternative to multi-level models estimated using :code:`Lmer`, which implicitly allow for both random-intercepts and random-slopes but shares no information across each groups/clusters to help during estimation.

Standard regression models
--------------------------
:code:`Lm` models which are equivalent to :code:`lm()` in R with the following additional features:

- Automatic inclusion of confidence intervals in model output
- Optional empirically bootstrapped 95% confidence intervals
- Cluster-robust, heteroscedasticity-robust or auto-correlation-robust, 'sandwich estimators' for standard errors (*note: these are not the same as auto-regressive models*)
- Weighted-least-squares models (experimental)
- Permutation tests on model parameters

Multi-level models
------------------
:code:`Lmer` models which are equivalent to :code:`glmer()` in R with the following additional features:

- Automatic inclusion of p-values in model output using `lmerTest <https://cran.r-project.org/web/packages/lmerTest/index.html>`_
- Automatic inclusion of confidence intervals in model output
- Automatic conversion and calculation of odds-ratios and probabilities for logit models
- Easy access to group/cluster fixed and random effects as pandas dataframes
- Random effects plotting using seaborn
- Easy post-hoc tests with multiple-comparisons correction via `emmeans <https://cran.r-project.org/web/packages/emmeans/index.html>`_
- Easy model predictions on new data
- Easy generation of new data from a fitted model
- Optional permuted p-value computation via within cluster permutation testing (experimental)

Other Features
--------------
- Highly customizable functions for simulating data useful for standard regression models and multi-level models
- Convenience methods for plotting model estimates, including random-effects terms in multi-level models
- Statistics functions for effect-size computation, permutations of various 1 and 2 sample tests, bootstrapping of various 1 and 2 sample tests, and two-one-sided equivalence tests
