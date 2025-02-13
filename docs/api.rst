API Reference
=============

:class:`pymer4.models.Lmer`
---------------------------
Model class for estimating linear mixed-effects models using :code:`lme4`. Primary methods include:

- :code:`.anova()` `type-3 omnibus F-tests with optional reorthogonalization of contrasts <#pymer4.models.Lmer.anova>`_
- :code:`.confint()`: `profile, Wald or bootstrapped confidence intervals <#pymer4.models.Lmer.confint>`_ (wrapper around :code:`confint.merMod` in R)
- :code:`.fit()`: fits a linear mixed-effects model
- :code:`.plot()`: plot estimates from fitted model
- :code:`.plot_summary()`: forestplot of estimates and confidence intervals
- :code:`.post_hoc()`: performs post-hoc pairwise comparisons
- :code:`.predict()`: predicts new data using fitted model
- :code:`.simulate()`: simulates new data using fitted model
- :code:`.summary()`: summary of fitted model

.. autoclass:: pymer4.models.Lmer
    :members: 
    :member-order: alphabetical

:class:`pymer4.models.Lm`: Lm
-----------------------------
Model class for estimating standard regression models

.. autoclass:: pymer4.models.Lm
    :members:

:class:`pymer4.models.Lm2`: Lm2
-------------------------------
Model class for estimating multi-level models in python using the summary-statistics approach

.. autoclass:: pymer4.models.Lm2
    :members:

:mod:`pymer4.simulate`: Simulation Functions
--------------------------------------------
Functions for generating data for use with various model types

.. automodule:: pymer4.simulate
    :members:
    :undoc-members:
    :show-inheritance:

:mod:`pymer4.stats`: Statistics Functions
-----------------------------------------
General purpose functions for various parametric and non-parametric statistical routines

.. automodule:: pymer4.stats
    :members:
    :undoc-members:
    :show-inheritance:

:mod:`pymer4.utils`: Utility Functions
--------------------------------------
Miscellaneous helper functions

.. automodule:: pymer4.utils
    :members:
    :undoc-members:
    :show-inheritance:

:mod:`pymer4.io`: Save/Load Functions
-------------------------------------
Functions for persisting models to disk

.. automodule:: pymer4.io
    :members:
    :undoc-members:
    :show-inheritance:
