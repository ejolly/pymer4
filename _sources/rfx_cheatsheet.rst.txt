Lme4 Random Effects Cheat Sheet
===============================

Because :code:`Lmer` models just call the :code:`lme4` package in R behind the scenes, some familiarity with :code:`lmer` model `formulae <https://stats.stackexchange.com/questions/18428/formula-symbols-for-mixed-model-using-lme4>`_ is required. Here is a quick reference for common random effects specifications:

.. code-block:: python

    #Random intercepts only
    (1 | Group)

    #Random slopes only
    (0 + Variable | Group)

    #Random intercepts and slopes (and their correlation)
    (Variable | Group)

    #Random intercepts and slopes (without their correlation)
    (1 | Group) + (0 + Variable | Group)

    #Same as above but will not separate factors (see: https://rdrr.io/cran/lme4/man/expandDoubleVerts.html)
    (Variable || Group)

    #Random intercept and slope for more than one variable (and their correlations)
    (Variable_1 + Variable_2 | Group)
