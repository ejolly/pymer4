Lme4 Random Effects Cheat Sheet
===============================

Because python is calling :code:`lmer` from the :code:`lme4` package in R behind the scenes, some familiarity with :code:`lmer` model `formulae <https://stats.stackexchange.com/questions/18428/formula-symbols-for-mixed-model-using-lme4>`_. Here is a quick reference for random effects specifications:

.. code-block:: python

    #Random intercepts only
    (1 | Group)

    #Random slopes only
    (0 + Variable | Group)

    #Random intercepts and slopes (and their correlation)
    (Variable | Group)

    #Random intercepts and slopes (without their correlation)
    (1 | Group) + (0 + Variable | Group)

    #Same as above only if Variable is *continuous* and not a factor
    (Variable || Group)

    #Random intercept and slope for more than one variable (and their correlations)
    (Variable_1 + Variable_2 | Group)