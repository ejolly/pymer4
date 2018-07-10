ANOVA tables and post-hoc comparisons
=====================================

ANOVA tables and orthogonal contrasts
-------------------------------------
Because ANOVA is just regression, :code:`pymer4` makes it easy to estimate ANOVA tables with F-results using the :code:`.anova()` method on a fitted model. By default this will compute a Type-III SS table given the coding scheme provided when the model was initially fit. Based on the distribution of data across factor levels and the specific factor-coding used, this may produce invalid Type-III SS computations. For this reason the :code:`.anova()` method has a :code:`force-orthogonal=True` option that will reparameterize and refit the model using orthogonal polynomial contrasts prior to computing an ANOVA table.

Marginal estimates and post-hoc comparisons
-------------------------------------------
:code:`pymer4` leverages the :code:`lsmeans` package in order to compute marginal estimates and pair-wise comparisons of models that contain categorical terms and/or interactions. This can be performed by using the :code:`.post_hoc()` method on fitted models with various examples in the method help. Currently post-hoc comparisons are not possible from :code:`Lm()` models, only from :code:`Lmer()` models.
