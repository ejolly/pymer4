# `pymer4.models.lmer`

:::{admonition} Tutorial
:class: tip
Check out the [LMMs ](../../tutorials/03_lmms.ipynb) and [GLMMs tutorial](../../tutorials/04_glmms.ipynb) for usage examples
:::

## LMMs

**L**inear **M**odels **M**odels fit using Restricted-Maximum-Likelihood-Estimation (REML) or Maximum-Likelihood-Estimation (MLE)
 
LMMs are also commonly known as linear-mixed-effects (LMEs), multi-level-models (MLMs), hierarchical-linear-models (HLMs), and are particularly useful in situations when observations are non-independent (e.g. repeated-measures designs, hierarchical data, panel-data, time-series, clustered data). To account for this LMMs include *random-effects* parameter estimates that capture cluster-level deviations around fixed effects parameter estimates (e.g. random-intercepts and/or slopes)

```python
from pymer4 import load_dataset('sleep')
from pymer4.models import lmer, compare

sleep = load_dataset('sleep')

# Random intercept for each Subject
lmm_i = lmer('Reaction ~ Days + (1 | Subject)', data=sleep)

# Random intercept and slope for each Subject
lmm_s = lmer('Reaction ~ Days + (Days | Subject)', data=sleep)

# Compare models with different rfx
compare(lmm_s, lmm_i)

```

---

```{eval-rst}
.. autoclass:: pymer4.models.lmer.lmer
  :exclude-members: fit

```

## Estimation Methods 

Estimation methods comprise the most common method you will work with on a routine basis for estimating model parameters, omnibus-tests, marginal estimations & comparisons, predictions, and simulations. 

```{eval-rst}
.. autofunction:: pymer4.models.lmer.lmer.fit
.. autofunction:: pymer4.models.lmer.lmer.anova
.. autofunction:: pymer4.models.lmer.lmer.emmeans
.. autofunction:: pymer4.models.base.model.empredict
.. autofunction:: pymer4.models.lmer.lmer.predict
.. autofunction:: pymer4.models.lmer.lmer.simulate
.. autofunction:: pymer4.models.base.model.vif

```

## Summary Methods

Summary methods return nicely formatted outputs of the `.result_*` attributes of a fitted model

```{eval-rst}
.. autofunction:: pymer4.models.base.model.summary
.. autofunction:: pymer4.models.base.model.summary_anova

```

## Transformation & Factor Methods

These methods are essential for working *categorical predictors* (factors), customizing specific linear hypotheses, and transforming continous predictors (e.g. mean-centering).

```{eval-rst}

.. autofunction:: pymer4.models.base.model.set_factors
.. autofunction:: pymer4.models.base.model.unset_factors
.. autofunction:: pymer4.models.base.model.show_factors
.. autofunction:: pymer4.models.base.model.set_contrasts
.. autofunction:: pymer4.models.base.model.show_contrasts
.. autofunction:: pymer4.models.base.model.set_transforms
.. autofunction:: pymer4.models.base.model.unset_transforms
.. autofunction:: pymer4.models.base.model.show_transforms

```

## Auxillary Methods

Helper methods for more advanced functionality and debugging

```{eval-rst}

.. autofunction:: pymer4.models.base.model.to_sklearn
.. autofunction:: pymer4.models.base.model.report
.. autofunction:: pymer4.models.base.model.show_logs
.. autofunction:: pymer4.models.base.model.clear_logs

```
