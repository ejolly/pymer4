# `pymer4.models.glmer`

:::{admonition} Tutorial
:class: tip
Check out the [LMMs ](../../tutorials/03_lmms.ipynb) and [GLMMs tutorial](../../tutorials/04_glmms.ipynb) for usage examples
:::

## GLMMs

**G**eneralized **L**inear **M**odels **M**odels fit using Restricted-Maximum-Likelihood-Estimation (REML) or Maximum-Likelihood-Estimation (MLE)
 
GLMMs generalize LMMs like GLMs generalize LMs. They are well suited for non-independent data and non-gaussian outcome variables such as binary outcomes or counts. This includes models like mixed-effects logistic-regression and multi-level poisson models. 

Like LMMs they are particularly useful in situations when observations are non-independent (e.g. repeated-measures designs, hierarchical data, panel-data, time-series, clustered data). To account for this GLMMs estimate additional *random-effects* estimates that reflect how a cluster of observations deviates from fixed effects estimates (e.g. random-intercepts and/or random-slopes)

For some models like mixed logistic-regression, it can be helpful to use `.fit(exponentiate=True)` to transform estimates to the odds scale to aid interpretability. By default the `'fitted'` column in `model.data` and the output of `model.predict()` uses `type_predict = 'response'` so that model predictions are on the *response* scale, i.e. probabilities for mixed logistic-regression.

```python
from pymer4 import load_dataset('titanic')
from pymer4.models import glm

titanic = load_dataset('titanic')

# Logistic regression accounting repeated observations within pclass
# by estimating a random intercept per level of pclass
log_reg = glm('survived ~ fare + (1|plass)', family='binomial', data=titanic)
log_reg.set_factors('pclass')

# See parameter estimates on odds scale
log_reg.fit(exponentiate=True)
```

---

```{eval-rst}
.. autoclass:: pymer4.models.glmer.glmer
  :exclude-members: fit, summary

```

## Estimation Methods 

Estimation methods comprise the most common method you will work with on a routine basis for estimating model parameters, omnibus-tests, marginal estimations & comparisons, predictions, and simulations. 

```{eval-rst}
.. autofunction:: pymer4.models.glmer.glmer.fit
.. autofunction:: pymer4.models.lmer.lmer.anova
.. autofunction:: pymer4.models.lmer.lmer.emmeans
.. autofunction:: pymer4.models.base.model.empredict
.. autofunction:: pymer4.models.glmer.glmer.predict
.. autofunction:: pymer4.models.lmer.lmer.simulate
.. autofunction:: pymer4.models.base.model.vif

```

## Summary Methods

Summary methods return nicely formatted outputs of the `.result_*` attributes of a fitted model

```{eval-rst}
.. autofunction:: pymer4.models.glmer.glmer.summary
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
