# `pymer4.models.glm`

:::{admonition} Tutorial
:class: tip
Check out the [linear regression](../../tutorials/01_lm.ipynb) and [GLMs tutorial](../../tutorials/04_glmms.ipynb) for usage examples
:::

## GLM

**G**eneralized **L**inear **M**odels fit using Maximum-Likelihood-Estimation (MLE)

GLMs are useful for estimating models with non-gaussian outcome variables. These include models like logistic regression for binary data and poisson regression for count data.

GLMs work just like LMs but additionally take a `family` and `link` argument when being initialized. For convenience, when estimating a logistic regression specifically (`family ='binomial'`), parameter estimates and uncertainty can be auto-converted to log-odd and probability scales.

```python
from pymer4 import load_dataset('titanic')
from pymer4.models import glm

titanic = load_dataset('titanic')

# Logistic regression with logit link
log_reg = glm('survived ~ fare', family='binomial', data=titanic)
log_reg.fit()

# See parameter estimates on probability scale
log_reg.summary(to_probs=True)

# Logistic regression with probit link
probit_reg = glm('survived ~ fare', family='binomial', link='probit', data=titanic)
probit_reg.fit()

```

---

```{eval-rst}
.. autoclass:: pymer4.models.glm.glm
  :exclude-members: fit

```

## Estimation Methods

Estimation methods comprise the most common method you will work with on a routine basis for estimating model parameters, omnibus-tests, marginal estimations & comparisons, predictions, and simulations. 

```{eval-rst}
.. autofunction:: pymer4.models.glm.glm.fit
.. autofunction:: pymer4.models.base.model.anova
.. autofunction:: pymer4.models.base.model.emmeans
.. autofunction:: pymer4.models.base.model.empredict
.. autofunction:: pymer4.models.glm.glm.predict
.. autofunction:: pymer4.models.base.model.simulate
.. autofunction:: pymer4.models.base.model.vif

```

## Summary Methods

Summary methods return nicely formatted outputs of the `.result_*` attributes of a fitted model

```{eval-rst}
.. autofunction:: pymer4.models.glm.glm.summary
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
