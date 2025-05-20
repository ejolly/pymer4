# `pymer4.models.lm`

:::{admonition} Tutorial
:class: tip
Check out the [linear regression](../../tutorials/01_lm.ipynb) and [ANOVA](../../tutorials/02_categorical.ipynb) for usage examples
:::

## LM

**L**inear **M**odels fit using Ordinary-Least-Squares (OLS)

Linear models estimate a response $y$ as a linear combination of predictors $X$

---

```{eval-rst}
.. autoclass:: pymer4.models.lm.lm
  :exclude-members: fit

```

## Estimation Methods 

Estimation methods comprise the most common method you will work with on a routine basis for estimating model parameters, omnibus-tests, marginal estimations & comparisons, predictions, and simulations. 

```{eval-rst}
.. autofunction:: pymer4.models.lm.lm.fit
.. autofunction:: pymer4.models.base.model.anova
.. autofunction:: pymer4.models.base.model.emmeans
.. autofunction:: pymer4.models.base.model.empredict
.. autofunction:: pymer4.models.base.model.predict
.. autofunction:: pymer4.models.base.model.simulate
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
