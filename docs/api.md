# API Reference

## `pymer4.models.Lmer`

Model class for estimating linear mixed-effects models using `lme4` behind-the-scenes. Primary methods include:

- `.fit()`: [fits a linear mixed-effects model](#pymer4.models.Lmer.fit)
- `.summary()`: [summary of fitted model](#pymer4.models.Lmer.summary)
- `.anova()`: [type-3 omnibus F-tests with optional reorthogonalization of contrasts](#pymer4.models.Lmer.anova)
- `.post_hoc()`: [get marginal estimates and post-hoc comparisons from models with categorical predictors](#pymer4.models.Lmer.post_hoc)
- `.confint()`: [profile, Wald or bootstrapped confidence intervals](#pymer4.models.Lmer.confint)
- `.predict()`: [predicts new data using a fitted model](#pymer4.models.Lmer.predict)
- `.simulate()`: [simulates new data using a fitted model](#pymer4.models.Lmer.simulate)
<!-- - `.plot()`: plot estimates from fitted model -->
<!-- - `.plot_summary()`: forestplot of estimates and confidence intervals -->

```{eval-rst}
.. autoclass:: pymer4.models.Lmer
    :members:
    :member-order: alphabetical
```

## `pymer4.models.Lm`

Model class for estimating standard regression models similar to `lm()` in R, but evaluated entirely in Python. Primary methods include:

- `.fit()`: [fits a linear model](#pymer4.models.Lm.fit)
- `.summary()`: [summary of fitted model](#pymer4.models.Lm.summary)
- `.predict()`: [predicts new data using a fitted model](#pymer4.models.Lm.predict)
- `.to_corrs()`: [transform coefficients to partial or semi-partial correlation scaled to aid interpretability](#pymer4.models.Lm.to_corrs)
<!-- - `.plot_summary()`: forestplot of estimates and confidence intervals -->

```{eval-rst}
.. autoclass:: pymer4.models.Lm
    :members:
    :member-order: alphabetical
```

## `pymer4.models.Lm2`

Model class for estimating multi-level models in Python using the summary-statistics approach. Primary methods include:

- `.fit()`: [fits a multi-level model](#pymer4.models.Lm2.fit)
- `.summary()`: [summary of fitted model](#pymer4.models.Lm.summary)
<!-- - `.plot_summary()`: [predicts new data using a fitted model](#pymer4.models.Lm.predict) -->

```{eval-rst}
.. autoclass:: pymer4.models.Lm2
    :members:
    :member-order: alphabetical
```

## `pymer4.simulate`

Functions for generating data for use with various model types

```{eval-rst}
.. automodule:: pymer4.simulate
    :members:
    :undoc-members:
    :show-inheritance:
```

## `pymer4.stats`

General purpose functions for various parametric and non-parametric statistical routines

```{eval-rst}
.. automodule:: pymer4.stats
    :members:
    :undoc-members:
    :show-inheritance:
```

## `pymer4.utils`

Miscellaneous helper functions

```{eval-rst}
.. automodule:: pymer4.utils
    :members:
    :undoc-members:
    :show-inheritance:
```

## `pymer4.io`

Functions for persisting models to disk

```{eval-rst}
.. automodule:: pymer4.io
    :members:
    :undoc-members:
    :show-inheritance:
```
