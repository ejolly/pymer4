# R libraries

Starting in version `0.9.0` `pymer4` introduced a new module called `tidystats` which provides a functional interface to several different R libraries. These functions are used behind-the-scenes for `lm`, `glm`, `lmer` and `glmer` models and automatically handle most Python-to-R-and-back conversion. To help encourage future contributions we've included a [guide for adding additional R libraries](./extending.ipynb) which outlines the (relatively straightforward) process. That means if there's package in R you really enjoy working with and it's available on `conda-forge` it can probably be added to `pymer4` in the future!

## Current libraries

### `base`
```{eval-rst}
.. currentmodule:: pymer4.tidystats.base

.. autosummary:: 
  :nosignatures:

  names
  summary
  row_names
```

### `broom/broom.mixed`
```{eval-rst}
.. currentmodule:: pymer4.tidystats.broom

.. autosummary:: 
  :nosignatures:

  augment
  glance
  tidy
```

### `emmeans`
```{eval-rst}
.. currentmodule:: pymer4.tidystats.emmeans_lib

.. autosummary:: 
  :nosignatures:

  emmeans
  emtrends
  joint_tests
  ref_grid
```


### `lme4/lmerTest`
```{eval-rst}
.. currentmodule:: pymer4.tidystats.lmerTest

.. autosummary:: 
  :nosignatures:

  fixef
  glmer
  lmer
  ranef
```

### `stats`
```{eval-rst}
.. currentmodule:: pymer4.tidystats.stats

.. autosummary:: 
  :nosignatures:

  anova
  lm
  glm
  resid
  model_matrix
```

### `tibble`
```{eval-rst}
.. currentmodule:: pymer4.tidystats.tibble

.. autosummary:: 
  :nosignatures:

  as_tibble
```

### `multimodel`

```{note}
Functions in this module consider the *type* of model passed as input, and intelligently switch between the appropriate method (e.g. `lm` vs `lmer` models)
```

```{eval-rst}
.. currentmodule:: pymer4.tidystats.multimodel

.. autosummary:: 
  :nosignatures:

  boot
  coef
  confint
  predict
  simulate
```

