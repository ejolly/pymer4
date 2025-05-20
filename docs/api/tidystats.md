# `pymer4.tidystats`

The `tidystats` module contains all the functions that support the features of `pymer4`'s models. Moreover, `tidystats` can be used as a functional alternative to the object-oriented-approach of `pymer4.models`

```python
import pymer4.tidystats as ts
from pymer4 import load_dataset

df = load_dataset('sleep')
model = ts.lm('Reaction ~ Days', data=df)

# Like calling coef() in R
ts.coef(model)

# Like calling tidy() in R
ts.tidy(model)
```

## `base`

Wraps functionality from the [`base`](https://www.rdocumentation.org/packages/base/versions/3.6.2) R library

```{eval-rst}
.. automodule:: pymer4.tidystats.base
  :members:
  :member-order: alphabetical
```

## `broom`

Wraps functionality from the [`broom`](https://broom.tidymodels.org/) and [`broom.mixed`](https://cran.r-project.org/web/packages/broom.mixed/index.html) libraries

```{eval-rst}
.. automodule:: pymer4.tidystats.broom
  :members:
  :member-order: alphabetical
```

## `easystats`

Wraps functionality from various sub-libraries in the [`easystats`](https://easystats.github.io/easystats/) ecoysystem

```{eval-rst}
.. automodule:: pymer4.tidystats.easystats
  :members:
  :member-order: alphabetical
```

## `emmeans_lib`

Wraps functionality from the [`emmeans`](https://rvlenth.github.io/emmeans/index.html) library

```{eval-rst}
.. automodule:: pymer4.tidystats.emmeans_lib
  :members:
  :member-order: alphabetical
```

## `lmerTest`

Wraps functionality from the [`lme4`](https://cran.r-project.org/web/packages/lme4/index.html) and [`lmerTest`](https://rdrr.io/cran/lmerTest/man/lmerTest-package.html) libraries

```{eval-rst}
.. automodule:: pymer4.tidystats.lmerTest
  :members:
  :member-order: alphabetical
```

## `multimodel`

Functions that intelligently switch their functionality based on whether they received and `lm`, `glm`, `lmer` or `glmer` model as input, mimicking "function overloading" in R

```{eval-rst}
.. automodule:: pymer4.tidystats.multimodel
  :members:
  :member-order: alphabetical
```

## `stats`

Wraps functionality from the [`stats`](https://www.rdocumentation.org/packages/stats/versions/3.6.2) library

```{eval-rst}
.. automodule:: pymer4.tidystats.stats
  :members:
  :member-order: alphabetical
```

## `tibble`

Wraps functionality from the [`tibble`](https://tibble.tidyverse.org/) library

```{eval-rst}
.. automodule:: pymer4.tidystats.tibble
  :members:
  :member-order: alphabetical
```

## `bridge`

This is a special module that helps with converting between R and Python datatypes. It's particularly useful if you want to try to add [additional features](../contributing/extending.ipynb) to `pymer4`

```{eval-rst}
.. automodule:: pymer4.tidystats.bridge
  :members:
  :member-order: alphabetical
```

## `plutils`

Utility functions for working with `polars` dataframes

```{eval-rst}
.. automodule:: pymer4.tidystats.plutils
  :members:
  :member-order: alphabetical
```

```{eval-rst}
.. autoclass:: pymer4.tidystats.plutils.RandomExpr
  :members:

```

## `tables`

Functions to generate [`great tables`](https://posit-dev.github.io/great-tables/articles/intro.html) formatted summary tables for models

```{eval-rst}
.. automodule:: pymer4.tidystats.tables
  :members:
  :member-order: alphabetical
```