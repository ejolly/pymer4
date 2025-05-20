# `pymer4.models`

## Overview

`pymer4` includes 4 types of models that share a consistent API and can be imported like this:

```python
from pymer4.models import lm, glm, lmer, glmer
```

Clicking the links below will take you to their respective API documentation pages.

| Model | Description |
|--------|-------------|
| [`lm()`](./lm.md) | linear regression fit via ordinary-least-squares (OLS) |
| [`glm()`](./glm.md) | generalized linear models (e.g logistic regression) fit via maximum-likelihood-estimate (MLE) |
| [`lmer()`](./lmer.md) | linear-mixed / multi-level models |
| [`glmer()`](./glmer.md) | geneneralized linear-mixed / multi-level models |


## Comparing Models

Nested model comparison is available across all model types using the `compare()` function

```{eval-rst}
.. autofunction:: pymer4.models.compare
```

<!-- #TODO -->
<!-- Additional model comparison is possible via `scikit-learn` using `.to_sklearn()` -->