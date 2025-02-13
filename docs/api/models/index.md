# `pymer4.models`

`pymer4` includes 4 types of models that share a consistent API:

| Model | Description |
|--------|-------------|
| [`lm()`](./lm.md) | linear regression fit via ordinary-least-squares (OLS) |
| [`glm()`](./glm.md) | generalized linear models (e.g logistic regression) fit via maximum-likelihood-estimate (MLE) |
| [`lmer()`](./lmer.md) | linear-mixed / multi-level models |
| [`glmer()`](./glmer.md) | geneneralized linear-mixed / multi-level models |


Nested model comparison is available across all model types using the `compare()` function

```{eval-rst}
.. autofunction:: pymer4.models.compare
```