# Migrating from Previous Versions

If you were using a version of `pymer4` older than `v0.9.X` you can refer to the notes and table below for how to migrate your workflow. These changes were motivated by two goals: (a) making a more consistent and intuitive API; (b) simplifying library maintenance and extensibility. In particular, we now offload *all computations* to R including multi-core parallelization (e.g. for bootstrapping) by leveraging various popular R libraries (e.g. `parameters::model_parameters`, `lme4::bootMer`). Python code is almost entirely a means to store and convert the inputs and outputs of a variety of R functions in the [`tidystats`](../api/tidystats.md) module using [`polars` dataframes](https://docs.pola.rs/api/python/stable/reference/index.html).

## API Changes

| New models classes | Old model classes |
|---------|---------|
| `lm()`  | `Lm()` |
| `glm()`  | `Lm()`; only `family ='binomial'`|
| `lmer()`  | `Lmer()` |
| `glmer()`  | `Lmer()`; with `family` kwarg |

| Function | New models API | Old model API |
|-------|---------|---------|
| Create a model | `model('y ~ x', data=polars_df)` | `Model('y ~ x', data=pandas_df)` |
| Fit a model | `.fit()` | same | 
| ANOVA table | `.anova()` | same | 
| Marginal estimates & comparisons | `.emmeans()` / `.empredict()` | `.post_hoc()` | 
| Parameter estimates | `.params`, `.result_fit` | `.coef` | 
| Fit statistics | `.result_fit_stats` | `.AIC`, `.BIC`, etc | 
| Fixed-effects LMMs/GLMMs | `.fixef` | `.fixef` | 
| Random-effects LMMs/GLMMs | `.ranef` | `.ranef` | 
| Random-effects-variances LMMs/GLMMs | `.ranef_var` | `.ranef_var` | 
| Use various statistical functions | `pymer4.tidystats` (R-based) | `pymer4.stats` (python-based) |

## Deprecated Features

- Removed support for `Lm2()` models
- Removed plotting methods (please use `seaborn` or `matplotlib`)
- Removed permutation-based inference for, e.g. `model.fit(permute=1000)`

## Planned Improvements

- integration with `scikit-learn` Estimators and cross-validation tools
- additional models (e.g. ordinal regression via `ordinal::clm`)