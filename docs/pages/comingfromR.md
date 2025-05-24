# Coming from R

If you're familiar with R's statistical modeling ecosystem this guide will help you understand the similarities and differences between R's libraries and `pymer4`. 

```{admonition} Note
:class: note
Pymer4 is *not* a reimplementation of R's models — it's a Python interface to the same R functions you know and love. When you fit a model in `pymer4`, it calls R code behind-the-scenes from a variety of different libraries and integrates the output into a user-friendly Pythonic interface.
```

## Quick Reference

| Functionality | R `library::function` used | Python `pymer4` code |
|------|------------|--------|
| Linear models | `stats::lm()` | `pymer4.models.lm` |
| Generalized linear models | `stats::glm()` | `pymer4.models.glm` |
| Mixed models | `lme4::lmer()`, `lmerTest` | `pymer4.models.lmer` |
| Generalized mixed models | `lme4::glmer()` | `pymer4.models.glmer` |
| Model summaries | `broom/broom.mixed::tidy()` | `.summary()` |
| Model fit statistics | `broom/broom.mixed::glance()`, `performance::model_performance()` | `.result_fit_stats` |
| Model predictions, residuals, etc | `broom/broom.mixed::augment()` | `.data` gains extra cols after using `.fit()` |
| ANOVA (Type-III, Type-I) | `emmeans::joint_tests()`, `stats::anova()` | `.anova()`, `.anova(auto_ss_3=False)` |
| Marginal effects estimation and comparison | `emmeans::emmeans()`, `emtrends()`, `contrasts()`, `refgrid()` | `.emmeans()`, `.empredict()` |

## Side-by-Side Examples

### Factors, Contrasts, Transforms

In R you convert variables to factors, adjust contrasts, or transform variables on the *dataframe.* In `pymer4` you use the `.set_*` methods on a *model*.

::::{tab-set}
:::{tab-item} R
```r
# Load data
data(mtcars)

# Convert variables to factors
mtcars$cyl <- as.factor(mtcars$cyl)

# Center a predictor
mtcars$wt_c <- scale(mtcars$wt, center = TRUE, scale = FALSE)

# Fit model
model <- lm(mpg ~ wt_c * cyl, data = mtcars)

# Change the default contrasts and refit
contrasts(mtcars$cyl) <- contr.sum(3)
model <- lm(mpg ~ wt_c * cyl, data = mtcars)
```
:::
:::{tab-item} Python
```python
# Load data
from pymer4 import load_dataset
from pymer4.models import lm
mtcars = load_dataset('mtcars')

# Create model
model = lm("mpg ~ wt * cyl", data=data)

# Conver variable to factor
model.set_factors("cyl") 

# Center a predictor
model.set_transforms({"wt": "center"})

# Fit model
model.fit()

# Change the default contrasts and refit
model.set_contrasts({'wt': 'contr.sum'})
model.fit()
```
:::
::::

### Summarizing model fit

In R you use *functions* from different packages (e.g. `broom::tidy()`) to summarize or extract information from a model. In `pymer4` these are *automatically calculated* and stored as model *attributes* accessible with `.attribute_name`.  

::::{tab-set}
:::{tab-item} R
```r
# Load packages
library(broom)
library(performance)
library(emmeans)

# Fit model
model <- lm(mpg ~ wt * cyl, data = mtcars)

# Standard R summary
summary(model)

# Enhanced tidy summary
tidy(model, conf.int = TRUE)

# Predictions, residuals, etc
augment(model)

# Type-III ANOVA
joint.tests(model)

# Model fit metrics
model_performance(model)
```
:::

:::{tab-item} Python
```python
# Fit model
model = lm("mpg ~ wt * cyl", data=data)
model.fit()

# Standard R summary
model.summary(pretty=False)

# Enhanced tidy summary
model.summary()

# Predictions, residuals, etc
model.data

# Type-III ANOVA
model.anova()

# Model fit metrics
model.result_fit_stats
```
:::
::::

### Mixed-Effects Models

For mixed-models in R you need `lmerTest` to get p-values, use functions to extract different model components and estimates (e.g. `fixef()`, `broom.mixed::tidy()`), and navigate a few different APIs to get bootstrapped parameter estimates (e.g. `bootMer()`, `performance::bootstrap_model()`). In `pymer4` this is all handled for you using methods and attributes of the *model*.

::::{tab-set}
:::{tab-item} R
```r
# Load libraries
library(lme4)
library(lmerTest)  # for p-values
library(broom.mixed)
library(performance)

# Load data
data(sleepstudy)

# Fit model
model <- lmer(Reaction ~ Days + (Days | Subject), data = sleepstudy)

# Get results
summary(model)

# Get nicer summary
tidy(model, conf.int=TRUE)

# Extract params
coef(model)
fixef(model)
ranef(model)

# Or nicer
tidy(model, effects = "ran_pars")

# Model diagnostics and variance partitioning
icc(model)
model_performance(model)

# Bootstrapping
bootMer(model, ...)

# Without saving bootstraps
bootstrap_model(model)
```
:::

:::{tab-item} Python
```python
# Load data
sleepstudy = load_dataset("sleep")

# Fit model
model =.lmer("Reaction ~ Days + (Days | Subject)", data=sleepstudy)
# Print all-inclusive summary after fit
model.fit(summary=True)

# Params saved for you
model.params

# BLUPs
model.fixef

# RFX
model.ranef

# Variances
model.ranef_var

# As well as ICC, variance partitioning, etc
model.result_fit_stats

# Bootstrapping
model.fit(conf_method='boot')

# .result_fit auto-matically updated with bootstrapped CIs

# To inspect individual bootstraps:
model.result_boots
```
:::
::::

### Marginal Effects Estimation

In R `emmeans` is the an incredibly popular package that provides a highly-flexible set of functions for a variety of estimates and comparisons. In `pymer4` most of these features are available in a simplified API using `.emmeans()` for aggregated estimates and contrasts or `.empredict()` for arbitrary observation-level predictions, contrasts, counter-factual estimates etc.

::::{tab-set}
:::{tab-item} R

```r
library(emmeans)

# Fit model
mtcars$cyl <- as.factor(mtcars$cyl)
model <- lm(mpg ~ wt * cyl, data = mtcars)

# Get marginal means
emm <- emmeans(model, ~ cyl)
summary(emm)

# Pairwise comparisons
pairs(emm)

# Interaction contrasts
emmeans(model, pairwise ~ cyl | wt)
```
:::
:::{tab-item} Python
```python
# Fit model
model = lm("mpg ~ wt * cyl", data=data)
model.set_factors("cyl")
model.fit()

# Get marginal means
model.emmeans("cyl")

# Pairwise comparisons
model.emmeans("cyl", contrasts="pairwise")

# Interaction contrasts
model.emmeans(["cyl", "wt"], contrasts="pairwise")
```
:::
::::
