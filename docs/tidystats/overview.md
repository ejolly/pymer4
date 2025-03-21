# Introducing the `pymer4.tidystats` interface

Starting in version `0.9.0` `pymer4` introduced a new module called `tidystats` as a completely new interface for estimating and working with *any* general-linear-models or general-linear-mixed-models estimable in R with `lm`, `glm`, `lmer` and `glmer`. It offers a more consistent, composable, and maintainable API by integration several functions from various R packages that are design to work well together (e.g. `tidyverse`). To help encourage future contributions we've included a [guide for adding additional R libraries](./extending.ipynb) which outlines the (relatively straightforward) process. That means if there's package in R you really enjoy working with and it's available on `conda-forge` it can be added to `tidystats`!

The heart of `tidystats` is the `_bridge` module which defines a serious of functions and decorators that can be used to "wrap" functions imported using `rpy2`. 

!!! note
    This `tidystats` inteface is currently experimental and incomplete. It's stable enough to serious use, but there are certainly some rough edges and missing features that we hope to add in the future

## Key Features & Differences from `pymer4.models`

- Works with `polars` Dataframes instead of `pandas` DataFrames
- Intended to be used in a *functional* style similar to the `tidyverse` and `magittr` pipes in R
- Departs from some coventional objected-orient Python conventions in favor of a more similar API to R

## Current R libraries and functions
Currently these include:

- R base and stats (e.g. `lm`, `anova`)
- lmer4/lmerTest (e.g. `lmer`)
- janitor (e.g. `clean_names`)
- broom (e.g. `tidy`)
- emmeans (e.g. `joint_tests`)

### `base`

| R | `tidystats` | input | output | usage notes |
|---|-------------|-------------|-------|--------------|
| `lm` | `summary` | model | printed summary and results | N/A |

### `broom/broom.mixed`

| R | `tidystats` | input | output | usage notes |
|---|-------------|-------------|-------|--------------|
| `tidy` | `tidy` | `lm`/`glm` output | DataFrame | N/A |
| `glance` | `glance` | `lm`/`glm` output | DataFrame | N/A |
| `augment` | `augment` | `lm`/`glm` output | DataFrame | N/A |

### `emmeans`

| R | `tidystats` | input | output | usage notes |
|---|-------------|-------------|-------|--------------|
| `joint_tests` | `joint_tests` | `lm`/`glm` output | DataFrame | N/A |
| `emmeans` + `contrast` | `emmeans` | `lm`/`glm` output | DataFrame | N/A |

### `janitor`

| R | `tidystats` | input | output | usage notes |
|---|-------------|-------------|-------|--------------|
|`clean_names` | `clean_names` | DataFrame | DataFrame | N/A |
|`get_dupes` | `get_dupes` | DataFrame | DataFrame | N/A |

### `lme4/lmerTest`

| R | `tidystats` | input | output | usage notes |
|---|-------------|-------------|-------|--------------|
|`lmerTest` | `lmer` | DataFrame | TBD | Currently you should pass output to `tidy`, `summary`, etc |

### `stats`

| R | `tidystats` | input | output | usage notes |
|---|-------------|-------------|-------|--------------|
| `lm` | `lm` | DataFrame | TBD | Currently you should pass output to `tidy`, `summary`, etc |
| `glm` | `glm` | DataFrame | TBD | Currently you should pass output to `tidy`, `summary`, etc |
| `anova` | `anova` |  `lm`/`glm` output | DataFrame | N/A |
| `coef` | `coef`/`coefficients` | `lm`/`glm` output | DataFrame | N/A |
| `resid` | `resid`/`residuals` | `lm`/`glm` output | DataFrame | N/A |
| `model_matrix` | `model_matrix` | `lm`/`glm` output | DataFrame | N/A |
