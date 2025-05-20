# Design & Tooling

:::{admonition} Note
:class: note
This page walks-through design and development details that you don't *really* need to know about if you're [just interested in making a quick contribution](./contributing.md)
:::

## Library Architecture

Version `0.9.0` involved a major-rewrite of the `pymer4` internals to follow a more "functional-core, imperative shell" pattern. The heart of the library is really `pymer4.tidystats.bridge` which defines a series of functions and decorators used throughout the library to predictably convert between R and Python types. These are used to bring-in functionality from a variety [R libraries](../api/functions.md) under the broader `pymer4.tidystats` module. 

Sitting on-top of this "functional core" is the `pymer4.models` module which  serves as the "imperative shell" that defines the 4 core classes intended for typical use (`lm`, `glm`, `lmer`, `glmer`). These models follow an inheritence diagram that looks like this:

- `pymer4.models.base.model`: base class that implements most of the core functionality shared by all models
- `pymer4.models.lm.lm`: extends the base class with support for bootstrapping
- `pymer4.models.lmer.lmer`: extends the base class with support for bootstrapping and random-effects
- `pymer4.models.glm.glm`: extends `lm` with support for `family` and `link` functions, parameter expontentiation (`False` by default), and predictions on the `'response'` scale by default
- `pymer4.models.glmer.glmer`: extends `lmer` with support for `family` and `link` functions, parameter expontentiation (`False` by default), and predictions on the `'response'` scale by default

## How models are fit

Calling `.fit()` on a model goes through the following sequence of steps with many pass-through key-word arguments going to the underlying R functions in `pymer4.tidystats`:

1. `._initialize()` to create an R-model object with optional contrasts and weights
2. `._get_design()` to save the model's design-matrix
3. `._get_params()` to estimate fixed-effects terms using `parameters::model_parameters()` with satterthwaite degrees-of-freedom for `lmer()` models
4. `._get_fit_stats()` to get quality-of-fit estimates using `broom/broom.mixed::glance()` 
5. `._get_fits_resids()` to add columns to `.data` using `broom/broom.mixed::augment()`, with predictions (`'fitted'`) values coming from a model's `.predict()` method to support `type_predict = 'response'`

To create new model classes, it's preferable to inherit from one of the existing models and define a new `.fit()` method that calls `super().fit(*args, **kwargs)` plus extra code.


## Development Tooling

### Overview

Since version `0.9.0` `pymer4` uses [Pixi](https://prefix.dev/blog/pixi_for_scientists) to simplify package management, development, and testing orchestrated through a single [`pyproject.toml`](https://packaging.python.org/en/latest/guides/writing-pyproject-toml/). While modern tooling like `uv` works for pure Python packages, `pymer4` needs a properly configured and working R installation with several libraries pre-installed. `pixi` allows us to do this by supporting everything that `uv` can do via `pip` *as well as* everything `conda` can do via `conda-forge`.

[Pixi](https://prefix.dev/blog/pixi_for_scientists) is a modern, extremely fast project-management tool that excels at handling Python environments with mixed dependecies from `conda` and `pip`, while building upon Python standards. In other words using Pixi, **the `pyproject.tml` acts as a single source of truth for *all* of pymer4's dependencies**, including both Python and R packages.

Pixi manages projects in a style similar to popuar Javascript tools like `npm` rather than traditional Anaconda environments and is powered by extremely fast tooling like Rust, `uv` for `pip` packages, and `mamba` for `conda` packages. Using `pixi install`, Pixi creates 1 or more environments in a hidden `.pixi` folder that are *automatically* used for running a variety of [tasks](https://pixi.sh/latest/features/advanced_tasks/), short commands that can be executed with `pixi run taskName` similar to a `Makefile`. These environments are **completeley isolated** just like traditional `conda` environments, but you don't need to manage or switch to them; Pixi handles all that for you based on the configuration in `pyproject.toml` 


### `pyproject.toml`

The various sections of `pyproject.toml` specify:

- `pip` dependencies in `project.dependencies`
- `conda-forge` dependencies in `tool.pixi.dependencies`
- optional development only dependences (`conda-forge`) in `tool.pixi.feature.dev.dependencies`
- the `default` environment and make sure it includes optional dependencies
- various Pixi tasks in `tool.pixi.feature.dev.tasks`

:::{admonition} Note
:class: note
*We hope to replace the last step above with `pixi build` when it [comes out of beta](https://pixi.sh/latest/build/getting_started/) and integrates with `rattler-build`, a replacement for `conda-build`*
:::


### Running pre-configured `pixi` tasks

Installing Pixi is very easy as it has no dependencies and doesn't affect any other Python versions or tools you may already use (e.g. Anaconda). Just copy the following command into your terminal:  

```bash
curl -fsSL https://pixi.sh/install.sh | bash
```

Then after cloning the repo just run

```bash
pixi install
```

This will automatically setup all the project dependencies in an isolated environment and make several *task commands* available to help with development You can see all the ones we've setup using `pixi task list` and run one with `pixi run *cmd*`. We've configured several to make it super easy to run tests, build documentation, and build the conda package itself. Running a task will automatically run it in environment for the project without you having to activate or deactivate anything. You can try them out for yourself as you're working with the code base. 

| Pixi Command |  Description |
|--------------|------------------|
| `pixi run tests` | Runs the full test-suite with `pytest` |
| `pixi run lint` | Runs `ruff` to check for errors and formatting issues |
| `pixi run lint-fix` | Runs `ruff` fix errors and formatting issues, rewriting files |
| `pixi run docs-build` | Builds the documentation |
| `pixi run docs-preview` | Open built docs in the browser |
| `pixi run docs-clean` | Removes any built documentation (good for weird caching errors) |


### Additional `pixi` commands

Just like `pip` or `conda` you can use `pixi` to add/remove packages and environments and have this tracked for you in the `pyproject.toml` and `pixi.lock` files

| Pixi Command | Conda/Pip Equivalent | Description |
|--------------|------------------|-------------|
| `pixi install` | `conda create -n env_name` + `conda install ...` | Creates a new environment and installs all dependencies |
| `pixi add package_name` | `conda install package_name` | Adds a `conda-forge` package to the environment |
| `pixi add --feature dev package_name` | `conda install package_name` | Adds a `conda-forge` package to the `dev` environment |
| `pixi add --pypi package_name` | `pip install package_name` | Adds a `pip` package to the environment |
| `pixi add --pypi --feature dev package_name` | `pip install package_name` | Adds a `pip` package to the `dev` environment |
| `pixi remove --pypi package_name` | `pip uninstall package_name` | Removes a `pip` package from the environment |
| `pixi remove package_name` | `conda remove package_name` | Removes a `conda-forge` package from the environment |
| `pixi run task_name` | `conda run -n env_name command` | Runs a command in the project environment |
| `pixi shell` | `conda activate env_name` | Activates the project environment in the current shell |

## Automatic Testing and Deployment

Automated package testing and deployment are handled by a single `pixi.yml` Github Actions (GA) workflow that does the following:

- Setup up Pixi
- Configure an environment and install runtime & development depenencies
- Runs tests using `pixi run tests`
- Build and deploys docs using `pixi run docs-build`
- Build and deploys the package to the `pre-release` or `main` labels on the `ejolly` channel at Anaconda.org using `conda-build` and `anaconda-client`

A few other notes about automation & deployment

- Conda packages are exclusively `noarch` builds that should run be cross-compatible for macOS & Linux as long as the minimum required Python version is met (Windows not officially supported)
- The configured GA workflow automatically runs on any commits/PRs against the `main` branch as well as on a weekly schedule every Sunday
- Built packages are available on the `ejolly` channel on [Anaconda.org](https://anaconda.org/ejolly/pymer4) under one of the 2 following *labels*
- `main`: built from official Github Release only
- `pre-release`: built from every new commit to the `main` branch on github

:::{admonition} Note
*We hope to have `pymer4` available on the `conda-forge` channel instead of `ejolly` soon!*
:::
