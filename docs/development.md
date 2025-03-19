# Development & Contributing

Since version `0.9.0` `pymer4` uses [Pixi](https://prefix.dev/blog/pixi_for_scientists) to simplify package management, development, and testing.  
This page provides more background details and context on how Pixi and the other packaging development tools work.  
[Check-out this page](./contributing.md) if you're just interested in the TL;DR on making a contribution.

## Overview

`pymer4` uses a single [`pyproject.toml`](https://packaging.python.org/en/latest/guides/writing-pyproject-toml/) file to configure package setup and requirements instead of the traditional `setup.py` & `requirements.txt` approach. This file also contains additional instructions that specify `setuptools` as the build system as well as additional instructions for Pixi (see below) to manage environments, tasks, and additional development dependencies. While this is sufficient for building a traditional `pip`-installable Python package, `pymer4` also needs a properly configured and working R installation with particular libraries pre-installed. To accomplish this, `pymer4` is distributed as a `conda` package that's built from `conda/meta.yaml`. This file is created by reading meta-data and requirements from `pyproject.toml`.  

Currently, `pymer4` uses `pyproject.toml` to specify:

- `pip` dependencies in `project.dependencies`
- `conda-forge` dependencies in `tool.pixi.dependencies`
- optional development only dependences (`conda-forge`) in `tool.pixi.featre.dev.dependencies`
- the `default` environment and make sure it includes optional dependencies
- various Pixi tasks in `tool.pixi.feature.dev.tasks`

!!! note "Note"
  *We hope to replace the last step above with `pixi build` when it [comes out of beta](https://pixi.sh/latest/build/getting_started/) and integrates with `rattler-build`, a replacement for `conda-build`*

## Pixi

[Pixi](https://prefix.dev/blog/pixi_for_scientists) is a modern, extremely fast project-management tool that excels at handling Python environments with mixed dependecies from `conda` and `pip`, while building upon Python standards. In other words using Pixi, **the `pyproject.tml` acts as a single source of truth for *all* of pymer4's dependencies**, including both Python and R packages.

Pixi manages projects in a style similar to popuar Javascript tools like `npm` rather than traditional Anaconda environments and is powered by extremely fast tooling like Rust, `uv` for `pip` packages, and `mamba` for `conda` packages. Using `pixi install`, Pixi creates 1 or more environments in a hidden `.pixi` folder that are *automatically* used for running a variety of [tasks](https://pixi.sh/latest/features/advanced_tasks/), short commands that can be executed with `pixi run taskName` similar to a `Makefile`. These environments are **completeley isolated** just like traditional `conda` environments, but you don't need to manage or switch to them; Pixi handles all that for you based on the configuration in `pyproject.toml` 

### Tasks

Pixi also allows us to define handy commands (like a `Makefile`) that it calls *tasks*. You can see all the ones we've setup using `pixi task list` and run one with `pixi run *cmd*`. We've configured several to make it super easy to run tests, build documentation, and build the conda package itself. Running a task will automatically run it in environment for the project without you having to activate or deactivate anything. You can try them out for yourself as you're working with the code base. 

### Traditional environment activation
When you ran `pixi install` Pixi created an isolated environment with all the dependencies listed in `pyproject.toml`. You can activate this environment conda-style using `pix shell` after which commands like `jupyter-book` or `pytest` will become available in your terminal. This is similar to using `conda activate *my_environment*`

### Additional Guides & Resources
- [Official Github Action Workflow](https://github.com/prefix-dev/setup-pixi/tree/v0.8.3/)
- [Python Tutorial](https://pixi.sh/latest/tutorials/python/)
- [Initial `pyproject.toml` setup](https://pixi.sh/latest/advanced/pyproject_toml/#python-dependency)
- [`pyproject.toml` reference](https://pixi.sh/latest/reference/pixi_manifest/)

## Getting Started

Installing Pixi is very easy as it has no dependencies and doesn't affect any other Python versions or tools you may already use (e.g. Anaconda). Just copy the following command into your terminal:  

```bash
curl -fsSL https://pixi.sh/install.sh | bash
```

Then clone the repository and run `pixi install` which will configure and create an isolated development environment for you. Once you're setup you can use any of the Pixi tasks we've configured to run tests, build docs, etc or use the Pixi cheatsheet to add/remove dependencies:

### Configured Tasks

The following tasks have already been configured in `pyproject.toml` and can be utilized with `pixi run taskName`

| Pixi Command |  Description |
|--------------|------------------|
| `pixi run test-install` | Runs the `test_install()` function |
| `pixi run tests` | Runs the full test-suite with `pytest` |
| `pixi run lint` | Runs `ruff` to check for errors and formatting issues |
| `pixi run lint-fix` | Runs `ruff` fix errors and formatting issues, rewriting files |
| `pixi run docs-serve` | Preview docs with live-reloading in the browser |
| `pixi run docs-build` | Builds the documentation using `mkdocs build` and outputs it to `site/` |
| `pixi run build-pip` | Uses `pyhon -m build` to make a `pip` package |
| `pixi run build` | Uses `conda-build` to make the package and verify it installs |
| `pixi run build-output` | Gets the location of the built conda package (used by GA during upload) |
| `pixi run build-clean` | Remove previously built package |
| `pixi run upload-pre` | Uploads built package to `pre-release` label; requires `$token` and `$file` environment variables to be set | 
| `pixi run main` | Uploads built package to `main` label; requires `$token` and `$file` environment variables to be set | 

### Additional Commands

You can perform additional actions like adding/removing packages or activating the pixi environment with the following commands:

| Pixi Command | Conda/Pip Equivalent | Description |
|--------------|------------------|-------------|
| `pixi install` | `conda create -n env_name` + `conda install ...` | Creates a new environment and installs all dependencies |
| `pixi add package_name` | `conda install package_name` | Adds a `conda-forge` package to the environment |
| `pixi add --pypi package_name` | `pip install package_name` | Adds a `pip` package to the environment |
| `pixi remove --pypi package_name` | `pip uninstall package_name` | Removes a `pip` package from the environment |
| `pixi remove package_name` | `conda remove package_name` | Removes a `conda-forge` package from the environment |
| `pixi run task_name` | `conda run -n env_name command` | Runs a command in the project environment |
| `pixi shell` | `conda activate env_name` | Activates the project environment |

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

!!! note "Note"
  *We hope to have `pymer4` available on the `conda-forge` channel instead of `ejolly` soon!*
