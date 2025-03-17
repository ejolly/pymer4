# Development

Starting from version `0.9.0`, development was overhauled in several noteable ways to aid maintability. This page describes the latest tools we use and you can use them to contribute to `pymer4` on Github.

##`pyproject.toml` & `meta.yaml`

`pymer4` uses a single `pyproject.toml` file instead of the traditional `setup.py` + `requirements.txt` + `requirements-dev.txt` setup. This file acts as a **single source of truth** for build both Pypi compatible packages *and* conda packages. Conda packages are built using the `.conda/meta.yaml` file which read from the `pyproject.toml` file for meta-data and dependencies.

## Pixi

Due to the challenging compatibilitiy issues ensuing from `pymer4`'s cross-language and cross-library design (e.g. R packages like `lme4` are *only* available via `conda` not `pip`), we've decided to go with [`pixi`](https://pixi.sh/latest/) for development and collaboration. Pixi is a tool that makes it much easier to work with *mixed* `conda` and `pip` packages and environments, while building upon standard eco-system tooling like a `pyproject.toml` file.

Pixi works much more like `npm` or `bun` from Javascript than traditional Anaconda environments: all environments and are *co-located* with you package and described by your `pyproject.toml` file. Using a single command `pixi install` you can have a fully-configured development environment for `pymer4` on your own computer that's **totally isolated** from any existing python packages and environments you have. 

### Setup

It's very easy to install Pixi as it has no other dependencies. To get started:  

```bash
curl -fsSL https://pixi.sh/install.sh | bash
```

Then clone the repository and run `pixi install`

You might notice a new hidden `.pixi` directory in which any environments and packages are installed. All the instructions for how to set that up *including* `pymer4`'s dependencies are listed in the `pyproject.toml` file.

### Environment

When you ran `pixi install` Pixi created an isolated environment with all the dependencies listed in `pyproject.toml`. You can activate this environment conda-style using `pix shell` after which commands like `jupyter-book` or `pytest` will become available in your terminal. This is similar to using `conda activate *my_environment*`

### Tasks

Pixi also allows us to define handy commands (like a `Makefile`) that it calls *tasks*. You can see all the ones we've setup using `pixi task list` and run one with `pixi run *cmd*`. We've configured several to make it super easy to run tests, build documentation, and build the conda package itself. Running a task will automatically run it in environment for the project without you having to activate or deactivate anything. You can try them out for yourself as you're working with the code base. 

## Documenation

All documentation is built using [jupyter book](https://jupyterbook.org/en/stable/intro.html). Using `pixi run docs-build` and `pixi run docs-preview` will build the docs for you.

Aside from the `api.md` file which uses special directives to automatically extract function and method docstrings, new documentation can be easily added by create new markdown files or jupyter notebooks

## Github Actions

### Building Documentation

All of `pymer4`'s testing and documentation is executed and built automatically using Github Actions. The workflow called `Docs.yml` uses Pixi to setup an environment, build the documentation, and deploy it using the github pages branch of the repository. You can emulate this locally using `pixi run build-docs`

### Testing & Installable `conda` package

Unfortuantely, until `pixi build` comes out of beta development there are some limitations that force us to use `conda-build`:
- Pixi restricts building to `tool.pixi.project.platforms` specified in the `pyproject.toml` file, but then tries to create all those environments when developing locally on a single machine
- `rattler-build` is under activate development and not full integrated with Pixi yet

Our `Build.yml` uses the specifications in `.conda/meta.yaml` to tell `conda-build` how to build `pymer4`. This `.yaml` file sources all its information about requirements *from* the `pyproject.toml` file. This file also performs several post-build operations:

- sets up a test environment and installs the build
- checks for module importability
- runs `pymer4`'s install test function
- runs `pymer4`'s full testing suite using `pytest`

All of this is handled when `conda-build` executes `.conda/meta.yaml` and doesn't need additional configuration. You can emulate this locally using `pixi run build`

