# How to Contribute

We've tried to simplify development as much as possible to encourage contributions from anyone with experience in Python and/or R. Follow the directions on this page to get going quickly. Check-out the [development page](./development.md) for more context and details

## Quick start

### 1. Install `pixi`

```bash
curl -fsSL https://pixi.sh/install.sh | bash
```

### 2. Clone the repository and setup a development environment

```bash
git clone https://github.com/YOURFORK/pymer4.git
cd pymer4
pixi install
```

### 3. Make code/documentation changes with tests and test them

```bash
pixi run tests
```

### 4. Push your changes to Github and open a pull request!

### General Recommendations

- Please use [google style docstrings](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html/) for documenting all functions, methods, and classes.
- Please make sure to follow proper PEP code-formatting. You can use `pixi run lint` to check or `pixi run lint-fix` to format your files
- Please be sure to include tests with any code submissions and verify they pass using `pixi run tests`

## Updating Documentation

Documentation is written using [`mkdocs`](https://www.mkdocs.org/) with the following plugins:

- [`mkdocs-material`](https://squidfunk.github.io/mkdocs-material/getting-started/)
- [`mkdocs-jupyter`](https://github.com/danielfrg/mkdocs-jupyter)
- [`mkdocstrings-python`](https://mkdocstrings.github.io/python/)

New documentation can be created by:

- Adding new markdown files to `docs/` to create new pages
- Adding new jupyter notebooks to `docs/tutorials/` to create new tutorials
- Adding the filename(s) to the `nav` section of `mkdocs.yml`
- Using `pixi run docs` to preview changes

## Development Tools

We utilize the following tools for development:

- `pixi` for project and environment management
- `pyproject.toml` for specifying, pip, conda, runtime, and development dependencies as well as Pixi tasks
- `conda/meta.yaml` reads from `pyproject.toml` to specify conda build instructions
- `mkdocs` for documentation
- `ruff` for linting and formatting
- `pytest` for testing
- `VSCode` for development
