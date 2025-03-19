# How to Contribute

We've tried to simplify development as much as possible to encourage contributions from anyone with experience in Python and/or R. Following the directions on this page to get going quickly. Check-out the [development page](./development.md) for more context and details

## Quick start

1. Install `pixi`

```bash
curl -fsSL https://pixi.sh/install.sh | bash
```

2. Clone the repository and setup a development environment

```bash
git clone https://github.com/YOURFORK/pymer4.git
cd pymer4
pixi install
```

3. Make code/documentation changes with tests and test them

```bash
pixi run tests
```

4. Push your changes to Github and open a pull request!

## Development Tools

- `pixi` for project and environment management
- `pyproject.toml` for specifying, pip, conda, runtime, and development dependencies as well as Pixi tasks
- `conda/meta.yaml` reads from `pyproject.toml` to specify conda build instructions
- `mkdocs` for documentation
- `pytest` for testing
- `VSCode` for development

## Code Guidelines

- TODO - note `ruff`

Please use [google style docstrings](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html/) for documenting all functions, methods, and classes.

Please be sure to include tests with any code submissions and verify
they pass using [pytest](https://docs.pytest.org/en/latest/). To run all
package tests you can use `pytest -s --capture=no` in the project root.
To run specific tests you can point to a file or even a test function
within a file, e.g.
`pytest pymer4/test/test_models.py -k "test_gaussian_lm"`

## Versioning Guidelines

The current `pymer4` scheme is [PEP
440](https://www.python.org/dev/peps/pep-0440/) compliant with two and
only two forms of version strings: `M.N.P` and `M.N.P.devX`. Versions
with the `.devX` designation denote development versions typically on
the `master` branch or `conda` pre-release channel.

This simplifed scheme is not illustrated in the PEP 440 examples, but if
was it would be described as \"major.minor.micro\" with development
releases. To illustrate, the version sequence would look like this:

> ``` bash
> 0.7.0
> 0.7.1.dev0
> 0.7.1.dev1
> 0.7.1
> ```

The third digit(s) in the `pymer4` scheme, i.e. PEP 440 \"micro,\" are
not strictly necessary but are useful for semantically versioned
\"patch\" designations. The `.devX` extension on the other hand denotes
a sequence of incremental work in progress like the alpha, beta,
developmental, release candidate system without the alphabet soup.

## Documentation Guidelines

Documentation is written using [`mkdocs`](https://www.mkdocs.org/) with the following plugins:
- [`mkdocs-material`](https://squidfunk.github.io/mkdocs-material/getting-started/)
- [`mkdocs-jupyter`](https://github.com/danielfrg/mkdocs-jupyter)
- [`mkdocstrings-python`](https://mkdocstrings.github.io/python/)

### Adding new pages & tutorials

- Add new markdown files to `docs/` to create new pages
- Add new jupyter notebooks to `docs/tutorials/` to create new tutorials
- Add the filename(s) to the `nav` section of `mkdocs.yml`
- Use `pixi run docs` to preview changes
