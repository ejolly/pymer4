# How to Contribute

We **always welcome contributions** and have tried to simplify the process of adding new features or fixing bugs to `pymer4`

## 1. Get the code & setup a development environment

Install `pixi`

```bash
curl -fsSL https://pixi.sh/install.sh | bash
```

Fork the `main` branch on github, clone your fork, and install the development dependencies in an isolated Python environment:

  ```bash
  git clone https://github.com/YOURFORK/pymer4.git
  cd pymer4
  pixi install
  ```
## 2. Make code changes and test them

After editing any files with your changes you can run the full test-suite with 

```bash
pixi run tests
```

And check code formatting with

```bash
pixi run lint
```

Or try to fix code formatting

```bash
pixi run lint-fix
```

Or build docs

```bash
pixi run docs-build
```

And preview them

```bash
pixi run docs-preview
```

## 3. Push your changes to Github and open a pull request!

Opening a PR will setup a fresh `pymer4` install and rerun the test-suite with your changes while also ensuring the package can be built. We'll review your changes and request any modifications before accepting a merge!

## General Recommendations

- Please use [google style docstrings](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html/) for documenting all functions, methods, and classes.
- Please make sure to follow proper PEP code-formatting (`pixi run lint` checks this for you using Ruff)
- When adding *new* functionality, please be sure to create *new* tests and verify they pass using `pixi run tests`

## Updating Documentation

Documentation is written using [`jupyterbook`](https://jupyterbook.org/en/stable/intro.html) with the following sphinx extensions:
- [sphinx autodoc](https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html)
- [sphinx autosummary](https://www.sphinx-doc.org/en/master/usage/extensions/autosummary.html)

New documentation can be created by:

- Adding new markdown files to `docs/` to create new pages
- Adding new jupyter notebooks to `docs/tutorials/` to create new tutorials
- Adding the filename(s) to the `nav` section of `docs/_toc.yml`
- Using `pixi run doc-build` and `pixi run docs-preview` to preview changes
