# How to Contribute

Maintaining this package is tricky because of its inter-language
operability. In particular this requires keeping up with API changes to
Python packages (e.g. pandas), R packages (e.g. lmerTest) as well as
changes in rpy2 (which tend to break between versions), the interface
package between them. For these reasons contributions are **always**
welcome! Checkout the [development roadmap on
Github](https://github.com/users/ejolly/projects/1). Also note the
diagram and explanation below which illustrate how code development
cycles work and how automated deployment is handled through Travis CI.

# Development Cycle and workflow

All automation for testing, documentation, and packaging is handled
through Github Actions. We use separate workflows to handle testing and
packaging.

## Testing

Any pushes or PRs against the `master` branch will trigger the **Tests**
GA workflow. This is a simple workflow that:

-   sets up a `conda` environment with required `R` dependencies
-   installs the latest code from the `master` branch
-   runs tests and builds documentation (as an additional testing layer)

## Packaging Stable Releases

A stable release can be installed from `pip` or from `conda` using the
`-c ejolly` channel flag. Packaging a stable release requires building 3
artifacts:

1.  Conda packages for multiple platforms uploaded to the main `ejolly`
    channel on anaconda cloud
2.  A pip installable package uploaded to Pypi
3.  Documentation site deployed to github pages

To create a new release:

1.  Publish a new release via github
2.  Manually trigger the **Build** and **Build_noarch** workflows and
    enable uploading to the main channel on anaconda, uploading to pypi,
    and deploying documentation

*Note: Previously this process was automated to trigger when a github
release is made, but this seems to be unreliable as the commit hash is
missing and causes runners to not find the built tarballs*

## Packaging Development Releases

Development releases can be install directly from the `master` branch on
github using `pip install git+https://github.com/ejolly/pymer4.git` or
conda using the `-c ejolly/label/pre-release` channel flag.

A development release only includes 1 artifact:

1.  Conda packages for multiple platforms uploaded to the
    `ejolly/label/pre-release` channel on anaconda cloud

Development releases are created the same way as stable releases using
the same **Build** and **Build_noarch** workflows, but choosing the
\"pre-release\" option for uploading to anaconda cloud and disabling
pypi and documentation deploys. The default options for these works flow
will simply build packages but perform no uploading at all which can
useful for testing package builds.

## Updating deployed documentation

To deploy only documentation changes you can use *either* the **Build**
workflow and enable the documentation deploy or the **Docs** workflow
which is a bit faster as it skips packaging building.

# Code Guidelines

Please use the [black](https://black.readthedocs.io/en/stable/) code
formatter for styling code. Any easy way to check if code is formatted
properly is to use a [git pre-commit hook](https://githooks.com/). After
installing black, just create a file called `.git/hooks/pre-commit` and
put the following inside:

> ``` bash
> #!/bin/sh
> black --check .    
> ```

This will prevent the use of the `git commit` command if black notes any
files that have not been formatted. Just format those files and you
should be able to proceed with the commit!

Please use [google style
docstrings](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html/)
for documenting all functions, methods, and classes.

Please be sure to include tests with any code submissions and verify
they pass using [pytest](https://docs.pytest.org/en/latest/). To run all
package tests you can use `pytest -s --capture=no` in the project root.
To run specific tests you can point to a file or even a test function
within a file, e.g.
`pytest pymer4/test/test_models.py -k "test_gaussian_lm"`

# Versioning Guidelines

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

# Documentation Guidelines

Documentation is written with
[sphinx](https://www.sphinx-doc.org/en/master/) using the [bootstrap
theme](https://ryan-roemer.github.io/sphinx-bootstrap-theme/). Tutorial
usage of package features is written using [sphinx
gallery](https://sphinx-gallery.github.io/).

To edit and build docs locally you\'ll need to install these packages
using: `pip install sphinx sphinx_bootstrap_theme sphinx-gallery`. Then
from within the `docs` folder you can run `make html`.

To add new examples to the tutorials simply create a new `.py` file in
the `examples/` directory that begins with `example_`. Any python code
will be executed with outputs when the `make html` command is run and
automatically rendered in the tutorial gallery. You can add non-code
comments with [rST
syntax](https://sphinx-gallery.github.io/syntax.html/) using other files
in the `examples/` directory as a guide.

In addition to making it easy to create standalone examples of package
features, the tutorial gallery serves as another layer of testing for
the package. This can be really useful to ensure previous functionality
is preserved when adding new features or fixing issues.
