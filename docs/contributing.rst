Contributing
============
Maintaining this package is tricky because of its inter-language operability. In particular this requires keeping up with API changes to Python packages (e.g. pandas), R packages (e.g. lmerTest) as well as changes in rpy2 (which tend to break between versions), the interface package between them. For these reasons contributions are **always** welcome! Feel free to ask questions, make suggestions, or contribute changes/additions on `github <https://github.com/ejolly/pymer4/>`_. If you do so, here are some general guidelines for structuring contributions:

Code Guidelines
---------------
Please fork and make pull requests from the `development branch <https://github.com/ejolly/pymer4/tree/dev/>`_ on github. This branch will usually have additions and bug fixes not in master and is easier to integrate with contributions.

Please use the `black <https://black.readthedocs.io/en/stable/>`_ code formatter for styling code.

Please use `google style docstrings <https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html/>`_ for documenting all functions, methods, and classes.

Please be sure to include tests with any code submissions and verify they pass using `pytest <https://docs.pytest.org/en/latest/>`_. To run all package tests you can use :code:`pytest -s --capture=no` in the project root. To run specific tests you can point to a file or even a test function within a file, e.g. :code:`pytest pymer4/test/test_models.py -k "test_gaussian_lm"`

Documentation Guidelines
------------------------
Documentation is written with `sphinx <https://www.sphinx-doc.org/en/master/>`_ using the `bootstrap theme <https://ryan-roemer.github.io/sphinx-bootstrap-theme/>`_. Tutorial usage of package features is written using `sphinx gallery <https://sphinx-gallery.github.io/>`_. 

To edit and build docs locally you'll need to install these packages using: :code:`pip install sphinx sphinx_bootstrap_theme sphinx-gallery`. Then from within the :code:`docs` folder you can run :code:`make html`. 

To add new examples to the tutorials simply create a new :code:`.py` file in the :code:`examples/` directory that begins with :code:`example_`. Any python code will be executed with outputs when the :code:`make html` command is run and automatically rendered in the tutorial gallery. You can add non-code comments with `rST syntax <https://sphinx-gallery.github.io/syntax.html/>`_ using other files in the :code:`examples/` directory as a guide. 

In addition to making it easy to create standalone examples of package features, the tutorial gallery serves as another layer of testing for the package. This can be really useful to ensure previous functionality is preserved when adding new features or fixing issues. 