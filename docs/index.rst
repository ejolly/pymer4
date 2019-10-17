Pymer4
======
.. image:: https://travis-ci.org/ejolly/pymer4.svg?branch=master
    :target: https://travis-ci.org/ejolly/pymer4

.. image:: https://badge.fury.io/py/pymer4.svg
    :target: https://badge.fury.io/py/pymer4

.. image:: https://img.shields.io/badge/python-3.6%20%7C%203.7-blue

.. image:: http://joss.theoj.org/papers/10.21105/joss.00862/status.svg
    :target: https://doi.org/10.21105/joss.00862

.. image:: https://zenodo.org/badge/90598701.svg
    :target: https://zenodo.org/record/1523205


:code:`pymer4` is a statistics library for estimating various regression and multi-level models in Python. Love `lme4  <https://cran.r-project.org/web/packages/lme4/index.html>`_ in R, but prefer to work in the scientific Python ecosystem? This package has got you covered!

:code:`pymer4` provides a clean interface that hides the back-and-forth code required when moving between R and Python. In other words, you can work completely in Python, never having to deal with R, but get (most) of lme4's goodness. This is accomplished using `rpy2 <hhttps://rpy2.github.io/doc/latest/html/index.html/>`_ to interface between langauges.

Additionally :code:`pymer4` can fit various additional regression models with some bells, such as robust standard errors, and two-stage regression (summary statistics) models. See the features page for more information.  

**TL;DR** This package is your new *simple* Pythonic drop-in replacement for :code:`lm()` or :code:`glmer()` in R.

.. raw:: html

  <div class="col-md-6">
    <div class="panel panel-default">
      <div class="panel-heading">
        <h3 class="panel-title">Contents</h3>
      </div>
      <div class="panel-body">

.. toctree::
   :maxdepth: 1

   Features <features>
   Installation <installation>
   What's New <new>
   Tutorial <auto_examples/index>
   Lme4 RFX Cheatsheet <rfx_cheatsheet>
   API reference <api>
   Citation <citation>
   Contributing <contributing>


.. raw:: html

      </div>
    </div>
  </div>

