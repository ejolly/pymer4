Pymer4
======
.. image:: https://github.com/ejolly/pymer4/actions/workflows/CI.yml/badge.svg
    :target: https://github.com/ejolly/pymer4/actions/workflows/CI.yml

.. image:: https://badge.fury.io/py/pymer4.svg
    :target: https://badge.fury.io/py/pymer4

.. image:: https://anaconda.org/ejolly/pymer4/badges/version.svg
   :target: https://anaconda.org/ejolly/pymer4

.. image:: https://anaconda.org/ejolly/pymer4/badges/platforms.svg   
  :target: https://anaconda.org/ejolly/pymer4
  
.. image:: https://pepy.tech/badge/pymer4
  :target: https://pepy.tech/project/pymer4


.. image:: http://joss.theoj.org/papers/10.21105/joss.00862/status.svg
    :target: https://doi.org/10.21105/joss.00862

.. image:: https://zenodo.org/badge/90598701.svg
    :target: https://zenodo.org/record/1523205

.. image:: https://img.shields.io/badge/python-3.6%20%7C%203.7%20%7C%203.8-blue

.. raw:: html

  <br />

.. image:: https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat
  :target: https://github.com/ejolly/pymer4/issues

:code:`pymer4` is a statistics library for estimating various regression and multi-level models in Python. Love `lme4  <https://cran.r-project.org/web/packages/lme4/index.html>`_ in R, but prefer to work in the scientific Python ecosystem? This package has got you covered!

:code:`pymer4` provides a clean interface that hides the back-and-forth code required when moving between R and Python. In other words, you can work completely in Python, never having to deal with R, but get (most) of lme4's goodness. This is accomplished using `rpy2 <https://rpy2.github.io/doc/latest/html/index.html/>`_ to interface between langauges.

Additionally :code:`pymer4` can fit various additional regression models with some bells, such as robust standard errors, and two-stage regression (summary statistics) models. See the features page for more information.  

**TL;DR** This package is your new *simple* Pythonic drop-in replacement for :code:`lm()` or :code:`glmer()` in R.

For an example of what's possible check out the tutorials or `this blog post <https://eshinjolly.com/2019/02/18/rep_measures/>`_ comparing different modeling strategies for clustered/repeated-measures data.

.. raw:: html

  <div class="col-md-12">
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

Publications
++++++++++++
:code:`pymer4` has been used to analyze data is several publications including but not limited to:

- Jolly, E., Sadhukha, S., & Chang, L.J. (in press). Custom-molded headcases have limited efficacy in reducing head motion during naturalistic fMRI expreiments. *NeuroImage*. 
- Sharon, G., Cruz, N. J., Kang, D. W., et al. (2019). Human gut microbiota from autism spectrum disorder promote behavioral symptoms in mice. *Cell*, 177(6), 1600-1618.
- Urbach, T. P., DeLong, K. A., Chan, W. H., & Kutas, M. (2020). An exploratory data analysis of word form prediction during word-by-word reading. *Proceedings of the National Academy of Sciences*, 117(34), 20483-20494.
- Chen, P. H. A., Cheong, J. H., Jolly, E., Elhence, H., Wager, T. D., & Chang, L. J. (2019). Socially transmitted placebo effects. *Nature Human Behaviour*, 3(12), 1295-1305.

Citing
++++++
If you use :code:`pymer4` in your own work, please cite:  

Jolly, (2018). Pymer4: Connecting R and Python for Linear Mixed Modeling. *Journal of Open Source Software*, 3(31), 862, https://doi.org/10.21105/joss.00862
