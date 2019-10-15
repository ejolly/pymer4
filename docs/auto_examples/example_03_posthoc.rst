.. note::
    :class: sphx-glr-download-link-note

    Click :ref:`here <sphx_glr_download_auto_examples_example_03_posthoc.py>` to download the full example code
.. rst-class:: sphx-glr-example-title

.. _sphx_glr_auto_examples_example_03_posthoc.py:


3. ANOVA tables and post-hoc comparisons
========================================

.. note::
  ANOVAs and post-hoc tests are only available for :code:`Lmer` models estimated using the :code:`factors` argument of :code:`model.fit()` and rely on implementations in R

In the previous tutorial where we looked at categorical predictors, behind the scenes :code:`pymer4` was using the :code:`factor` functionality in R. This means the output of :code:`model.fit()` looks a lot like :code:`summary()` in R applied to a model with categorical predictors. But what if we want to compute an F-test across *all levels* of our categorical predictor? 

:code:`pymer4` makes this easy to do, and makes it easy to ensure Type III sums of squares infereces are valid. It also makes it easy to follow up omnibus tests with post-hoc pairwise comparisons. 

ANOVA tables and orthogonal contrasts
-------------------------------------
Because ANOVA is just regression, :code:`pymer4` can estimate ANOVA tables with F-results using the :code:`.anova()` method on a fitted model. This will compute a Type-III SS table given the coding scheme provided when the model was initially fit. Based on the distribution of data across factor levels and the specific coding-scheme used, this may produce invalid Type-III SS computations. For this reason the :code:`.anova()` method has a :code:`force-orthogonal=True` argument that will reparameterize and refit the model using orthogonal polynomial contrasts prior to computing an ANOVA table.

Here we first estimate a mode with dummy-coded categories and suppress the summary output of :code:`.fit()`. Then we use :code:`.anova()` to examine the F-test results. 


.. code-block:: default


    # import basic libraries and sample data
    import os
    import pandas as pd
    from pymer4.utils import get_resource_path
    from pymer4.models import Lmer

    # IV3 is a categorical predictors with 3 levels in the sample data
    df = pd.read_csv(os.path.join(get_resource_path(), 'sample_data.csv'))

    # # We're going to fit a multi-level regression using the 
    # categorical predictor (IV3) which has 3 levels
    model = Lmer('DV ~ IV3 + (1|Group)', data=df)

    # Using dummy-coding; suppress summary output
    model.fit(factors={
        'IV3': ['1.0', '0.5', '1.5']
    }, summarize=False)

    # Get ANOVA table
    print(model.anova())





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    SS Type III Analysis of Variance Table with Satterthwaite approximated degrees of freedom:
    (NOTE: Using original model contrasts, orthogonality not guaranteed)
                  SS           MS  NumDF  DenomDF    F-stat     P-val Sig
    IV3  2359.778135  1179.889067      2    515.0  5.296284  0.005287  **



Type III SS inferences will only be valid if data are fully balanced across levels or if contrasts between levels are orthogonally coded and sum to 0. Below we tell :code:`pymer4` to respecify our contrasts to ensure this before estimating the ANOVA. :code:`pymer4` also saves the last set of contrasts used priory to forcing orthogonality. 

Because the sample data is balanced across factor levels and there are not interaction terms, in this case orthogonal contrast coding doesn't change the results.


.. code-block:: default


    # Get ANOVA table, but this time force orthogonality 
    # for valid SS III inferences
    # In this case the data are balanced so nothing changes
    print(model.anova(force_orthogonal=True))





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    SS Type III Analysis of Variance Table with Satterthwaite approximated degrees of freedom:
    (NOTE: Model refit with orthogonal polynomial contrasts)
                  SS           MS  NumDF     DenomDF    F-stat     P-val Sig
    IV3  2359.778135  1179.889067      2  515.000001  5.296284  0.005287  **




.. code-block:: default


    # Checkout current contrast scheme (for first contrast)
    # Notice how it's simply a linear contrast across levels
    print(model.factors) 





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    {'IV3': ['0.5', '1.0', '1.5']}




.. code-block:: default


    # Checkout previous contrast scheme 
    # which was a treatment contrast with 1.0
    # as the reference level
    print(model.factors_prev_)





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    {'IV3': ['1.0', '0.5', '1.5']}



Marginal estimates and post-hoc comparisons
-------------------------------------------
:code:`pymer4` leverages the :code:`emmeans` package in order to compute marginal estimates ("cell means" in ANOVA lingo) and pair-wise comparisons of models that contain categorical terms and/or interactions. This can be performed by using the :code:`.post_hoc()` method on fitted models. Let's see an example: 

First we'll quickly create a second categorical IV to demo with and estimate a 3x3 ANOVA to get main effects and the interaction.


.. code-block:: default


    # Fix the random number generator 
    # for reproducibility
    import numpy as np
    np.random.seed(10)

    # Create a new categorical variable with 3 levels
    df = df.assign(IV4=np.random.choice(['1', '2', '3'], size=df.shape[0]))

    # Estimate model with orthogonal polynomial contrasts
    model = Lmer('DV ~ IV4*IV3 + (1|Group)', data=df)
    model.fit(factors={
        'IV4': ['1', '2', '3'],
        'IV3': ['1.0', '0.5', '1.5']},
        ordered=True,
        summarize=False
    )
    # Get ANOVA table
    # We can ignore the note in the output because
    # we manually specified polynomial contrasts
    print(model.anova())





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    SS Type III Analysis of Variance Table with Satterthwaite approximated degrees of freedom:
    (NOTE: Using original model contrasts, orthogonality not guaranteed)
                      SS           MS  NumDF     DenomDF    F-stat     P-val Sig
    IV4       449.771051   224.885525      2  510.897775  1.006943  0.366058    
    IV3      2486.124318  1243.062159      2  508.993080  5.565910  0.004063  **
    IV4:IV3   553.852530   138.463132      4  511.073624  0.619980  0.648444    



Example 1
~~~~~~~~~
Compare each level of IV3 to each other level of IV3, *within* each level of IV4. Use default Tukey HSD p-values.


.. code-block:: default


    # Compute post-hoc tests
    marginal_estimates, comparisons = model.post_hoc(marginal_vars='IV3', grouping_vars='IV4')

    # "Cell" means of the ANOVA
    print(marginal_estimates)





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    P-values adjusted by tukey method for family of 3 estimates
       IV3 IV4  Estimate  2.5_ci  97.5_ci     SE      DF
    1  1.0   1    42.554  33.778   51.330  4.398  68.140
    2  0.5   1    45.455  36.644   54.266  4.417  69.299
    3  1.5   1    40.904  32.196   49.612  4.361  65.943
    4  1.0   2    42.092  33.301   50.882  4.406  68.609
    5  0.5   2    41.495  32.829   50.161  4.339  64.626
    6  1.5   2    38.786  29.961   47.612  4.425  69.746
    7  1.0   3    43.424  34.741   52.107  4.348  65.149
    8  0.5   3    46.008  37.261   54.755  4.383  67.208
    9  1.5   3    38.119  29.384   46.854  4.376  66.801




.. code-block:: default


    # Pairwise comparisons
    print(comparisons)





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

        Contrast IV4  Estimate  2.5_ci  97.5_ci     SE       DF  T-stat  P-val Sig
    1  1.0 - 0.5   1    -2.901  -9.523    3.721  2.817  510.016  -1.030  0.558    
    2  1.0 - 1.5   1     1.650  -4.750    8.050  2.723  510.137   0.606  0.817    
    3  0.5 - 1.5   1     4.552  -1.951   11.054  2.766  510.267   1.645  0.228    
    4  1.0 - 0.5   2     0.596  -5.749    6.942  2.700  510.249   0.221  0.973    
    5  1.0 - 1.5   2     3.305  -3.387    9.998  2.847  510.883   1.161  0.477    
    6  0.5 - 1.5   2     2.709  -3.749    9.166  2.747  510.732   0.986  0.586    
    7  1.0 - 0.5   3    -2.584  -8.893    3.725  2.684  510.213  -0.963  0.601    
    8  1.0 - 1.5   3     5.305  -1.006   11.615  2.685  510.710   1.976  0.119    
    9  0.5 - 1.5   3     7.889   1.437   14.340  2.745  510.663   2.874  0.012   *



Example 2
~~~~~~~~~
Compare each unique IV3,IV4 "cell mean" to every other IV3,IV4 "cell mean" and used FDR correction for multiple comparisons:


.. code-block:: default



    # Compute post-hoc tests
    marginal_estimates, comparisons = model.post_hoc(marginal_vars=['IV3', 'IV4'], p_adjust='fdr')

    # Pairwise comparisons
    print(comparisons)





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    P-values adjusted by fdr method for 36 comparisons
             Contrast  Estimate  2.5_ci  97.5_ci     SE       DF  T-stat  P-val Sig
    1   1.0,1 - 0.5,1    -2.901 -11.957    6.155  2.817  510.016  -1.030  0.535    
    2   1.0,1 - 1.5,1     1.650  -7.102   10.403  2.723  510.137   0.606  0.726    
    3   1.0,1 - 1.0,2     0.463  -8.657    9.582  2.837  511.103   0.163  0.871    
    4   1.0,1 - 0.5,2     1.059  -7.649    9.766  2.709  510.435   0.391  0.835    
    5   1.0,1 - 1.5,2     3.768  -5.364   12.899  2.841  510.737   1.326  0.473    
    6   1.0,1 - 1.0,3    -0.870  -9.659    7.918  2.734  510.723  -0.318  0.869    
    7   1.0,1 - 0.5,3    -3.454 -12.306    5.398  2.754  509.926  -1.254  0.473    
    8   1.0,1 - 1.5,3     4.435  -4.426   13.296  2.757  510.425   1.609  0.390    
    9   0.5,1 - 1.5,1     4.552  -4.341   13.444  2.766  510.267   1.645  0.390    
    10  0.5,1 - 1.0,2     3.364  -5.732   12.460  2.829  510.264   1.189  0.493    
    11  0.5,1 - 0.5,2     3.960  -4.883   12.803  2.751  510.486   1.440  0.446    
    12  0.5,1 - 1.5,2     6.669  -2.568   15.906  2.873  510.672   2.321  0.186    
    13  0.5,1 - 1.0,3     2.031  -6.796   10.858  2.746  510.241   0.740  0.637    
    14  0.5,1 - 0.5,3    -0.552  -9.603    8.498  2.815  510.401  -0.196  0.869    
    15  0.5,1 - 1.5,3     7.336  -1.568   16.241  2.770  509.937   2.648  0.118    
    16  1.5,1 - 1.0,2    -1.188 -10.044    7.669  2.755  510.808  -0.431  0.827    
    17  1.5,1 - 0.5,2    -0.591  -9.041    7.858  2.628  510.149  -0.225  0.869    
    18  1.5,1 - 1.5,2     2.117  -6.937   11.172  2.817  511.496   0.752  0.637    
    19  1.5,1 - 1.0,3    -2.520 -11.037    5.996  2.649  510.392  -0.951  0.535    
    20  1.5,1 - 0.5,3    -5.104 -13.818    3.610  2.711  510.376  -1.883  0.362    
    21  1.5,1 - 1.5,3     2.785  -5.986   11.555  2.728  511.139   1.021  0.535    
    22  1.0,2 - 0.5,2     0.596  -8.082    9.274  2.700  510.249   0.221  0.869    
    23  1.0,2 - 1.5,2     3.305  -5.848   12.458  2.847  510.883   1.161  0.493    
    24  1.0,2 - 1.0,3    -1.333 -10.235    7.570  2.769  511.440  -0.481  0.811    
    25  1.0,2 - 0.5,3    -3.916 -12.888    5.055  2.791  510.691  -1.403  0.446    
    26  1.0,2 - 1.5,3     3.972  -4.883   12.828  2.755  510.379   1.442  0.446    
    27  0.5,2 - 1.5,2     2.709  -6.123   11.540  2.747  510.732   0.986  0.535    
    28  0.5,2 - 1.0,3    -1.929 -10.318    6.460  2.610  510.175  -0.739  0.637    
    29  0.5,2 - 0.5,3    -4.513 -13.207    4.181  2.705  510.802  -1.669  0.390    
    30  0.5,2 - 1.5,3     3.376  -5.172   11.924  2.659  510.356   1.270  0.473    
    31  1.5,2 - 1.0,3    -4.638 -13.457    4.181  2.743  510.454  -1.691  0.390    
    32  1.5,2 - 0.5,3    -7.222 -16.183    1.740  2.788  510.132  -2.590  0.118    
    33  1.5,2 - 1.5,3     0.667  -8.475    9.810  2.844  511.638   0.235  0.869    
    34  1.0,3 - 0.5,3    -2.584 -11.212    6.044  2.684  510.213  -0.963  0.535    
    35  1.0,3 - 1.5,3     5.305  -3.325   13.935  2.685  510.710   1.976  0.351    
    36  0.5,3 - 1.5,3     7.889  -0.935   16.712  2.745  510.663   2.874  0.118    



Example 3
~~~~~~~~~
For this example we'll estimate a more complicated ANOVA with 1 continuous IV and 2 categorical IVs with 3 levels each. This is the same model as before but with IV2 thrown into the mix. Now, pairwise comparisons reflect changes in the *slope* of the continuous IV (IV2) between levels of the categorical IVs (IV3 and IV4).

First let's get the ANOVA table


.. code-block:: default

    model = Lmer('DV ~ IV2*IV3*IV4 + (1|Group)', data=df)
    # Only need to polynomial contrasts for IV3 and IV4
    # because IV2 is continuous
    model.fit(factors={
        'IV4': ['1', '2', '3'],
        'IV3': ['1.0', '0.5', '1.5']},
        ordered=True,
        summarize=False
    )

    # Get ANOVA table
    print(model.anova())





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    SS Type III Analysis of Variance Table with Satterthwaite approximated degrees of freedom:
    (NOTE: Using original model contrasts, orthogonality not guaranteed)
                           SS            MS  NumDF     DenomDF      F-stat         P-val  Sig
    IV2          46010.245471  46010.245471      1  535.763367  306.765451  1.220547e-54  ***
    IV3            726.318000    363.159000      2  500.573997    2.421301  8.984551e-02    .
    IV4            143.379932     71.689966      2  502.297291    0.477981  6.203159e-01     
    IV2:IV3        613.455876    306.727938      2  500.403443    2.045056  1.304528e-01     
    IV2:IV4          4.914900      2.457450      2  502.300664    0.016385  9.837494e-01     
    IV3:IV4         92.225327     23.056332      4  502.950771    0.153724  9.612985e-01     
    IV2:IV3:IV4    368.085569     92.021392      4  503.354865    0.613537  6.530638e-01     



Now we can compute the pairwise difference in slopes 


.. code-block:: default


    # Compute post-hoc tests with bonferroni correction
    marginal_estimates, comparisons = model.post_hoc(marginal_vars='IV2',
                                        grouping_vars=['IV3', 'IV4'],
                                        p_adjust='bonf')

    # Pairwise comparisons
    print(comparisons)




.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    P-values adjusted by bonf method for 3 comparisons
        Contrast IV4  Estimate  2.5_ci  97.5_ci     SE       DF  T-stat  P-val Sig
    1  1.0 - 0.5   1    -0.053  -0.254    0.147  0.084  502.345  -0.638  1.000    
    2  1.0 - 1.5   1    -0.131  -0.313    0.050  0.076  502.494  -1.734  0.250    
    3  0.5 - 1.5   1    -0.078  -0.278    0.122  0.083  502.821  -0.933  1.000    
    4  1.0 - 0.5   2    -0.038  -0.210    0.134  0.072  501.096  -0.526  1.000    
    5  1.0 - 1.5   2     0.002  -0.184    0.189  0.078  502.745   0.031  1.000    
    6  0.5 - 1.5   2     0.040  -0.142    0.222  0.076  502.836   0.530  1.000    
    7  1.0 - 0.5   3    -0.134  -0.329    0.061  0.081  502.956  -1.646  0.301    
    8  1.0 - 1.5   3    -0.110  -0.302    0.083  0.080  502.109  -1.368  0.516    
    9  0.5 - 1.5   3     0.024  -0.166    0.214  0.079  502.538   0.304  1.000    




.. _sphx_glr_download_auto_examples_example_03_posthoc.py:


.. only :: html

 .. container:: sphx-glr-footer
    :class: sphx-glr-footer-example



  .. container:: sphx-glr-download

     :download:`Download Python source code: example_03_posthoc.py <example_03_posthoc.py>`



  .. container:: sphx-glr-download

     :download:`Download Jupyter notebook: example_03_posthoc.ipynb <example_03_posthoc.ipynb>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
