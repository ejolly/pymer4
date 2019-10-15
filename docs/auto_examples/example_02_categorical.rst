.. note::
    :class: sphx-glr-download-link-note

    Click :ref:`here <sphx_glr_download_auto_examples_example_02_categorical.py>` to download the full example code
.. rst-class:: sphx-glr-example-title

.. _sphx_glr_auto_examples_example_02_categorical.py:


2. Categorical Predictors
=========================

The syntax for handling categorical predictors is **different** between standard regression models/two-stage-models (i.e. :code:`Lm` and :code:`Lm2`) and multi-level models (:code:`Lmer`) in :code:`pymer4`. This is because formula parsing is passed to R for :code:`Lmer` models, but handled by Python for other models. 

Lm and Lm2 Models
-----------------
:code:`Lm` and :code:`Lm2` models use `patsy  <https://patsy.readthedocs.io/en/latest/>`_ to parse model formulae. Patsy is very powerful and has built-in support for handling categorical coding schemes by wrapping a predictor in then :code:`C()` *within* the module formula. Patsy can also perform some pre-processing such as scaling and standardization using special functions like :code:`center()`. Here are some examples.


.. code-block:: default


    # import basic libraries and sample data
    import os
    import pandas as pd
    from pymer4.utils import get_resource_path
    from pymer4.models import Lm

    # IV3 is a categorical predictors with 3 levels in the sample data
    df = pd.read_csv(os.path.join(get_resource_path(), 'sample_data.csv'))







Dummy-coded/Treatment contrasts
+++++++++++++++++++++++++++++++


.. code-block:: default


    # Estimate a model using Treatment contrasts (dummy-coding)
    # with '1.0' as the reference level
    # This is the default of the C() function 
    model = Lm("DV ~ C(IV3, levels=[1.0, 0.5, 1.5])", data=df)
    print(model.fit())





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Formula: DV~C(IV3,levels=[1.0,0.5,1.5])

    Family: gaussian         Estimator: OLS

    Std-errors: non-robust  CIs: standard 95%       Inference: parametric 

    Number of observations: 564      R^2: 0.004      R^2_adj: 0.001

    Log-likelihood: -2728.620        AIC: 5463.241   BIC: 5476.246

    Fixed effects:

                                           Estimate  2.5_ci  97.5_ci     SE   DF  T-stat  P-val  Sig
    Intercept                                42.721  38.334   47.108  2.233  561  19.129  0.000  ***
    C(IV3, levels=[1.0, 0.5, 1.5])[T.0.5]     1.463  -4.741    7.667  3.158  561   0.463  0.643     
    C(IV3, levels=[1.0, 0.5, 1.5])[T.1.5]    -3.419  -9.622    2.785  3.158  561  -1.082  0.280     



Orthogonal Polynomial Contrasts
+++++++++++++++++++++++++++++++


.. code-block:: default


    # Patsy can do this using the Poly argument to the 
    # C() function
    model = Lm('DV ~ C(IV3, Poly)', data=df)
    print(model.fit())





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Formula: DV~C(IV3,Poly)

    Family: gaussian         Estimator: OLS

    Std-errors: non-robust  CIs: standard 95%       Inference: parametric 

    Number of observations: 564      R^2: 0.004      R^2_adj: 0.001

    Log-likelihood: -2728.620        AIC: 5463.241   BIC: 5476.246

    Fixed effects:

                            Estimate  2.5_ci  97.5_ci     SE   DF  T-stat  P-val  Sig
    Intercept                 42.069  39.537   44.602  1.289  561  32.627  0.000  ***
    C(IV3, Poly).Linear       -3.452  -7.838    0.935  2.233  561  -1.546  0.123     
    C(IV3, Poly).Quadratic    -0.798  -5.185    3.588  2.233  561  -0.357  0.721     



Sum-to-zero contrasts
+++++++++++++++++++++


.. code-block:: default


    # Similar to before but with the Sum argument
    model = Lm('DV ~ C(IV3, Sum)', data=df)
    print(model.fit())





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Formula: DV~C(IV3,Sum)

    Family: gaussian         Estimator: OLS

    Std-errors: non-robust  CIs: standard 95%       Inference: parametric 

    Number of observations: 564      R^2: 0.004      R^2_adj: 0.001

    Log-likelihood: -2728.620        AIC: 5463.241   BIC: 5476.246

    Fixed effects:

                        Estimate  2.5_ci  97.5_ci     SE   DF  T-stat  P-val  Sig
    Intercept             42.069  39.537   44.602  1.289  561  32.627  0.000  ***
    C(IV3, Sum)[S.0.5]     2.115  -1.467    5.697  1.823  561   1.160  0.247     
    C(IV3, Sum)[S.1.0]     0.652  -2.930    4.234  1.823  561   0.357  0.721     



Scaling/Centering
+++++++++++++++++


.. code-block:: default


    # Moderation with IV2, but centering IV2 first
    model = Lm('DV ~ center(IV2) * C(IV3, Sum)', data=df)
    print(model.fit())





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Formula: DV~center(IV2)*C(IV3,Sum)

    Family: gaussian         Estimator: OLS

    Std-errors: non-robust  CIs: standard 95%       Inference: parametric 

    Number of observations: 564      R^2: 0.511      R^2_adj: 0.507

    Log-likelihood: -2528.051        AIC: 5068.102   BIC: 5094.113

    Fixed effects:

                                    Estimate  2.5_ci  97.5_ci     SE   DF  T-stat  P-val  Sig
    Intercept                         42.051  40.268   43.833  0.908  558  46.329  0.000  ***
    C(IV3, Sum)[S.0.5]                 0.580  -1.942    3.102  1.284  558   0.452  0.652     
    C(IV3, Sum)[S.1.0]                 0.383  -2.136    2.903  1.282  558   0.299  0.765     
    center(IV2)                        0.746   0.685    0.807  0.031  558  24.012  0.000  ***
    center(IV2):C(IV3, Sum)[S.0.5]     0.050  -0.037    0.137  0.044  558   1.132  0.258     
    center(IV2):C(IV3, Sum)[S.1.0]    -0.057  -0.144    0.029  0.044  558  -1.306  0.192     



Please refer to the `patsy documentation <https://patsy.readthedocs.io/en/latest/categorical-coding.html>`_ for more details when working categorical predictors in :code:`Lm` or :code:`Lm2` models.

Lmer Models
-----------
:code:`Lmer()` models currently have support for handling categorical predictors in one of three ways based on how R's :code:`factor()` works (see the note at the end of this tutorial):

- Dummy-coded factor levels (treatment contrasts) in which each model term is the difference between a factor level and a selected reference level
- Orthogonal polynomial contrasts in which each model term is a polynomial contrast across factor levels (e.g. linear, quadratic, cubic, etc)
- Custom contrasts for each level of a factor, which should be provided in the manner expected by R.

To make re-parameterizing models easier, factor codings are passed as a dictionary to the :code:`factors` argument of a model's :code:`.fit()`. This obviates the need for adjusting data-frame properties as in R. Note that this is **different** from :code:`Lm` and :code:`Lm2` models above which expect factor codings in their formulae (because patsy does). 

Each of these ways also enables you to easily compute post-hoc comparisons between factor levels, as well as interactions between continuous predictors and each factor level. See tutorial 3 for more on post-hoc tests.


.. code-block:: default


    from pymer4.models import Lmer
    # We're going to fit a multi-level logistic regression using the 
    # dichotomous DV_l variable and the same categorical predictor (IV3)
    # as before
    model = Lmer('DV_l ~ IV3 + (IV3|Group)', data=df, family='binomial')







Dummy-coding factors
++++++++++++++++++++
First we'll use dummy-coding/treatment contrasts with 1.0 as the reference level. This will compute two coefficients: 0.5 > 1.0 and 1.5 > 1.0. 


.. code-block:: default


    print(model.fit(factors={
        'IV3': ['1.0', '0.5', '1.5']
    }))






.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    boundary (singular) fit: see ?isSingular 

    Formula: DV_l~IV3+(IV3|Group)

    Family: binomial         Inference: parametric

    Number of observations: 564      Groups: {'Group': 47.0}

    Log-likelihood: -389.003         AIC: 796.006

    Random effects:

                  Name    Var    Std
    Group  (Intercept)  0.022  0.148
    Group       IV30.5  0.060  0.246
    Group       IV31.5  0.038  0.196

                   IV1     IV2  Corr
    Group  (Intercept)  IV30.5  -1.0
    Group  (Intercept)  IV31.5  -1.0
    Group       IV30.5  IV31.5   1.0

    Fixed effects:

                 Estimate  2.5_ci  97.5_ci     SE     OR  OR_2.5_ci  OR_97.5_ci   Prob  Prob_2.5_ci  Prob_97.5_ci  Z-stat  P-val Sig
    (Intercept)    -0.129  -0.419    0.162  0.148  0.879      0.658       1.176  0.468        0.397         0.540  -0.867  0.386    
    IV30.5          0.129  -0.283    0.540  0.210  1.137      0.753       1.716  0.532        0.430         0.632   0.612  0.541    
    IV31.5         -0.128  -0.539    0.283  0.210  0.880      0.583       1.327  0.468        0.368         0.570  -0.612  0.541    



Polynomial contrast coding
++++++++++++++++++++++++++
Second we'll use orthogonal polynomial contrasts. This is accomplished using the :code:`ordered=True` argument and specifying the order of the *linear* contrast in increasing order. R will automatically compute higher order polynomial contrats that are orthogonal to this linear contrast. In this example, since there are 3 factor levels this will result in two polynomial terms: a linear contrast we specify below corresponding to 0.5 < 1.0 < 1.5 and an orthogonal quadratic contrast automatically determined by R, corresponding to 0.5 > 1 < 1.5


.. code-block:: default


    print(model.fit(factors={
        'IV3': ['0.5', '1.0', '1.5']},
        ordered=True
    ))





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    boundary (singular) fit: see ?isSingular 

    boundary (singular) fit: see ?isSingular 

    Formula: DV_l~IV3+(IV3|Group)

    Family: binomial         Inference: parametric

    Number of observations: 564      Groups: {'Group': 47.0}

    Log-likelihood: -389.003         AIC: 796.006

    Random effects:

                  Name    Var    Std
    Group  (Intercept)  0.000  0.000
    Group        IV3.L  0.001  0.035
    Group        IV3.Q  0.032  0.180

                   IV1    IV2  Corr
    Group  (Intercept)  IV3.L   NaN
    Group  (Intercept)  IV3.Q   NaN
    Group        IV3.L  IV3.Q  -1.0

    Fixed effects:

                 Estimate  2.5_ci  97.5_ci     SE     OR  OR_2.5_ci  OR_97.5_ci   Prob  Prob_2.5_ci  Prob_97.5_ci  Z-stat  P-val Sig
    (Intercept)    -0.128  -0.294    0.037  0.085  0.879      0.745       1.038  0.468        0.427         0.509  -1.518  0.129    
    IV3.L          -0.182  -0.469    0.106  0.147  0.834      0.626       1.112  0.455        0.385         0.526  -1.238  0.216    
    IV3.Q           0.000  -0.292    0.292  0.149  1.000      0.747       1.339  0.500        0.428         0.572   0.001  1.000    



Custom contrasts
++++++++++++++++
:code:`Lmer` models can also take custom factor contrasts based on how they are expected by R (see the note at the end of this tutorial for how contrasts work in R). Remember that there can be at most k-1 model terms representing any k level factor without over-parameterizing a model. If you specify a custom contrast, R will generate set of orthogonal contrasts for the rest of your model terms. 


.. code-block:: default


    # Compare level '1.0' to the mean of levels '0.5' and '1.5'
    # and let R determine the second contrast orthogonal to it

    print(model.fit(factors={
        'IV3': {'1.0': 1, '0.5': -.5, '1.5': -.5}
    }))





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    boundary (singular) fit: see ?isSingular 

    boundary (singular) fit: see ?isSingular 

    boundary (singular) fit: see ?isSingular 

    Formula: DV_l~IV3+(IV3|Group)

    Family: binomial         Inference: parametric

    Number of observations: 564      Groups: {'Group': 47.0}

    Log-likelihood: -389.003         AIC: 796.006

    Random effects:

                  Name    Var    Std
    Group  (Intercept)  0.022  0.148
    Group       IV30.5  0.060  0.246
    Group       IV31.5  0.038  0.196

                   IV1     IV2  Corr
    Group  (Intercept)  IV30.5  -1.0
    Group  (Intercept)  IV31.5  -1.0
    Group       IV30.5  IV31.5   1.0

    Fixed effects:

                 Estimate  2.5_ci  97.5_ci     SE     OR  OR_2.5_ci  OR_97.5_ci   Prob  Prob_2.5_ci  Prob_97.5_ci  Z-stat  P-val Sig
    (Intercept)    -0.129  -0.419    0.162  0.148  0.879      0.658       1.176  0.468        0.397         0.540  -0.867  0.386    
    IV30.5          0.129  -0.283    0.540  0.210  1.137      0.753       1.716  0.532        0.430         0.632   0.612  0.541    
    IV31.5         -0.128  -0.539    0.283  0.210  0.880      0.583       1.327  0.468        0.368         0.570  -0.612  0.541    



User-created contrasts (without R)
++++++++++++++++++++++++++++++++++
Another option available to you is fitting a model with *only* your desired contrast(s) rather than a full set of k-1 contrasts. Contrary to how statistics is usually taught, you don't ever *have to* include a full set of k-1 contrasts for a k level factor! The upside to doing this is that you won't need to rely on R to compute anything for you (aside from the model fit), and you will have a model with exactly the number of terms as contrasts you desire, giving you complete control. The downside is that post-hoc tests will no longer be available (see tutorial 3 for more information on post-hoc tests), but it's unlikely you're doing post-hoc tests if you are computing a subset of specific contrasts anyway. This is also a useful approach if you don't want to use patsy's formula syntax with :code:`Lm` and :code:`Lm2` as noted above.

This can be accomplished by creating new columns in your dataframe to test specific hypotheses and is trivial to do with pandas `map <https://pandas.pydata.org/pandas-docs/version/0.25/reference/api/pandas.Series.map.html/>`_ and `assign <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.assign.html/>`_ methods. For example, here we manually compute a linear contrast by creating a new column in our dataframe and treating it as a continuous variable.


.. code-block:: default


    # Create a new column in the dataframe with a custom (linear) contrast
    df = df.assign(
        IV3_custom_lin=df['IV3'].map({
            0.5: -1,
            1.0: 0,
            1.5: 1
        })
    )
    print(df.head())





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

       Group   IV1  DV_l         DV       IV2  IV3  IV3_custom_lin
    0      1  20.0     0   7.936508  4.563492  0.5              -1
    1      1  20.0     0  15.277778  0.000000  1.0               0
    2      1  20.0     1   0.000000  0.000000  1.5               1
    3      1  20.0     1   9.523810  0.000000  0.5              -1
    4      1  12.5     0   0.000000  0.000000  1.0               0



Now we can use this variable as a continuous predictor without the need for the :code:`factors` argument. Notice how the z-stat and p-value of the estimate are the same as the linear polynomial contrast estimated above. The coefficients differ in scale only because R uses [~-0.707, ~0, ~0.707] for its polynomial contrasts rather than [-1, 0, 1] like we did.


.. code-block:: default


    # Estimate model
    model = Lmer('DV_l ~ IV3_custom_lin + (IV3_custom_lin|Group)', data=df, family='binomial')
    print(model.fit())





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    boundary (singular) fit: see ?isSingular 

    Formula: DV_l~IV3_custom_lin+(IV3_custom_lin|Group)

    Family: binomial         Inference: parametric

    Number of observations: 564      Groups: {'Group': 47.0}

    Log-likelihood: -389.016         AIC: 788.031

    Random effects:

                     Name  Var  Std
    Group     (Intercept)  0.0  0.0
    Group  IV3_custom_lin  0.0  0.0

                   IV1             IV2  Corr
    Group  (Intercept)  IV3_custom_lin   NaN

    Fixed effects:

                    Estimate  2.5_ci  97.5_ci     SE    OR  OR_2.5_ci  OR_97.5_ci   Prob  Prob_2.5_ci  Prob_97.5_ci  Z-stat  P-val Sig
    (Intercept)       -0.128  -0.294    0.037  0.085  0.88      0.745       1.038  0.468        0.427         0.509  -1.517  0.129    
    IV3_custom_lin    -0.128  -0.331    0.075  0.104  0.88      0.718       1.077  0.468        0.418         0.519  -1.239  0.215    



A note on how contrasts in R work
---------------------------------
.. note::
  This is just for folks curious about how contrasts in R work

Specifying multiple custom contrasts in R has always been a point of confusion amongst users. This because the :code:`contrasts()` command in R doesn't actually expect contrast weights (i.e. a design matrix) as one would intuit. Rather, it is made for generating contrast coding schemes which are the inverse of the contrast weight matrix. For a longer explanation with examples see `this reference <https://rstudio-pubs-static.s3.amazonaws.com/65059_586f394d8eb84f84b1baaf56ffb6b47f.html>`_ and `this reference <https://github.com/ejolly/R/blob/master/Guides/Contrasts_in_R.md>`_. For these situations pymer4 offers a few utility functions to convert between these matrix types if desired in :code:`pymer4.utils`: :code:`R2con()` and :code:`con2R()`.


.. _sphx_glr_download_auto_examples_example_02_categorical.py:


.. only :: html

 .. container:: sphx-glr-footer
    :class: sphx-glr-footer-example



  .. container:: sphx-glr-download

     :download:`Download Python source code: example_02_categorical.py <example_02_categorical.py>`



  .. container:: sphx-glr-download

     :download:`Download Jupyter notebook: example_02_categorical.ipynb <example_02_categorical.ipynb>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
