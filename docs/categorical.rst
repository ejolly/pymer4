Categorical Predictors
=======================
The syntax for handling categorical predictors is **different** between standard regression models and multi-level models in :code:`pymer4`. This is because formula parsing is passed to R for multi-level models, but handled by Python for standard regression models. Please note the differences below.

Standard Regression Models
--------------------------
:code:`Lm()` models uses `patsy  <https://patsy.readthedocs.io/en/latest/>`_ to parse model formulae for *standard regression analyses*. Patsy is very powerful and has built-in support for handling categorical coding schemes (e.g. wrapping predictors in the :code:`C()` syntax). Patsy can also perform some pre-processing such as scaling and standardization.

A few common parameterizations include:

.. code-block:: python

    # Assuming a categorical predictor IV1 has 3 levels named, 'a', 'b', 'c'

    # Treatment contrasts (dummy-coding) with 'a' as the reference level
    model = Lm('DV ~ C(IV, levels=['a','b','c'])',data=data)
    model.fit()

    # Polynomial contrasts
    model = Lm('DV ~ C(IV, Poly)',data=data)
    model.fit()

    # Sum-to-zero contrasts
    model = Lm('DV ~ C(IV, Sum)',data=data)
    model.fit()

Please refer to the `patsy documentation <https://patsy.readthedocs.io/en/latest/categorical-coding.html>`_ for more details when working with standard regression models that have categorical predictors.

Multi-level Models
------------------
:code:`Lmer()` models currently have support for handling categorical predictors in one of three ways based on how R's :code:`factor()` works:

1. Dummy-coded factor levels (treatment contrasts) in which each model term is the difference between a factor level and a selected reference level
2. Orthogonal polynomial contrasts in which each model term is a polynomial contrast across factor levels (e.g. linear, quadratic, cubic, etc)
3. Custom contrasts for each level of a factor, which should be provide in the manner expected by R.

To make re-parameterizing models easier, factor codings are passed as an argument to a model's :code:`fit` method. This obviates the need for adjusting data-frame properties as in R.

Dummy coding factors
--------------------
To dummy code factors (i.e. use treatment contrasts) simply pass a model's :code:`fit` method a dictionary with keys containing model terms to be treated as factors, and values as a list of unique factor levels.

The *first* term in the list will be treated as the reference level.

.. code-block:: python

    # Initialize a multi-level-model
    # We're going to treat IV3 as categorical predictor during fitting
    model = Lmer('DV_l ~ IV3 + (IV3|Group)',data=df,family='binomial')

    # IV3 has 3 levels. Use dummy codes and set 1.0 as the reference level
    model.fit(factors={
        'IV3':['1.0','0.5','1.5'],
        })

Polynomial contrast coding
--------------------------
Representing factors as polynomial contrasts is very similar. Like before, simply pass a model's :code:`fit` method a dictionary with keys containing model terms to be treated as factors, and values as a list of unique factor levels, as well as the :code:`ordered = True` flag.

This will treat the order of list items as the order of factor levels for the *linear* polynomial term. Each higher order polynomial (if required) will be automatically calculated orthogonally with respect to the linear contrast.

.. code-block:: python

    # Initialize a model using a categorical predictor
    model = Lmer('DV_l ~ IV3 + (IV3|Group)',data=df,family='binomial')

    # Using polynomial coding for IV3, since there are 3 factor levels this will result in two polynomial terms: a linear and quadratic contrast
    # Setup the linear contrast to test: 0.5 < 1.0 < 1.5
    # This will produce a quadratic term automatically testing: 0.5 > 1 < 1.5
    model.fit(factors={
        'IV3':['1.5','1.0','0.5'],
        ordered = True
        })

Custom contrasts
---------------------------
:code:`Lmer()` models can also take custom factor contrasts based on how they are expected by R. Remember that there can be at most k-1 model terms representing any k level factor without over-parameterizing a model. If all that's desired is a specific contrast between factor levels (e.g. rather than all possible k-1 contrasts) you have 2 options:

1) Specify a custom contrast using the syntax below (also available in the :code:`model.fit()` method help). This will guarantee that one of your model terms reflects your desired contrast, and R will generate set of orthogonal contrasts for the rest of your model terms.

**Example**
Compare level '1.0' in 'IV3' to the mean of levels '0.5' and '1.5'

.. code-block:: python

    model.fit(factors={
        'IV3': {'1.0': 1, '0.5': -.5, '1.5': -.5}
    })

2) Fit a model with *only* the desired contrast rather than a full set of k-1 contrasts. Contrary to how statistics is usually taught, you don't ever *have to* include a full set of k-1 contrasts for a k level factor! Follow the directions below to do this (section: "simpler" custom contrasts). The upside is you won't need to rely on R to compute anything for you (aside from the model fit), and you will have a model with exactly the number of terms as contrasts you desire giving you complete control. The downside is that :code:`model.post_hoc()` methods will not be able to do pairwise comparisons as they rely on R's internal representation of factor levels.

(Simpler) Custom contrasts
--------------------------
Testing specific parameterizations without relying on R's factor coding is often easier done by creating new columns in a dataframe with specific coding schemes. These new columns can be utilized within models to test specific hypotheses. However, the downside of this approach is that the :code:`model.post_hoc()` method will no longer be able estimate simple-effects because it will not be able to group factor levels automatically.  *Note: this is also a useful approach if you don't want to use patsy's formula langauge with standard regression models as noted above*.

This is trivial using pandas map and assign methods. Here we'll only build a linear contrast across factor levels (0.5 < 1.0 < 1.5), without all exhaustive higher level polynomial terms:

.. code-block:: python

    df = df.assign(
    IV_3_custom_lin = df['IV3'].map({
                                    0.5: -1,
                                    1.0: 0,
                                    1.5: 1
                                    })
    df.head()

.. image:: ../misc/sample_data_custom_head.png

Now we can estimate this model without the need to use the :code:`factor` argument to the model's :code:`fit` method.

.. code-block:: python

    model = Lmer('DV ~ IV3_custom_lin + (IV3_custom_lin|Group)', data=df)
    model.fit()

How contrasts in R work
-----------------------
*This is just for folks curious about how contrasts in R work*.
Specifying multiple custom contrasts in R has always been a point of confusion amongst users. This because the :code:`contrasts()` command in R doesn't actually expect contrast weights (i.e. a design matrix) as one would intuit. Rather, it is made for generating contrast coding schemes which are the inverse of the contrast weight matrix. For a longer explanation with examples see `this reference <https://rstudio-pubs-static.s3.amazonaws.com/65059_586f394d8eb84f84b1baaf56ffb6b47f.html>`_ and `this reference <https://github.com/ejolly/R/blob/master/Guides/Contrasts_in_R.md>`_. For these situations pymer4 offers a few utility functions to convert between these matrix types if desired in :code:`pymer4.utils`: :code:`R2con` and :code:`con2R`.
