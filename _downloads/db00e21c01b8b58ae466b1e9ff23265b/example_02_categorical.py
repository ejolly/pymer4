"""
2. Categorical Predictors
=========================
"""

###############################################################################
# The syntax for handling categorical predictors is **different** between standard regression models/two-stage-models (i.e. :code:`Lm` and :code:`Lm2`) and multi-level models (:code:`Lmer`) in :code:`pymer4`. This is because formula parsing is passed to R for :code:`Lmer` models, but handled by Python for other models.

###############################################################################
# Lm and Lm2 Models
# -----------------
# :code:`Lm` and :code:`Lm2` models use `patsy  <https://patsy.readthedocs.io/en/latest/>`_ to parse model formulae. Patsy is very powerful and has built-in support for handling categorical coding schemes by wrapping a predictor in then :code:`C()` *within* the module formula. Patsy can also perform some pre-processing such as scaling and standardization using special functions like :code:`center()`. Here are some examples.

# import basic libraries and sample data
import os
import pandas as pd
from pymer4.utils import get_resource_path
from pymer4.models import Lm

# IV3 is a categorical predictors with 3 levels in the sample data
df = pd.read_csv(os.path.join(get_resource_path(), "sample_data.csv"))

###############################################################################
# Dummy-coded/Treatment contrasts
# +++++++++++++++++++++++++++++++

# Estimate a model using Treatment contrasts (dummy-coding)
# with '1.0' as the reference level
# This is the default of the C() function
model = Lm("DV ~ C(IV3, levels=[1.0, 0.5, 1.5])", data=df)
print(model.fit())

###############################################################################
# Orthogonal Polynomial Contrasts
# +++++++++++++++++++++++++++++++

# Patsy can do this using the Poly argument to the
# C() function
model = Lm("DV ~ C(IV3, Poly)", data=df)
print(model.fit())

###############################################################################
# Sum-to-zero contrasts
# +++++++++++++++++++++

# Similar to before but with the Sum argument
model = Lm("DV ~ C(IV3, Sum)", data=df)
print(model.fit())

###############################################################################
# Scaling/Centering
# +++++++++++++++++

# Moderation with IV2, but centering IV2 first
model = Lm("DV ~ center(IV2) * C(IV3, Sum)", data=df)
print(model.fit())

###############################################################################
# Please refer to the `patsy documentation <https://patsy.readthedocs.io/en/latest/categorical-coding.html>`_ for more details when working categorical predictors in :code:`Lm` or :code:`Lm2` models.

###############################################################################
# Lmer Models
# -----------
# :code:`Lmer()` models currently have support for handling categorical predictors in one of three ways based on how R's :code:`factor()` works (see the note at the end of this tutorial):
#
# - Dummy-coded factor levels (treatment contrasts) in which each model term is the difference between a factor level and a selected reference level
# - Orthogonal polynomial contrasts in which each model term is a polynomial contrast across factor levels (e.g. linear, quadratic, cubic, etc)
# - Custom contrasts for each level of a factor, which should be provided in the manner expected by R.
#
# To make re-parameterizing models easier, factor codings are passed as a dictionary to the :code:`factors` argument of a model's :code:`.fit()`. This obviates the need for adjusting data-frame properties as in R. Note that this is **different** from :code:`Lm` and :code:`Lm2` models above which expect factor codings in their formulae (because patsy does).
#
# Each of these ways also enables you to easily compute post-hoc comparisons between factor levels, as well as interactions between continuous predictors and each factor level. See tutorial 3 for more on post-hoc tests.

from pymer4.models import Lmer

# We're going to fit a multi-level logistic regression using the
# dichotomous DV_l variable and the same categorical predictor (IV3)
# as before
model = Lmer("DV_l ~ IV3 + (IV3|Group)", data=df, family="binomial")

###############################################################################
# Dummy-coding factors
# ++++++++++++++++++++
# First we'll use dummy-coding/treatment contrasts with 1.0 as the reference level. This will compute two coefficients: 0.5 > 1.0 and 1.5 > 1.0.

print(model.fit(factors={"IV3": ["1.0", "0.5", "1.5"]}))


###############################################################################
# Polynomial contrast coding
# ++++++++++++++++++++++++++
# Second we'll use orthogonal polynomial contrasts. This is accomplished using the :code:`ordered=True` argument and specifying the order of the *linear* contrast in increasing order. R will automatically compute higher order polynomial contrats that are orthogonal to this linear contrast. In this example, since there are 3 factor levels this will result in two polynomial terms: a linear contrast we specify below corresponding to 0.5 < 1.0 < 1.5 and an orthogonal quadratic contrast automatically determined by R, corresponding to 0.5 > 1 < 1.5

print(model.fit(factors={"IV3": ["0.5", "1.0", "1.5"]}, ordered=True))

###############################################################################
# Custom contrasts
# ++++++++++++++++
# :code:`Lmer` models can also take custom factor contrasts based on how they are expected by R (see the note at the end of this tutorial for how contrasts work in R). Remember that there can be at most k-1 model terms representing any k level factor without over-parameterizing a model. If you specify a custom contrast, R will generate set of orthogonal contrasts for the rest of your model terms.

# Compare level '1.0' to the mean of levels '0.5' and '1.5'
# and let R determine the second contrast orthogonal to it

print(model.fit(factors={"IV3": {"1.0": 1, "0.5": -0.5, "1.5": -0.5}}))

###############################################################################
# User-created contrasts (without R)
# ++++++++++++++++++++++++++++++++++
# Another option available to you is fitting a model with *only* your desired contrast(s) rather than a full set of k-1 contrasts. Contrary to how statistics is usually taught, you don't ever *have to* include a full set of k-1 contrasts for a k level factor! The upside to doing this is that you won't need to rely on R to compute anything for you (aside from the model fit), and you will have a model with exactly the number of terms as contrasts you desire, giving you complete control. The downside is that post-hoc tests will no longer be available (see tutorial 3 for more information on post-hoc tests), but it's unlikely you're doing post-hoc tests if you are computing a subset of specific contrasts anyway. This is also a useful approach if you don't want to use patsy's formula syntax with :code:`Lm` and :code:`Lm2` as noted above.
#
# This can be accomplished by creating new columns in your dataframe to test specific hypotheses and is trivial to do with pandas `map <https://pandas.pydata.org/pandas-docs/version/0.25/reference/api/pandas.Series.map.html/>`_ and `assign <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.assign.html/>`_ methods. For example, here we manually compute a linear contrast by creating a new column in our dataframe and treating it as a continuous variable.

# Create a new column in the dataframe with a custom (linear) contrast
df = df.assign(IV3_custom_lin=df["IV3"].map({0.5: -1, 1.0: 0, 1.5: 1}))
print(df.head())

###############################################################################
# Now we can use this variable as a continuous predictor without the need for the :code:`factors` argument. Notice how the z-stat and p-value of the estimate are the same as the linear polynomial contrast estimated above. The coefficients differ in scale only because R uses [~-0.707, ~0, ~0.707] for its polynomial contrasts rather than [-1, 0, 1] like we did.

# Estimate model
model = Lmer(
    "DV_l ~ IV3_custom_lin + (IV3_custom_lin|Group)", data=df, family="binomial"
)
print(model.fit())

###############################################################################
# A note on how contrasts in R work
# ---------------------------------
# .. note::
#   This is just for folks curious about how contrasts in R work
#
# Specifying multiple custom contrasts in R has always been a point of confusion amongst users. This because the :code:`contrasts()` command in R doesn't actually expect contrast weights (i.e. a design matrix) as one would intuit. Rather, it is made for generating contrast coding schemes which are the inverse of the contrast weight matrix. For a longer explanation with examples see `this reference <https://rstudio-pubs-static.s3.amazonaws.com/65059_586f394d8eb84f84b1baaf56ffb6b47f.html>`_ and `this reference <https://github.com/ejolly/R/blob/master/Guides/Contrasts_in_R.md>`_. For these situations pymer4 offers a few utility functions to convert between these matrix types if desired in :code:`pymer4.utils`: :code:`R2con()` and :code:`con2R()`.
