"""
3. ANOVA tables and post-hoc comparisons
========================================
"""

################################################################################
# .. note::
#   ANOVAs and post-hoc tests are only available for :code:`Lmer` models estimated using the :code:`factors` argument of :code:`model.fit()` and rely on implementations in R
#
# In the previous tutorial where we looked at categorical predictors, behind the scenes :code:`pymer4` was using the :code:`factor` functionality in R. This means the output of :code:`model.fit()` looks a lot like :code:`summary()` in R applied to a model with categorical predictors. But what if we want to compute an F-test across *all levels* of our categorical predictor?
#
# :code:`pymer4` makes this easy to do, and makes it easy to ensure Type III sums of squares infereces are valid. It also makes it easy to follow up omnibus tests with post-hoc pairwise comparisons.

################################################################################
# ANOVA tables and orthogonal contrasts
# -------------------------------------
# Because ANOVA is just regression, :code:`pymer4` can estimate ANOVA tables with F-results using the :code:`.anova()` method on a fitted model. This will compute a Type-III SS table given the coding scheme provided when the model was initially fit. Based on the distribution of data across factor levels and the specific coding-scheme used, this may produce invalid Type-III SS computations. For this reason the :code:`.anova()` method has a :code:`force-orthogonal=True` argument that will reparameterize and refit the model using orthogonal polynomial contrasts prior to computing an ANOVA table.
#
# Here we first estimate a mode with dummy-coded categories and suppress the summary output of :code:`.fit()`. Then we use :code:`.anova()` to examine the F-test results.

# import basic libraries and sample data
import os
import pandas as pd
from pymer4.utils import get_resource_path
from pymer4.models import Lmer

# IV3 is a categorical predictors with 3 levels in the sample data
df = pd.read_csv(os.path.join(get_resource_path(), "sample_data.csv"))

# # We're going to fit a multi-level regression using the
# categorical predictor (IV3) which has 3 levels
model = Lmer("DV ~ IV3 + (1|Group)", data=df)

# Using dummy-coding; suppress summary output
model.fit(factors={"IV3": ["1.0", "0.5", "1.5"]}, summarize=False)

# Get ANOVA table
print(model.anova())

################################################################################
# Type III SS inferences will only be valid if data are fully balanced across levels or if contrasts between levels are orthogonally coded and sum to 0. Below we tell :code:`pymer4` to respecify our contrasts to ensure this before estimating the ANOVA. :code:`pymer4` also saves the last set of contrasts used priory to forcing orthogonality.
#
# Because the sample data is balanced across factor levels and there are not interaction terms, in this case orthogonal contrast coding doesn't change the results.

# Get ANOVA table, but this time force orthogonality
# for valid SS III inferences
# In this case the data are balanced so nothing changes
print(model.anova(force_orthogonal=True))

################################################################################

# Checkout current contrast scheme (for first contrast)
# Notice how it's simply a linear contrast across levels
print(model.factors)

################################################################################

# Checkout previous contrast scheme
# which was a treatment contrast with 1.0
# as the reference level
print(model.factors_prev_)

################################################################################
# Marginal estimates and post-hoc comparisons
# -------------------------------------------
# :code:`pymer4` leverages the :code:`emmeans` package in order to compute marginal estimates ("cell means" in ANOVA lingo) and pair-wise comparisons of models that contain categorical terms and/or interactions. This can be performed by using the :code:`.post_hoc()` method on fitted models. Let's see an example:
#
# First we'll quickly create a second categorical IV to demo with and estimate a 3x3 ANOVA to get main effects and the interaction.

# Fix the random number generator
# for reproducibility
import numpy as np

np.random.seed(10)

# Create a new categorical variable with 3 levels
df = df.assign(IV4=np.random.choice(["1", "2", "3"], size=df.shape[0]))

# Estimate model with orthogonal polynomial contrasts
model = Lmer("DV ~ IV4*IV3 + (1|Group)", data=df)
model.fit(
    factors={"IV4": ["1", "2", "3"], "IV3": ["1.0", "0.5", "1.5"]},
    ordered=True,
    summarize=False,
)
# Get ANOVA table
# We can ignore the note in the output because
# we manually specified polynomial contrasts
print(model.anova())

################################################################################
# Example 1
# ~~~~~~~~~
# Compare each level of IV3 to each other level of IV3, *within* each level of IV4. Use default Tukey HSD p-values.

# Compute post-hoc tests
marginal_estimates, comparisons = model.post_hoc(
    marginal_vars="IV3", grouping_vars="IV4"
)

# "Cell" means of the ANOVA
print(marginal_estimates)

################################################################################

# Pairwise comparisons
print(comparisons)

################################################################################
# Example 2
# ~~~~~~~~~
# Compare each unique IV3,IV4 "cell mean" to every other IV3,IV4 "cell mean" and used FDR correction for multiple comparisons:


# Compute post-hoc tests
marginal_estimates, comparisons = model.post_hoc(
    marginal_vars=["IV3", "IV4"], p_adjust="fdr"
)

# Pairwise comparisons
print(comparisons)

################################################################################
# Example 3
# ~~~~~~~~~
# For this example we'll estimate a more complicated ANOVA with 1 continuous IV and 2 categorical IVs with 3 levels each. This is the same model as before but with IV2 thrown into the mix. Now, pairwise comparisons reflect changes in the *slope* of the continuous IV (IV2) between levels of the categorical IVs (IV3 and IV4).
#
# First let's get the ANOVA table
model = Lmer("DV ~ IV2*IV3*IV4 + (1|Group)", data=df)
# Only need to polynomial contrasts for IV3 and IV4
# because IV2 is continuous
model.fit(
    factors={"IV4": ["1", "2", "3"], "IV3": ["1.0", "0.5", "1.5"]},
    ordered=True,
    summarize=False,
)

# Get ANOVA table
print(model.anova())

################################################################################
# Now we can compute the pairwise difference in slopes

# Compute post-hoc tests with bonferroni correction
marginal_estimates, comparisons = model.post_hoc(
    marginal_vars="IV2", grouping_vars=["IV3", "IV4"], p_adjust="bonf"
)

# Pairwise comparisons
print(comparisons)
