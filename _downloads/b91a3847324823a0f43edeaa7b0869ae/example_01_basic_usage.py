"""
1. Basic Usage Guide
====================
"""

###############################################################################
# :code:`pymer4` comes with sample data for testing purposes which we'll utilize for most of the tutorials.
# This sample data has:
#
# - Two kinds of dependent variables: *DV* (continuous), *DV_l* (dichotomous)
# - Three kinds of independent variables: *IV1* (continuous), *IV2* (continuous), *IV3* (categorical)
# - One grouping variable for multi-level modeling: *Group*.
#
# Let's check it out below:

# import some basic libraries
import os
import pandas as pd

# import utility function for sample data path
from pymer4.utils import get_resource_path

# Load and checkout sample data
df = pd.read_csv(os.path.join(get_resource_path(), "sample_data.csv"))
print(df.head())

###############################################################################
# Standard regression models
# ------------------------------------
# Fitting a standard regression model is accomplished using the :code:`Lm` model class in :code:`pymer4`. All we need to do is initialize a model with a formula, some data, and call its :code:`.fit()` method.
#
# By default the output of :code:`.fit()` has been formated to be a blend of :code:`summary()` in R and :code:`.summary()` from `statsmodels <http://www.statsmodels.org/dev/index.html/>`_. This includes metadata about the model, data, and overall fit as well as estimates and inference results of model terms.

# Import the linear regression model class
from pymer4.models import Lm

# Initialize model using 2 predictors and sample data
model = Lm("DV ~ IV1 + IV2", data=df)

# Fit it
print(model.fit())

###############################################################################
# All information about the model as well as data, residuals, estimated coefficients, etc are saved as attributes and can be accessed like this:

# Print model AIC
print(model.AIC)

###############################################################################

# Look at residuals (just the first 10)
print(model.residuals[:10])

###############################################################################
# A copy of the dataframe used to estimate the model with added columns for residuals and fits are are available at :code:`model.data`. Residuals and fits can also be directly accessed using :code:`model.residuals` and :code:`model.fits` respectively

# Look at model data
print(model.data.head())

###############################################################################
# This makes it easy to assess overall model fit visually, for example using seaborn

# import dataviz
import seaborn as sns

# plot model predicted values against true values
sns.regplot(x="fits", y="DV", data=model.data, fit_reg=True)

###############################################################################
# Robust and WLS estimation
# -------------------------
# :code:`Lm` models can also perform inference using robust-standard errors or perform weight-least-squares (experimental feature) for models with categorical predictors (equivalent to Welch's t-test).

# Refit previous model using robust standard errors
print(model.fit(robust="hc1"))

###############################################################################

# Since WLS is only supported with 2 groups for now, filter the data first
df_two_groups = df.query("IV3 in [0.5, 1.0]").reset_index(drop=True)

# Fit new a model using a categorical predictor with unequal variances (WLS)
model = Lm("DV ~ IV3", data=df_two_groups)
print(model.fit(weights="IV3"))

###############################################################################
# Multi-level models
# ----------------------------
# Fitting a multi-level model works similarly and actually just calls :code:`lmer` or :code:`glmer` in R behind the scenes. The corresponding output is also formatted to be very similar to output of :code:`summary()` in R.

# Import the lmm model class
from pymer4.models import Lmer

# Initialize model instance using 1 predictor with random intercepts and slopes
model = Lmer("DV ~ IV2 + (IV2|Group)", data=df)

# Fit it
print(model.fit())

###############################################################################
# Similar to :code:`Lm` models, :code:`Lmer` models save details in model attributes and have additional methods that can be called using the same syntax as described above.

# Get population level coefficients
print(model.coefs)

###############################################################################

# Get group level coefficients (just the first 5)
# Each row here is a unique intercept and slope
# which vary because we parameterized our rfx that way above
print(model.fixef.head(5))

###############################################################################

# Get group level deviates from population level coefficients (i.e. rfx)
print(model.ranef.head(5))

###############################################################################
# :code:`Lmer` models also have some basic plotting abilities that :code:`Lm` models do not

# Visualize coefficients with group/cluster fits overlaid ("forest plot")
model.plot_summary()

###############################################################################
# Plot coefficients for each group/cluster as separate regressions
model.plot("IV2", plot_ci=True, ylabel="predicted DV")

###############################################################################
# Because :code:`Lmer` models rely on R, they have also some extra arguments to the :code:`.fit()` method for controlling things like optimizer behavior, as well as additional methods such for post-hoc tests and ANOVAs. See tutorial 2 for information about this functionality.

###############################################################################
# Two-stage summary statistics models
# -----------------------------------
# Fitting :code:`Lm2` models are also very similar

# Import the lm2 model class
from pymer4.models import Lm2

# This time we use the 'group' argument when initializing the model
model = Lm2("DV ~ IV2", group="Group", data=df)

# Fit it
print(model.fit())

###############################################################################
# Like :code:`Lmer` models, :code:`Lm2` models also store group/cluster level estimates and have some basic plotting functionality

# Get group level coefficients, just the first 5
print(model.fixef.head(5))

###############################################################################

# Visualize coefficients with group/cluster fits overlaid ("forest plot")
model.plot_summary()

###############################################################################
# Model Persistence
# -----------------
# All pymer4 models can be saved and loaded from disk. Doing so will persist *all* model attributes and data i.e. anything accessible with the '.' syntax. Models are saved and loaded using the `HDF5 format <https://support.hdfgroup.org/HDF5/whatishdf5.html/>`_ using the `deepdish <https://deepdish.readthedocs.io/en/latest/>`_ python library. This ensures near universal accesibility on different machines and operating systems. Therefore all filenames must end with :code:`.h5` or :code:`.hdf5`. For :code:`Lmer` models, an additional file ending in :code:`.rds` will be saved in the same directory as the HDF5 file. This is the R model object readable in R using :code:`readRDS`.
#
# To persist models you can use the dedicated :code:`save_model` and :code:`load_model` functions from the :code:`pymer4.io` module

# Import functions
from pymer4.io import save_model, load_model

# Save the Lm2 model above
save_model(model, "mymodel.h5")
# Load it back up
model = load_model("mymodel.h5")
# Check that it looks the same
print(model)

###############################################################################
# Wrap Up
# -------
# This was a quick overview of the 3 major model classes in :code:`pymer4`. However, it's highly recommended to check out the API to see *all* the features and options that each model class has including things like permutation-based inference (:code:`Lm` and :code:`Lm2` models) and fine-grain control of optimizer and tolerance settings (:code:`Lmer` models).
