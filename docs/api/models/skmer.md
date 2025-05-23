# `pymer4.models.skmer`

## Scikit-learn compatible estimators

`skmer()` models adhere to the `scikit-learn` API making them compatible with all model validation, estimation, and prediction tools:

### Linear Models

```python
from pymer4 import load_dataset
from pymer4.models import skmer
from sklearn.metrics import r2_score

# Prepare data sklearn style
penguins = load_dataset('penguins')
penguins = penguins.drop_nulls()

# Features
X = penguins[["bill_length_mm", "bill_depth_mm", "body_mass_g"]].to_numpy()

# Labels
y = penguins["flipper_length_mm"].to_numpy()

# Split up
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Linear Regression initialized with a formula
ols = skmer("flipper_length_mm ~ bill_length_mm + bill_depth_mm + body_mass_g")

# Fit & predict
ols.fit(X_train, y_train)
preds = ols.predict(X_test)

# Evaluate
r2_score(y_test, preds)

```

### Multi-level models

```python
from pymer4 import load_dataset
from pymer4.models import skmer
from sklearn.metrics import r2_score
from sklearn.model_selection import LeaveOneGroupOut

# Prepare data sklearn style
penguins = load_dataset('penguins')
penguins = penguins.drop_nulls()

# We pass in the rfx column as the last column of X
X_with_group = penguins[["bill_length_mm", "species"]].to_numpy()
y = penguins["flipper_length_mm"].to_numpy()

# This is for the cross-validator to know how to split up the data
groups = pengins[['species']].to_numpy()

lmm = skmer("flipper_length_mm ~ bill_length_mm + (bill_length_mm | species)")

# Out-of-sample r2 per species
scores = cross_val_score(lmm, X_with_group, y, cv=LeaveOneGroupOut(), groups=group)
```


## API

```{eval-rst}

.. autoclass:: pymer4.models.skmer.skmer

```

```{eval-rst}

.. autofunction:: pymer4.models.skmer.skmer.fit
.. autofunction:: pymer4.models.skmer.skmer.predict

```
