import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted
import polars as pl
from formulae import model_description
from ..tidystats.lmerTest import fixef
from ..tidystats.multimodel import coef
from ..tidystats.plutils import make_factors
from .lm import lm as lm_
from .glm import glm as glm_
from .lmer import lmer as lmer_
from .glmer import glmer as glmer_


class skmer(RegressorMixin, BaseEstimator):
    """Scikit-learn compatible wrapper for pymer4 models.

    This class provides a scikit-learn compatible interface to pymer4's
    statistical models, allowing them to be used in scikit-learn pipelines,
    cross-validation, and other workflows.

    Args:
        formula (str): R-style formula string (e.g., "y ~ x1 + x2")
        model_class: pymer4 model class (lm, glm, lmer, or glmer)
        family (str, optional): Distribution family for GLM models
        link (str, optional): Link function for GLM models
        weights (array-like, optional): Sample weights
        **kwargs: Additional arguments passed to the model constructor

    Attributes:
        coef_ (ndarray): Model coefficients
        coef_rfx_ (ndarray): Model coefficients (random effects for mixed models)
        model_: Fitted pymer4 model instance
        n_features_in_ (int): Number of features seen during fit
        feature_names_ (list): Names of features seen during fit
        model_terms_ (ModelTerms): Model terms object
        term_response_ (Term): Response term object
        term_ffx_ (list): Fixed effects terms
        term_rfx_ (list): Random effects terms

    """

    def __init__(self, formula, model_class="auto", family=None, link=None):
        self.formula = formula
        self.family = family
        self.link = link

        # Parse formula to detect model type
        model_terms = model_description(formula)
        has_random_effects = len(model_terms.group_terms) > 0

        supported_models = {"lm": lm_, "glm": glm_, "lmer": lmer_, "glmer": glmer_}

        # Auto-detect model class based on formula and family
        if model_class == "auto":
            if has_random_effects:
                model_class = "glmer" if family is not None else "lmer"
            else:
                model_class = "glm" if family is not None else "lm"

        self.model_class = supported_models[model_class]
        self._model_class_str = model_class

    def _make_dataframe(self, X, y, group=None):
        """Convert sklearn-style arrays to polars DataFrame for pymer4 models.

        Args:
            X (array-like): Feature matrix of shape (n_samples, n_features)
            y (array-like): Target values of shape (n_samples,)
            group (array-like, optional): Group labels for mixed-effects models

        Returns:
            DataFrame: Polars DataFrame with named columns matching the formula
        """
        # Convert to numpy arrays if needed
        X = np.asarray(X)
        y = np.asarray(y)

        # Extract variable names from formula
        # Get all unique variable names from fixed and random effects
        all_vars = set()
        for term in self.term_ffx_:
            # Skip intercept terms
            if hasattr(term, "components"):
                # Extract variable names from components
                for comp in term.components:
                    if hasattr(comp, "name"):
                        all_vars.add(comp.name)
            elif hasattr(term, "name") and term.name not in ["1", "Intercept"]:
                all_vars.add(term.name)

        for group_term in self.term_rfx_:
            # Handle the expression part (can be a single term or iterable)
            expr = group_term.expr
            if hasattr(expr, "__iter__") and not isinstance(expr, str):
                # Multiple terms
                for term in expr:
                    if hasattr(term, "components"):
                        # Extract variable names from components
                        for comp in term.components:
                            if hasattr(comp, "name"):
                                all_vars.add(comp.name)
                    elif hasattr(term, "name") and term.name not in ["1", "Intercept"]:
                        all_vars.add(term.name)
            else:
                # Single term
                if hasattr(expr, "components"):
                    # Extract variable names from components
                    for comp in expr.components:
                        if hasattr(comp, "name"):
                            all_vars.add(comp.name)
                elif hasattr(expr, "name") and expr.name not in ["1", "Intercept"]:
                    all_vars.add(expr.name)

            # Extract the group factor name
            if hasattr(group_term.factor, "components"):
                for comp in group_term.factor.components:
                    if hasattr(comp, "name"):
                        all_vars.add(comp.name)

        # Remove response variable and intercept
        all_vars.discard(self.term_response_.term.name)
        all_vars.discard("1")

        # Remove group variables from predictor list if groups will be passed separately
        group_vars = set()
        if self.term_rfx_:
            for group_term in self.term_rfx_:
                if hasattr(group_term.factor, "components"):
                    for comp in group_term.factor.components:
                        if hasattr(comp, "name"):
                            group_vars.add(comp.name)

        # Use the feature names from the formula (which preserves order)
        # If not set yet (shouldn't happen), fall back to sorted list
        if hasattr(self, "feature_names_"):
            predictor_vars = self.feature_names_
        else:
            # Predictor variables are all variables minus group variables
            predictor_vars = sorted(list(all_vars - group_vars))

        # Validate that we have the right number of columns
        if X.shape[1] != len(predictor_vars):
            raise ValueError(
                f"X has {X.shape[1]} columns but formula expects {len(predictor_vars)} features: {predictor_vars}"
            )

        # Build the dataframe dictionary
        data_dict = {self.term_response_.term.name: y}
        for i, var_name in enumerate(predictor_vars):
            col_data = X[:, i]
            # Convert to appropriate type if needed
            if col_data.dtype == np.object_:
                # Try to infer the actual type
                try:
                    # Check if all values are strings
                    if all(isinstance(x, str) or x is None for x in col_data):
                        data_dict[var_name] = col_data.astype(str)
                    else:
                        # Try numeric conversion
                        data_dict[var_name] = col_data.astype(float)
                except (ValueError, TypeError):
                    # Fall back to object
                    data_dict[var_name] = col_data
            else:
                data_dict[var_name] = col_data

        # Add group column if provided for mixed models
        if group is not None and self.term_rfx_:
            group = np.asarray(group)
            # Get the group factor name from the random effects
            group_factor = self.term_rfx_[0].factor
            if hasattr(group_factor, "components") and group_factor.components:
                group_name = group_factor.components[0].name
            elif hasattr(group_factor, "name"):
                group_name = group_factor.name
            else:
                group_name = "group"
            data_dict[group_name] = group

        # Create polars DataFrame
        df = pl.DataFrame(data_dict)

        # Detect and convert string columns to categorical (Enum) types
        # This is necessary for R to recognize them as factors
        string_cols = []
        for col_name in df.columns:
            # Check for string columns (but not the response variable)
            if (
                col_name != self.term_response_.term.name
                and df[col_name].dtype == pl.Utf8
            ):
                string_cols.append(col_name)

        if string_cols:
            df = make_factors(df, string_cols)
            # Store factor information for prediction
            # Make a copy to avoid any reference issues
            self._factor_cols = string_cols.copy()
            self._factor_levels = {}
            for col_name in self._factor_cols:
                # Get the enum categories for each factor column
                if col_name in df.columns and hasattr(df[col_name].dtype, "categories"):
                    categories = df[col_name].dtype.categories
                    # Convert Series to list if needed
                    if isinstance(categories, pl.Series):
                        categories = categories.to_list()
                    self._factor_levels[col_name] = categories
        else:
            self._factor_cols = []
            self._factor_levels = {}

        return df

    def fit(self, X, y, **kwargs):
        """Fit the model to training data.

        For mixed-effects models (lmer/glmer), the group variable should be passed
        as the last column of X.

        Args:
            X (array-like): Feature matrix of shape (n_samples, n_features)
                For mixed models, last column should contain group labels
            y (array-like): Target values of shape (n_samples,)

        Returns:
            self: Returns the instance itself
        """
        self.model_terms_ = model_description(self.formula)
        self.term_response_ = self.model_terms_.response
        self.term_ffx_ = self.model_terms_.common_terms
        self.term_rfx_ = self.model_terms_.group_terms

        # Extract feature names in the order they appear in the formula
        # This is important to match the order of columns in X
        feature_names = []
        for term in self.term_ffx_:
            if hasattr(term, "name") and term.name not in ["Intercept", "1"]:
                feature_names.append(term.name)
            elif hasattr(term, "components"):
                # For complex terms, add each component
                for comp in term.components:
                    if hasattr(comp, "name") and comp.name not in feature_names:
                        feature_names.append(comp.name)

        self.feature_names_ = feature_names
        self.n_features_in_ = len(self.feature_names_)

        # Validate inputs
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples")

        # Handle embedded groups for mixed models
        # If we have random effects, last column must be the group
        group = None
        if self.term_rfx_:
            if X.shape[1] != self.n_features_in_ + 1:
                raise ValueError(
                    f"For mixed-effects models, X must have {self.n_features_in_ + 1} columns "
                    f"({self.n_features_in_} features + 1 group column), but got {X.shape[1]} columns"
                )
            # Extract group from last column
            group = X[:, -1]
            X = X[:, : self.n_features_in_]
        else:
            # For non-mixed models, X should have exactly n_features_in_ columns
            if X.shape[1] != self.n_features_in_:
                raise ValueError(
                    f"X must have {self.n_features_in_} columns, but has {X.shape[1]}"
                )

        # Prep model init
        df = self._make_dataframe(X, y, group)
        model_kwargs = kwargs.copy()
        if self.model_class in (glm_, glmer_):
            if self.family is not None:
                model_kwargs["family"] = self.family
            if self.link is not None:
                model_kwargs["link"] = self.link

        # Create and init model
        self.model_ = self.model_class(self.formula, data=df, **model_kwargs)
        self.model_._initialize()

        # Store coefs
        if isinstance(self.model_, (lm_, glm_)):
            self.coef_ = coef(self.model_.r_model)
        elif isinstance(self.model_, (lmer_, glmer_)):
            self.coef_ = fixef(self.model_.r_model)
            self.coef_rfx_ = coef(self.model_.r_model)

        return self

    def predict(self, X):
        """Generate predictions from the fitted model.

        For mixed-effects models (lmer/glmer), the group variable should be passed
        as the last column of X. If it is ommitted, predictions will be made using
        only fixed-effects.

        Args:
            X (array-like): Feature matrix of shape (n_samples, n_features)
                For mixed models, last column should contain group labels

        Returns:
            ndarray: Predicted values of shape (n_samples,)
        """
        check_is_fitted(self)

        # Handle embedded groups for mixed models
        group = None
        if self.term_rfx_:
            # For mixed models, check if group column is provided
            if X.shape[1] == self.n_features_in_ + 1:
                # Extract group from last column
                group = X[:, -1]
                X = X[:, : self.n_features_in_]
            elif X.shape[1] == self.n_features_in_:
                pass
            else:
                raise ValueError(
                    f"X must have {self.n_features_in_} to make predictions using only fixed-effects or {self.n_features_in_ + 1} columns (with group column) to make predictions using random effects, but has {X.shape[1]}"
                )
        else:
            # For non-mixed models, X should have exactly n_features_in_ columns
            if X.shape[1] != self.n_features_in_:
                raise ValueError(
                    f"X must have {self.n_features_in_} columns, but has {X.shape[1]}"
                )

        # Create a dataframe with the same column names as used during fit
        data_dict = {}
        for i, var_name in enumerate(self.feature_names_):
            col_data = X[:, i]
            # Convert to appropriate type if needed (same logic as in fit)
            if col_data.dtype == np.object_:
                # Try to infer the actual type
                try:
                    # Check if all values are strings
                    if all(isinstance(x, str) or x is None for x in col_data):
                        data_dict[var_name] = col_data.astype(str)
                    else:
                        # Try numeric conversion
                        data_dict[var_name] = col_data.astype(float)
                except (ValueError, TypeError):
                    # Fall back to object
                    data_dict[var_name] = col_data
            else:
                data_dict[var_name] = col_data

        # Add group column if it was used during fit and provided for prediction
        if self.term_rfx_ is not None and group is not None:
            group = np.asarray(group)
            # Get the group factor name from the random effects
            group_factor = self.term_rfx_[0].factor
            if hasattr(group_factor, "components") and group_factor.components:
                group_name = group_factor.components[0].name
            elif hasattr(group_factor, "name"):
                group_name = group_factor.name
            else:
                group_name = "group"
            data_dict[group_name] = group

        # Create polars DataFrame for prediction
        newdata = pl.DataFrame(data_dict)

        # Apply factor conversions if needed
        if hasattr(self, "_factor_cols") and self._factor_cols:
            # Convert string columns to categorical with the same levels as training
            for col_name in self._factor_cols:
                if col_name in newdata.columns and col_name in self._factor_levels:
                    # Use the stored factor levels from training
                    levels = self._factor_levels[col_name]
                    # Only convert if the column is actually a string type
                    if newdata[col_name].dtype == pl.Utf8:
                        newdata = newdata.with_columns(
                            pl.col(col_name).cast(pl.Enum(levels))
                        )

        # Use the model's predict method
        # For mixed models, use use_rfx parameter to control random effects
        # based on shape of X
        kwargs = {}
        if isinstance(self.model_, (lmer_, glmer_)):
            # Set use_rfx based on parameter or whether group is provided
            kwargs["use_rfx"] = True if group is not None else False
        try:
            predictions = self.model_.predict(newdata, **kwargs)
        except Exception as e:
            # If prediction fails due to new levels, try without random effects
            if "new levels detected" in str(e) and isinstance(
                self.model_, (lmer_, glmer_)
            ):
                kwargs["use_rfx"] = False
                predictions = self.model_.predict(newdata, **kwargs)
            else:
                raise

        # Extract numpy array from the result
        if isinstance(predictions, pl.DataFrame):
            # Get the prediction column (usually named 'fit' or similar)
            pred_col = predictions.columns[0]
            return predictions[pred_col].to_numpy()
        else:
            return np.asarray(predictions)

    def get_params(self, **kwargs):
        """Get parameters for this estimator."""
        params = {
            "formula": self.formula,
            "model_class": self._model_class_str,
            "family": self.family,
            "link": self.link,
        }
        return params

    def set_params(self, **params):
        """Set the parameters of this estimator."""
        # Handle model_class separately since it needs conversion
        if "model_class" in params:
            model_class_str = params.pop("model_class")
            supported_models = {"lm": lm_, "glm": glm_, "lmer": lmer_, "glmer": glmer_}
            self.model_class = supported_models[model_class_str]
            self._model_class_str = model_class_str

        # Handle other parameters
        for key in ["formula", "family", "link"]:
            if key in params:
                setattr(self, key, params.pop(key))

        return self
