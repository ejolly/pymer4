import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_is_fitted, validate_data
import polars as pl
from formulae import model_description
from ..tidystats.lmerTest import fixef
from ..tidystats.multimodel import coef
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
        coef_ (ndarray): Model coefficients (fixed effects for mixed models)
        n_features_in_ (int): Number of features seen during fit
        feature_names_ (list): Names of features seen during fit
        model: Fitted pymer4 model instance

    Examples:
        >>> from pymer4.models import lm, skmer
        >>> from sklearn.model_selection import cross_val_score
        >>> X = [[1, 2], [3, 4], [5, 6], [7, 8]]
        >>> y = [1.5, 3.5, 5.5, 7.5]
        >>> model = skmer("y ~ x1 + x2", lm)
        >>> model.fit(X, y)
        >>> model.predict([[9, 10]])
    """

    def __init__(self, formula, model_class="lm", family=None, link=None):
        self.formula = formula
        supported_models = {"lm": lm_, "glm": glm_, "lmer": lmer_, "glmer": glmer_}
        self.model_class = supported_models[model_class]
        self.family = family
        self.link = link

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

        # Predictor variables are all variables minus group variables
        predictor_vars = sorted(list(all_vars - group_vars))

        # Create column names for X
        if X.shape[1] != len(predictor_vars):
            # If mismatch, create generic column names
            predictor_vars = [f"X{i}" for i in range(X.shape[1])]

        # Build the dataframe dictionary
        data_dict = {self.term_response_.term.name: y}
        for i, var_name in enumerate(predictor_vars):
            data_dict[var_name] = X[:, i]

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
        # TODO: Handle categorical variables by setting them to categorical types
        return pl.DataFrame(data_dict)

    def fit(self, X, y, group=None, **kwargs):
        """Fit the model to training data.

        Args:
            X (array-like): Feature matrix of shape (n_samples, n_features)
            y (array-like): Target values of shape (n_samples,)
            group (array-like, optional): Group labels for mixed-effects models

        Returns:
            self: Returns the instance itself
        """
        self.model_terms_ = model_description(self.formula)
        self.term_response_ = self.model_terms_.response
        self.term_ffx_ = self.model_terms_.common_terms
        self.term_rfx_ = self.model_terms_.group_terms

        # Set number of features based on formula
        # only for FFX
        self.feature_names_ = [
            term.name for term in self.term_ffx_ if term.name != "Intercept"
        ]
        self.n_features_in_ = len(self.feature_names_)

        # Validate inputs
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples")
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

    def predict(self, X, group=None, **kwargs):
        """Generate predictions from the fitted model.

        Args:
            X (array-like): Feature matrix of shape (n_samples, n_features)
            group (array-like, optional): Group labels for mixed-effects models
            **kwargs: Additional arguments passed to the model's predict method

        Returns:
            ndarray: Predicted values of shape (n_samples,)
        """
        check_is_fitted(self)

        # Validate inputs
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X must have {self.n_features_in_} columns, but has {X.shape[1]}"
            )

        # Create a dataframe with the same column names as used during fit
        data_dict = {}
        for i, var_name in enumerate(self.feature_names_):
            data_dict[var_name] = X[:, i]

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

        # Use the model's predict method
        # For mixed models, use use_rfx parameter to control random effects
        if isinstance(self.model_, (lmer_, glmer_)):
            # Default to using random effects if group is provided
            if "use_rfx" not in kwargs:
                kwargs["use_rfx"] = group is not None

        predictions = self.model_.predict(newdata, **kwargs)

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
            "model_class": self.model_class.__name__,
            "family": self.family,
            "link": self.link,
        }
        return params

    def set_params(self, **params):
        """Set the parameters of this estimator."""
        # Handle the main parameters
        for key in ["formula", "model_class", "family", "link"]:
            if key in params:
                setattr(self, key, params.pop(key))

        return self

    # def score(self, X, y, sample_weight=None, group=None):
    #     """Return the coefficient of determination R^2 of the prediction."""
    #     y = np.asarray(y)

    #     # Pass group if it's a mixed model
    #     if self.term_rfx_ is not None and group is not None:
    #         y_pred = self.predict(X, group=group)
    #     else:
    #         y_pred = self.predict(X)

    #     # TODO: if group is provided, calculate R-squared for each group
    #     # and return a dictionary of group names and R-squared values
    #     # Calculate R-squared
    #     ss_res = np.sum((y - y_pred) ** 2)
    #     ss_tot = np.sum((y - np.mean(y)) ** 2)

    #     # Avoid division by zero
    #     if ss_tot == 0:
    #         return 1.0 if ss_res == 0 else 0.0

    #     return 1 - (ss_res / ss_tot)
