from ..tidystats.broom import tidy, glance, augment
from ..tidystats.stats import model_matrix, anova
from ..tidystats.multimodel import predict, simulate
from ..tidystats.emmeans_lib import emmeans, emtrends, joint_tests, ref_grid
from ..tidystats.easystats import report as report_, get_fixed_params, model_params
from ..tidystats.tables import anova_table
from ..tidystats.plutils import join_on_common_cols, make_factors, unmake_factors
from ..tidystats.bridge import con2R
from ..rfuncs import get_summary
from polars import DataFrame, col
from rpy2.rinterface_lib import callbacks
from rpy2.robjects.packages import importr
from ..expressions import center, scale, zscore, rank
import numpy as np
from functools import wraps

lib_stats = importr("stats")


def requires_fit(func):
    """Decorator for methods that require the model to be fitted.

    This decorator checks if the model has been fitted before executing the method.
    If not fitted, raises a ValueError.

    Example:
        @requires_fit
        def summary(self):
            # This will raise an error if the model is not fitted
            return self._summary_func(self)
    """

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if not self.fitted:
            raise ValueError(
                f"You must .fit() the model before calling .{func.__name__}()"
            )
        return func(self, *args, **kwargs)

    return wrapper


def enable_logging(func):
    """Handles the verbose=True argument for model methods that print to R the console."""

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        self.clear_logs()
        if kwargs.get("verbose", True):
            result = func(self, *args, **kwargs)
            self.show_logs()
            return result
        else:
            return func(self, *args, **kwargs)

    return wrapper


def requires_result(attr_name):
    """Decorator for methods that require a specific result attribute to exist.

    This decorator checks if the specified attribute exists before executing the method.
    If the attribute doesn't exist, raises a ValueError.

    Args:
        attr_name (str): Name of the attribute to check for

    Example:
        @requires_result('result_anova')
        def summary_anova(self):
            # This will raise an error if result_anova doesn't exist
            return anova_table(self)
    """

    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            if getattr(self, attr_name, None) is None:
                raise ValueError(
                    f"First use .{attr_name.split('_')[-1]}() before calling .{func.__name__}()"
                )
            return func(self, *args, **kwargs)

        return wrapper

    return decorator


class model(object):
    """Base model class

    Args:
        formula (str): R-style model formula
        data (DataFrame): polars DataFrame
    """

    def __init__(self, formula, data, family=None, link="default", **kwargs):
        self._r_func: callable = lambda _: None
        self._r_contrasts = None
        self._summary_func: callable = lambda _: None
        self.formula = formula.replace(" ", "")
        self.factors = None
        self.transformed = None
        self.data = data
        self.family = family.capitalize() if family == "gamma" else family
        self.link = link
        self.result_fit = None
        self.result_fit_stats = None
        self.result_anova = None
        self.result_bootci = None
        self.result_emmeans = None
        self.result_vif = None
        self.r_model = None
        self.design_matrix = None
        self.contrasts = None
        self.params = None
        self.data_augmented = None
        self.fit_stats = None
        self.fitted = False
        self.nboot = None
        self.conf_method = None
        self.ci_type = None
        self.conf_level = None
        self.r_console = []
        callbacks.consolewrite_print = self._r_console_handler
        callbacks.consolewrite_warnerror = self._r_console_handler
        self._configure_family_link()

    def _r_console_handler(self, msg):
        """Handle console messages from R.

        Args:
            msg (str): Message from R console
        """
        if msg:
            self.r_console.append(msg)

    def _r_warnerror_handler(self, msg):
        """Handle warning and error messages from R, printing them in addition to saving.

        Args:
            msg (str): Warning or error message from R
        """
        if msg and msg not in self.r_console:
            self.r_console.append(msg)
            print(msg)

    def _r_print_handler(self, msg):
        """Handle print messages from R, saving them without printing.

        Args:
            msg (str): Print message from R
        """
        if msg and msg not in self.r_console:
            self.r_console.append(msg)

    def __repr__(self):
        """Return a string representation of the model.

        Returns:
            str: String representation including class name, fitted status, and formula
        """
        if self.family is None:
            out = "{}(fitted={}, formula={})".format(
                self.__class__.__module__,
                self.fitted,
                self.formula,
            )
        else:
            out = "{}(fitted={}, formula={}, family={}, link={})".format(
                self.__class__.__module__,
                self.fitted,
                self.formula,
                self.family,
                self.link,
            )
        return out

    def _configure_family_link(self):
        """Configure the R family and link function objects for the model."""

        if self.family is not None:
            family = getattr(lib_stats, self.family)
            self._r_family_link = (
                family() if self.link == "default" else family(link=self.link)
            )

        self._convert_logit2odds = self.family == "binomial" and self.link in [
            "default",
            "logit",
        ]

    def _1_setup_R_model(self, **kwargs):
        """Set up the R model with optional contrasts, family, and link.

        Args:
            **kwargs: Additional keyword arguments to pass to the R model function
        """
        if self.contrasts:
            if self.family is None:
                self.r_model = self._r_func(
                    self.formula, self.data, contrasts=self._r_contrasts, **kwargs
                )
            else:
                self.r_model = self._r_func(
                    self.formula,
                    self.data,
                    contrasts=self._r_contrasts,
                    family=self._r_family_link,
                    **kwargs,
                )
        else:
            if self.family is None:
                self.r_model = self._r_func(self.formula, self.data, **kwargs)
            else:
                self.r_model = self._r_func(
                    self.formula, self.data, family=self._r_family_link, **kwargs
                )

    def _2_get_tidy_summary(self, **kwargs):
        """Get a summary of fixed effects from the fitted model using broom's ``tidy`` function.

        Creates a `.result_fit` attribute containing a polars DataFrame parameter estimates, uncertainty, and inference statistics
        """

        self.result_fit = (
            tidy(self.r_model, effects="fixed", conf_int=True, **kwargs)
            .drop("effect", strict=False)
            .select(
                "term",
                "estimate",
                "conf_low",
                "conf_high",
                "std_error",
                "statistic",
                "p_value",
            )
        )

    def _3_get_coefs(self):
        """Extract model coefficients (fixed effects)."""
        self.params = get_fixed_params(self.r_model)

    def _4_get_glance_fit_stats(self):
        """Get model fit statistics using broom's ``glance`` function."""
        self.result_fit_stats = glance(self.r_model)

    def _5_get_augment_fits_resids(self, **kwargs):
        """Add model predictions and residuals to the data using broom's ``augment`` function to the data."""
        self.data = join_on_common_cols(self.data, augment(self.r_model, **kwargs))

    def _6_get_design_matrix(self):
        """Get the model design matrix with unique rows."""
        self.design_matrix = model_matrix(self.r_model, unique=True)

    @enable_logging
    def fit(self, summary=False, **kwargs):
        """Fit the model to the data.

        This method performs the following steps:
        1. Sets up and fits the R model
        2. Gets tidy fixed effects summary
        3. Extracts fixed effects coefficients
        4. Gets model fit statistics
        5. Adds predictions and residuals to data
        6. Gets model design matrix

        Args:
            summary (bool): Whether to return a ``great_tables`` summary of the fitted model
            **kwargs: Additional keyword arguments passed to the R model fitting function

        Returns:
            GT, optional: ``great_tables`` summary of the fitted model if ``summary=True``
        """
        # 1) Fit model
        self._1_setup_R_model(**kwargs)

        # 2) Get tidy fixed effects summary table
        self._2_get_tidy_summary()

        # 3) Get fixed effects; coefs for lms; BLUPs for lmms
        self._3_get_coefs()

        # 4) Get fit stats from broom
        self._4_get_glance_fit_stats()

        # 5) Add predictions to data
        self._5_get_augment_fits_resids()

        # 6) Get model design matrix
        self._6_get_design_matrix()

        self.fitted = True
        if summary:
            return self.summary()

    def show_logs(self):
        """Show any captured messages and warnings from R.

        Prints all messages and warnings that have been captured from R during model fitting
        and analysis.
        """
        if self.r_console:
            messages = "\n".join(self.r_console)
            print(f"R messages: \n{messages}")

    def clear_logs(self):
        """Clear any captured messages and warnings from R.

        Resets the R console message buffer to empty.
        """
        self.r_console = []

    def set_factors(self, factors_and_levels: str | dict | list):
        """Turn 1 or more variables into factors or change the levels of existing factors. Provide either a list of column names or a dictionary where keys are column names and values are lists of levels in the requested order. Relies on the fact that ``rpy2`` will convert pandas categorical types to R factors: `src <https://rpy2.github.io/doc/v3.5.x/html/changes.html#id12>`_

        Any existing factors can be seen with ``.show_factors()``.

        Args:
            factors_and_levels (str | dict | list): factors and their levels
        """
        # Convert categorical dtypes
        self.data, self.factors = make_factors(
            self.data, factors_and_levels, return_factor_dict=True
        )

        # Explicity set default treatment contrasts
        contrasts = {factor: "contr.treatment" for factor in self.factors.keys()}
        self.set_contrasts(contrasts)

    def unset_factors(self):
        """Convert factors back to their original data types (e.g. strings, integers, or floats)"""
        if self.factors is not None:
            self.data = unmake_factors(self.data, self.factors)
            self.factors = None
            self.contrasts = None

    def show_factors(self):
        """Print any current factors and their levels. The order of factor levels determines what parameter estimates represent and what how post-hoc contrasts are specified."""

        if self.factors is None:
            print("No factors set")
        else:
            print(self.factors)

    def set_contrasts(self, contrasts: dict, normalize=False):
        """
        Change the `default contrast coding scheme used by R <https://www.rdocumentation.org/packages/stats/versions/3.6.2/topics/contrast>`_ for factors or specify a set of custom contrasts between factor levels. Unlike base R, custom contrasts should be provided in terms of a human-readable *contrast matrix* representing differences across factor levels. This is similar to the `make.contrasts <https://www.rdocumentation.org/packages/gmodels/versions/2.18.1.1/topics/make.contrasts>`_ function from the ``gmodels`` package. Custom contrast will be *automatically* converted to a *coding matrix* which is what R expects. This allows you specify fewer that *k-1* contrasts for a factor with *k* levels and we will solve for the remaining orthogonal contrasts just like R.

        Note: setting contrasts will not affect the results of ``anova()`` when used with the deafult ``auto_ss_3=True``

        Args:
            contrasts (dict): a dictionary where keys are variables that are factors and value is a string specifying the contrast type, e.g. ``"contr.treatment"``, ``"contr.poly"``, or ``"contr.sum"`` or numeric contrast codes to compare across factor levels
            normalize (bool): whether to normalize contrasts by dividing by their vector norm to put them in standard-deviation units similar to ``contr.poly``; only applies for custom contrasts
        """
        Rcons = {}
        for k, v in contrasts.items():
            if k not in self.factors.keys():
                raise ValueError(f"{k} is not a known factor")
            if isinstance(v, str):
                Rcons[k] = v
            elif isinstance(v, (list, np.ndarray)):
                con = np.array(v) / np.linalg.norm(v) if normalize else np.array(v)
                contrasts[k] = con
                Rcons[k] = con2R(con)
        self._r_contrasts = Rcons
        self.contrasts = contrasts

    def show_contrasts(self):
        """Show the contrasts that have been set"""
        if self.contrasts is None:
            print("No factors set")
        else:
            print(self.contrasts)

    # TODO: change params to be more like set_factors and set_contrasts and take
    # dict as input
    def set_transforms(self, cols, transform="center", group=None):
        """Scale numeric columns by centering and/or scaling

        Args:
            cols (str/list): column name(s) to scale
            center (bool): whether to center the data
            scale (bool): whether to scale the data
            group (str; optional): column name to group by before scaling
        """

        # TODO: move to a separate module
        # rank_func = lambda column, method="average", descending=False: col(column).rank(
        #     method, descending=descending
        # )
        # center_func = lambda column: col(column) - col(column).mean()
        # scale_func = lambda column: col(column) / col(column).std()
        # zscore_func = (
        #     lambda column: (col(column) - col(column).mean()) / col(column).std()
        # )
        supported_transforms = dict(
            center=center, scale=scale, zscore=zscore, rank=rank
        )
        if transform not in supported_transforms.keys():
            raise ValueError(f"transform must be one of {supported_transforms.keys()}")

        func = supported_transforms[transform]
        transform = transform if group is None else f"{transform}_by_{group}"

        cols = [cols] if isinstance(cols, str) else cols
        compound_expression = []
        transforms = dict()
        for column in cols:
            compound_expression.append(col(column).alias(f"{column}_orig"))
            if group is None:
                compound_expression.append(func(col(column)).alias(column))
            else:
                compound_expression.append(func(col(column)).over(group).alias(column))
            transforms[column] = transform

        self.data = self.data.with_columns(compound_expression)
        self.transformed = transforms

    def unset_transforms(self, cols=None):
        """Undo the effect of calling `.set_transforms()`

        Args:
            cols (str | list; optional): column name(s) to unscale; if None, all scaled columns will be unscaled
        """
        if self.transformed is not None:
            if cols is None:
                cols = self.transformed.keys()
            cols = [cols] if isinstance(cols, str) else cols

            originals = [f"{column}_orig" for column in cols]
            compound_expression = []
            for original, transformed in zip(originals, cols):
                compound_expression.append(col(original).alias(transformed))
            self.data = self.data.with_columns(compound_expression).drop(originals)
            self.transformed = None

    def show_transforms(self):
        """Show the columns that have been scaled"""
        if self.transformed is None:
            print("No transformed columns")
        else:
            print(self.transformed)

    @enable_logging
    def anova(self, auto_ss_3=True, summary=False, **fitkwargs):
        """Calculate a Type-III ANOVA table for the model using `joint_tests()` in R.

        Args:
            auto_ss_3 (bool): whether to automatically use balanced contrasts when calculating the result via `joint_tests()`. When False, will use the contrasts specified with `set_contrasts()` which defaults to `"contr.treatment"` and R's `anova()` function; Default is True.
        """
        if not self.fitted:
            self.fit(**fitkwargs)
        if auto_ss_3:
            self.result_anova = joint_tests(self.r_model)
        else:
            self.result_anova = anova(self.r_model)
        if summary:
            return self.summary_anova()

    @enable_logging
    @requires_fit
    def emmeans(
        self,
        marginal_var,
        by=None,
        p_adjust="sidak",
        type="response",
        normalize=False,
        **kwargs,
    ):
        """Compute marginal means/trends and optionally contrasts between those means/trends at different factor levels. ``marginal_var`` is the predictor whose levels will have means or trends. ``by`` is an optional factor predictor to calculate separate means or trends for. If ``contrasts`` is provided, they are computed with respect to the marginal means or trends calculated

        Args:
            marginal_var (str): name of predictor to compute means or contrasts for
            by (str/list): additional predictors to marginalize over
            contrasts (str | 'pairwise' | 'poly' | dict | None, optional): how to specify comparison within `marginal_var`. Defaults to None.
            normalize (bool): normalize numerical contrasts to generate orthogonal polynomial similar to R; preferable for contrasts across more that 2 factor levels; Default False
            type (str): compute marginal means and contrasts on the 'response' or 'link' scale; Default 'response' (e.g. probabilities for logistic regression)

        Returns:
            DataFrame: Table of marginal means or contrasts
        """

        contrasts = kwargs.get("contrasts", None)
        if contrasts and normalize:
            contrasts = {
                k: np.array(v) / np.linalg.norm(v) for k, v in contrasts.items()
            }
            kwargs["contrasts"] = contrasts

        infer = np.array([True, False]) if contrasts is None else np.array([True, True])
        marginal_var = [marginal_var] if isinstance(marginal_var, str) else marginal_var

        # Separate continuous and categorical predictors
        if self.factors:
            marginals_are_factors = sum(
                [var in self.factors.keys() for var in marginal_var]
            )
            all_marginal_factors = marginals_are_factors == len(marginal_var)
            all_marginal_trends = marginals_are_factors == 0
            if (not all_marginal_factors) and (not all_marginal_trends):
                raise TypeError(
                    "When using more than 1 marginal_var they must all be the same type, i.e. all factors or all continuous"
                )
        else:
            all_marginal_trends = True
        # Guard against the fact that emtrends works a little differently than emmeans.
        # It will always return slopes per factor level even when by is not specified
        # however specifying a contrast when by=None will produce an error as the
        # categorical variable will be missing in emmgrid. So to marginalize overall
        # factor levels we pass in the continuous variable to both var and specs
        if all_marginal_trends:
            self.result_emmeans = emtrends(
                self.r_model,
                var=marginal_var,
                specs=marginal_var if by is None and contrasts is None else by,
                infer=infer,
                adjust=p_adjust,
                type=type,
                **kwargs,
            )
            if by is None:
                self.result_emmeans = self.result_emmeans.drop(marginal_var)
        else:
            self.result_emmeans = emmeans(
                self.r_model,
                specs=marginal_var,
                by=by,
                infer=infer,
                adjust=p_adjust,
                type=type,
                **kwargs,
            )

        return self.result_emmeans

    @requires_fit
    def empredict(self, at: dict, invert_transforms=True, type="response", **kwargs):
        """Compute marginal predictions at arbitrary levels of predictors.

        This method allows computing model predictions at specific values of predictors while
        marginalizing over other variables in the model. It handles both raw and transformed
        predictors.

        Args:
            at (dict): Dictionary mapping predictor names to values at which to compute predictions. Use "data" as the value to use all observed values for that predictor.
            invert_transforms (bool, optional): Whether to invert any transformations (center/scale/zscore) that were applied to predictors. Defaults to True.

        Returns:
            emmGrid: An R emmGrid object containing the predicted values and their standard errors.
                    This can be further processed using emmeans functions.

        Examples:
            >>> model.empredict({'x': [1, 2, 3], 'group': 'data'})  # Predictions at x=1,2,3 using all group values
            >>> model.empredict({'x': [-1, 0, 1]}, invert_transforms=False)  # Use values on transformed scale
        """
        _at = dict()
        # TODO: support ommitting predictors and using all data as default
        for k, v in at.items():
            if isinstance(v, str) and v == "data":
                _at[k] = self.data[k].to_numpy()
            else:
                # TODO: refactor this into a new function based on supported transforms
                if invert_transforms and self.transformed:
                    transform = self.transformed.get(k, None)
                    if transform == "center":
                        v = np.array(v) + self.data[f"{k}_orig"].mean()
                    elif transform == "scale":
                        v = np.array(v) * self.data[f"{k}_orig"].std()
                    elif transform == "zscore":
                        v = (np.array(v) + self.data[f"{k}_orig"].mean()) * self.data[
                            f"{k}_orig"
                        ].std()
                _at[k] = v
        return ref_grid(
            self.r_model, at=_at, type=type, infer=np.array([True, False]), **kwargs
        )

    @requires_fit
    def predict(self, data: DataFrame, **kwargs):
        """Make predictions using new data

        Args:
            data (DataFrame): polars DataFrame

        Returns:
            predictions (ndarray): predicted values
        """
        return predict(self.r_model, data, **kwargs)

    @requires_fit
    def simulate(self, nsim: int = 1, **kwargs):
        """Simulate values from the model

        Args:
            nsim (int): number of simulations to run

        Returns:
            simulations (DataFrame): simulated values with the same number of rows as the original data and columns equal to `nsim`
        """
        return simulate(self.r_model, nsim, **kwargs)

    @requires_fit
    def summary(self, pretty=True, decimals=3):
        """Print a nicely formatted summary table that contains ``.result_fit``
        Uses the ``great_tables`` package, which can be `exported in a variety of formats <https://posit-dev.github.io/great-tables/reference/#export>`_

        Args:
            decimals (int): number of decimal places to round to; p-values are rounded to ``decimals + 1`` places
        """
        if pretty:
            return self._summary_func(self, decimals=decimals)
        print(get_summary(self.r_model))

    @requires_result("result_anova")
    def summary_anova(self, decimals=3):
        """Print a nicely formatted summary table that contains ``.result_anova``
        Uses the ``great_tables`` package, which can be `exported in a variety of formats <https://posit-dev.github.io/great-tables/reference/#export>`_


        Args:
            decimals (int): number of decimal places to round to; p-values are rounded to ``decimals + 1`` places
        """
        return anova_table(self, decimals=decimals)

    @requires_fit
    def vif(self):
        """Calculate the variance inflation factor (VIF) and confidence interval increase factor (CI) (square root of VIF) for each predictor in the model.

        Returns:
            DataFrame: A DataFrame containing the VIF and CI for each predictor.
        """
        corr_df = self.design_matrix.drop("(Intercept)", strict=False).corr()
        vifs = np.diag(np.linalg.inv(corr_df))
        ci_increase = np.sqrt(vifs)
        out = DataFrame(
            {
                "term": corr_df.columns,
                "vif": vifs,
                "ci_increase_factor": ci_increase,
            }
        )
        self.result_vif = out
        return out

    @requires_fit
    def report(self):
        """Generate a natural language report of the model results.

        Uses R's report package to generate a text description of the model,
        its parameters, and fit statistics.

        Returns:
            str: A natural language description of the model results
        """
        print(report_(self.r_model))

    @requires_fit
    def to_sklearn(self):
        """Convert the fitted model to a scikit-learn compatible format.

        Note: This method is not yet implemented.

        Raises:
            NotImplementedError: This method is not yet implemented
        """
        raise NotImplementedError("Not yet implemented")
