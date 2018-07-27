from __future__ import division
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
import rpy2
from copy import copy
import pandas as pd
import numpy as np
from scipy.stats import t as t_dist
import matplotlib.pyplot as plt
from patsy import dmatrices
import seaborn as sns
import warnings
from six import string_types
from joblib import Parallel, delayed
from pymer4.utils import (_sig_stars,
                          _chunk_boot_ols_coefs,
                          _chunk_perm_ols,
                          _ols,
                          _perm_find,
                          _return_t
                          )

__author__ = ['Eshin Jolly']
__license__ = "MIT"

warnings.simplefilter('always',UserWarning)
pandas2ri.activate()

class Lmer(object):

    """
    Model class to hold data outputted from fitting lmer in R and converting to Python object. This class stores as much information as it can about a merMod object computed using lmer and lmerTest in R. Most attributes will not be computed until the fit method is called.

    Args:
        formula (str): Complete lmer-style model formula
        data (pandas.core.frame.DataFrame): input data
        family (string): what distribution family (i.e.) link function to use for the generalized model; default is gaussian (linear model)

    Attributes:
        fitted (bool): whether model has been fit
        formula (str): model formula
        data (pandas.core.frame.DataFrame): model copy of input data
        grps (dict): groups and number of observations per groups recognized by lmer
        AIC (float): model akaike information criterion
        logLike (float): model Log-likelihood
        family (string): model family
        ranef (pandas.core.frame.DataFrame/list): cluster-level differences from population parameters, i.e. difference between coefs and fixefs; returns list if multiple cluster variables are used to specify random effects (e.g. subjects and items)
        fixef (pandas.core.frame.DataFrame/list): cluster-level parameters; returns list if multiple cluster variables are used to specify random effects (e.g. subjects and items)
        coefs (pandas.core.frame.DataFrame/list): model summary table of population parameters
        resid (numpy.ndarray): model residuals
        fits (numpy.ndarray): model fits/predictions
        model_obj(lmer model): rpy2 lmer model object
        factors (dict): factors used to fit the model if any

    """

    def __init__(self,formula,data,family='gaussian'):

        self.family = family
        implemented_fams = ['gaussian','binomial','gamma','inverse_gaussian','poisson']
        if self.family not in implemented_fams:
            raise ValueError("Family must be one of: gaussian, binomial, gamma, inverse_gaussian or poisson!")
        self.fitted = False
        self.formula = formula
        self.data = copy(data)
        self.grps = None
        self.AIC = None
        self.logLike = None
        self.warnings = None
        self.ranef_var = None
        self.ranef_corr = None
        self.ranef = None
        self.fixef = None
        self.design_matrix = None
        self.resid = None
        self.coefs = None
        self.model_obj = None
        self.factors = None
        self.marginal_estimates = None
        self.marginal_contrasts = None
        self.sig_type = None
        self.factors_prev_ = None

    def __repr__(self):
        out = "{}.{}(fitted = {}, formula = {}, family = {})".format(
        self.__class__.__module__,
        self.__class__.__name__,
        self.fitted,
        self.formula,
        self.family)
        return out

    def _make_factors(self,factor_dict, ordered=False):
        """
        Covert specific columns to R-style factors. Default scheme is dummy coding where reference is 1st level provided. Alternative is orthogonal polynomial contrasts. User can also specific custom contrasts.

        Args:
            factor_dict: (dict) dictionary with column names specified as keys, and lists of unique values to treat as factor levels
            ordered: (bool) whether to interpret factor_dict values as dummy-coded (1st list item is reference level) or as polynomial contrasts (linear contrast specified by ordered of list items)

        Returns:
            pandas.core.frame.DataFrame: copy of original data with factorized columns
        """
        if ordered:
            rstring = """
                function(df,f,lv){
                df[,f] <- factor(df[,f],lv,ordered=T)
                df
                }
            """
        else:
            rstring = """
                function(df,f,lv){
                df[,f] <- factor(df[,f],lv,ordered=F)
                df
                }
            """

        c_rstring = """
            function(df,f,c){
            contrasts(df[,f]) <- c(c)
            df
            }
            """

        factorize = robjects.r(rstring)
        contrastize = robjects.r(c_rstring)
        df = copy(self.data)
        for k in factor_dict.keys():
            df[k] = df[k].astype(str)

        r_df = pandas2ri.py2ri(df)
        for k,v in factor_dict.items():
            if isinstance(v, list):
                r_df = factorize(r_df,k,v)
            elif isinstance(v, dict):
                levels = list(v.keys())
                contrasts = np.array(list(v.values()))
                r_df = factorize(r_df,k,levels)
                r_df = contrastize(r_df,k,contrasts)
        return r_df

    def _refit_orthogonal(self):
        """
        Refit a model with factors organized as polynomial contrasts to ensure valid type-3 SS calculations with using `.anova()`. Previous factor specifications are stored in `model.factors_prev_`.
        """

        self.factors_prev_ = copy(self.factors)
        # Create orthogonal polynomial contrasts by just sorted factor levels alphabetically and letting R enumerate the required polynomial contrasts
        new_factors = {}
        for k in self.factors.keys():
            new_factors[k] = sorted(list(map(str, self.data[k].unique())))
        self.fit(factors = new_factors,ordered=True,summarize=False,permute=self._permute,conf_int=self._conf_int,REML=self._REML)

    def anova(self,force_orthogonal=False):
        """
        Return a type-3 ANOVA table from a fitted model. Like R, this method does not ensure that contrasts are orthogonal to ensure correct type-3 SS computation. However, the force_orthogonal flag can refit the regression model with orthogonal polynomial contrasts automatically guaranteeing valid SS type 3 inferences. Note that this will overwrite factors specified in the last call to `.fit()`

        Args:
            force_orthogonal (bool): whether factors in the model should be recoded using polynomial contrasts to ensure valid type-3 SS calculations. If set to True, previous factor specifications will be saved in `model.factors_prev_`; default False

        Returns:
            anova_results (pd.DataFrame): Type 3 ANOVA results
        """

        if self.factors:
            # Model can only have factors if it's been fit
            if force_orthogonal:
                self._refit_orthogonal()
        elif not self.fitted:
            raise ValueError("Model must be fit before ANOVA table can be generated!")

        rstring = """
            function(model){
            df<- anova(model)
            df
            }
        """
        anova = robjects.r(rstring)
        self.anova_results = pandas2ri.ri2py(anova(self.model_obj))
        if self.anova_results.shape[1] == 6:
                self.anova_results.columns = ['SS','MS','NumDF','DenomDF','F-stat','P-val']
                self.anova_results['Sig'] = self.anova_results['P-val'].apply(lambda x: _sig_stars(x))
        elif self.anova_results.shape[1] == 4:
            warnings.warn("MODELING FIT WARNING! Check model.warnings!! P-value computation did not occur because lmerTest choked. Possible issue(s): ranefx have too many parameters or too little variance...")
            self.anova_results.columns = ['DF','SS','MS','F-stat']
        if force_orthogonal:
            print("SS Type III Analysis of Variance Table with Satterthwaite approximated degrees of freedom:\n(NOTE: Model refit with orthogonal polynomial contrasts)")
        else:
            print("SS Type III Analysis of Variance Table with Satterthwaite approximated degrees of freedom:\n(NOTE: Using original model contrasts, orthogonality not guaranteed)")
        return self.anova_results

    def fit(self,conf_int='Wald',factors=None,permute=None,ordered=False,summarize=True,verbose=False,REML=True):
        """
        Main method for fitting model object. Will modify the model's data attribute to add columns for residuals and fits for convenience.

        Args:
            conf_int (str): which method to compute confidence intervals; 'profile', 'Wald' (default), or 'boot' (parametric bootstrap)
            factors (dict): Keys should be column names in data to treat as factors. Values should either be a list containing unique variable levels if dummy-coding or polynomial coding is desired. Otherwise values should be another dictionary with unique variable levels as keys and desired contrast values (as specified in R!) as keys. See examples below
            permute (int): if non-zero, computes parameter significance tests by permuting test stastics rather than parametrically. Permutation is done by shuffling observations within clusters to respect random effects structure of data.
            ordered (bool): whether factors should be treated as ordered polynomial contrasts; this will parameterize a model with K-1 orthogonal polynomial regressors beginning with a linear contrast based on the factor order provided; default is False
            summarize (bool): whether to print a model summary after fitting; default is True
            verbose (bool): whether to print when and which model and confidence interval are being fitted
            REML (bool): whether to fit using restricted maximum likelihood estimation instead of maximum likelihood estimation; default True

        Returns:
            DataFrame: R style summary() table

        Examples:
            The following examples demonstrate how to treat variables as categorical factors.

            Dummy-Coding: Treat Col1 as a factor which 3 levels: A, B, C. Use dummy-coding with A as the reference level. Model intercept will be mean of A, and parameters will be B-A, and C-A.

            >>> model.fit(factors = {"Col1": ['A','B','C']})

            Orthogonal Polynomials: Treat Col1 as a factor which 3 levels: A, B, C. Estimate a linear contrast of C > B > A. Model intercept will be grand-mean of all levels, and parameters will be linear contrast, and orthogonal polynomial contrast (auto-computed).

            >>> model.fit(factors = {"Col1": ['A','B','C']}, ordered=True)

            Custom-contrast: Treat Col1 as a factor which 3 levels: A, B, C. Compare A to the mean of B and C. Model intercept will be the grand-mean of all levels, and parameters will be the desired contrast, a well as an automatically determined orthogonal contrast.

            >>> model.fit(factors = {"Col1": {'A': 1, 'B': -.5, 'C': -.5}}))

        """

        # Save params for future calls
        self._permute = permute
        self._conf_int = conf_int
        self._REML = REML
        if factors:
            dat = self._make_factors(factors,ordered)
            self.factors = factors
        else:
            dat = self.data
        self.sig_type = 'parametric' if not permute else 'permutation' + ' (' + str(permute) + ')'
        if self.family == 'gaussian':
            _fam = 'gaussian'
            if verbose:
                print("Fitting linear model using lmer with "+conf_int+" confidence intervals...\n")

            lmer = importr('lmerTest')
            self.model_obj = lmer.lmer(self.formula,data=dat,REML=REML)
        else:
            if verbose:
                print("Fitting generalized linear model using glmer (family {}) with "+conf_int+" confidence intervals...\n".format(self.family))
            lmer = importr('lme4')
            if self.family == 'inverse_gaussian':
                _fam = 'inverse.gaussian'
            elif self.family == 'gamma':
                _fam = 'Gamma'
            else:
                _fam = self.family
            self.model_obj = lmer.glmer(self.formula,data=dat,family=_fam,REML=REML)

        if permute and verbose:
            print("Using {} permutations to determine significance...".format(permute))
        base = importr('base')

        summary = base.summary(self.model_obj)
        unsum = base.unclass(summary)

        #Do scalars first cause they're easier
        grps = pandas2ri.ri2py(base.data_frame(unsum.rx2('ngrps')))
        self.grps = dict(grps.T.iloc[0])
        self.AIC = unsum.rx2('AICtab')[0]
        self.logLike = unsum.rx2('logLik')[0]

        #First check for lme4 printed messages (e.g. convergence info is usually here instead of in warnings)
        fit_messages = unsum.rx2('optinfo').rx2('conv').rx2('lme4').rx2('messages')
        #Then check warnings for additional stuff
        fit_warnings = unsum.rx2('optinfo').rx2('warnings')
        if len(fit_warnings) != 0:
            fit_warnings = [str(elem) for elem in fit_warnings]
        else: fit_warnings = []
        if not isinstance(fit_messages,rpy2.rinterface.RNULLType):
            fit_messages = [str(elem) for elem in fit_messages]
        else:
            fit_messages = []

        fit_messages_warnings = fit_messages + fit_warnings
        if fit_messages_warnings:
            self.warnings = fit_messages_warnings
            for warning in self.warnings:
                print(warning + ' \n')
        else:
            self.warnings = None

        #Coefficients, and inference statistics
        if self.family in ['gaussian','gamma','inverse_gaussian','poisson']:

            rstring = """
                function(model){
                out.coef <- data.frame(unclass(summary(model))$coefficients)
                out.ci <- data.frame(confint(model,method='"""+conf_int+"""'))
                n <- c(rownames(out.ci))
                idx <- max(grep('sig',n))
                out.ci <- out.ci[-seq(1:idx),]
                out <- cbind(out.coef,out.ci)
                out
                }
            """
            estimates_func = robjects.r(rstring)
            df = pandas2ri.ri2py(estimates_func(self.model_obj))

            # gaussian
            if df.shape[1] == 7:
                df.columns = ['Estimate','SE','DF','T-stat','P-val','2.5_ci','97.5_ci']
                df = df[['Estimate','2.5_ci','97.5_ci','SE','DF','T-stat','P-val']]

            # gamma, inverse_gaussian
            elif df.shape[1] == 6:
                if self.family in ['gamma','inverse_gaussian']:
                    df.columns = ['Estimate','SE','T-stat','P-val','2.5_ci','97.5_ci']
                    df = df[['Estimate','2.5_ci','97.5_ci','SE','T-stat','P-val']]
                else:
                    df.columns = ['Estimate','SE','Z-stat','P-val','2.5_ci','97.5_ci']
                    df = df[['Estimate','2.5_ci','97.5_ci','SE','Z-stat','P-val']]

            # Incase lmerTest chokes it won't return p-values
            elif df.shape[1] == 5 and self.family == 'gaussian':
                if not permute:
                    warnings.warn("MODELING FIT WARNING! Check model.warnings!! P-value computation did not occur because lmerTest choked. Possible issue(s): ranefx have too many parameters or too little variance...")
                    df.columns =['Estimate','SE','T-stat','2.5_ci','97.5_ci']
                    df = df[['Estimate','2.5_ci','97.5_ci','SE','T-stat']]

        elif self.family == 'binomial':

            rstring = """
                function(model){
                out.coef <- data.frame(unclass(summary(model))$coefficients)
                out.ci <- data.frame(confint(model,method='"""+conf_int+"""'))
                n <- c(rownames(out.ci))
                idx <- max(grep('sig',n))
                out.ci <- out.ci[-seq(1:idx),]
                out <- cbind(out.coef,out.ci)
                odds <- exp(out.coef[1])
                colnames(odds) <- "OR"
                probs <- data.frame(sapply(out.coef[1],plogis))
                colnames(probs) <- "Prob"
                odds.ci <- exp(out.ci)
                colnames(odds.ci) <- c("OR_2.5_ci","OR_97.5_ci")
                probs.ci <- data.frame(sapply(out.ci,plogis))
                colnames(probs.ci) <- c("Prob_2.5_ci","Prob_97.5_ci")
                out <- cbind(out,odds,odds.ci,probs,probs.ci)
                out
                }
            """

            estimates_func = robjects.r(rstring)
            df = pandas2ri.ri2py(estimates_func(self.model_obj))

            df.columns = ['Estimate','SE','Z-stat','P-val','2.5_ci','97.5_ci','OR','OR_2.5_ci','OR_97.5_ci','Prob','Prob_2.5_ci','Prob_97.5_ci']
            df = df[['Estimate','2.5_ci','97.5_ci','SE','OR','OR_2.5_ci','OR_97.5_ci','Prob','Prob_2.5_ci','Prob_97.5_ci','Z-stat','P-val']]

        if permute:
            perm_dat = dat.copy()
            dv_var = self.formula.split('~')[0].strip()
            grp_vars = list(self.grps.keys())
            perms = []
            for i in range(permute):
                perm_dat[dv_var] = perm_dat.groupby(grp_vars)[dv_var].transform(lambda x: x.sample(frac=1))
                if self.family == 'gaussian':
                    perm_obj = lmer.lmer(self.formula,data=perm_dat,REML=REML)
                else:
                    perm_obj = lmer.glmer(self.formula,data=perm_dat,family=_fam,REML=REML)
                perms.append(_return_t(perm_obj))
            perms = np.array(perms)
            pvals = []
            for c in range(df.shape[0]):
                if self.family in  ['gaussian','gamma','inverse_gaussian']:
                    pvals.append(_perm_find(perms[:,c], df['T-stat'][c]))
                else:
                    pvals.append(_perm_find(perms[:,c], df['Z-stat'][c]))
            df['P-val'] = pvals
            df['DF'] = [permute] * df.shape[0]
            df = df.rename(columns={'DF':'Num_perm','P-val':'Perm-P-val'})

        if 'P-val' in df.columns:
            df['Sig'] = df['P-val'].apply(lambda x: _sig_stars(x))
        elif 'Perm-P-val' in df.columns:
            df['Sig'] = df['Perm-P-val'].apply(lambda x: _sig_stars(x))
        if permute:
            # Because all models except lmm have no DF column make sure Num_perm gets put in the right place
            cols = list(df.columns)
            col_order = cols[:-4] + ['Num_perm'] + cols[-4:-2] +[cols[-1]]
            df = df[col_order]
        self.coefs = df
        self.fitted = True

        #Random effect variances and correlations
        df = pandas2ri.ri2py(base.data_frame(unsum.rx2('varcor')))
        ran_vars = df.query("(var2 == 'NA') | (var2 == 'N')").drop('var2',axis=1)
        ran_vars.index = ran_vars['grp']
        ran_vars.drop('grp',axis=1,inplace=True)
        ran_vars.columns = ['Name','Var','Std']
        ran_vars.index.name = None
        ran_vars.replace('NA','',inplace=True)

        ran_corrs = df.query("(var2 != 'NA') & (var2 != 'N')").drop('vcov',axis=1)
        if ran_corrs.shape[0] != 0:
            ran_corrs.index = ran_corrs['grp']
            ran_corrs.drop('grp',axis=1,inplace=True)
            ran_corrs.columns = ['IV1','IV2','Corr']
            ran_corrs.index.name = None
        else:
            ran_corrs = None

        self.ranef_var = ran_vars
        self.ranef_corr = ran_corrs

        #Cluster (e.g subject) level coefficients
        rstring = """
            function(model){
            out <- coef(model)
            out
            }
        """
        fixef_func = robjects.r(rstring)
        fixefs = fixef_func(self.model_obj)
        if len(fixefs) > 1:
            f_corrected_order = []
            for f in fixefs:
                f = pandas2ri.ri2py(f)
                f_corrected_order.append(f[list(self.coefs.index)+[elem for elem in f.columns if elem not in self.coefs.index]])
            self.fixef = f_corrected_order
            #self.fixef = [pandas2ri.ri2py(f) for f in fixefs]
        else:
            self.fixef = pandas2ri.ri2py(fixefs[0])
            self.fixef = self.fixef[list(self.coefs.index)+[elem for elem in self.fixef.columns if elem not in self.coefs.index]]

        #Sort column order to match population coefs
        #This also handles cases in which random slope terms exist in the model without corresponding fixed effects terms, which generates extra columns in this dataframe. By default put those columns *after* the fixed effect columns of interest (i.e. population coefs)


        #Cluster (e.g subject) level random deviations
        rstring = """
            function(model){
            uniquify <- function(df){
            colnames(df) <- make.unique(colnames(df))
            df
            }
            out <- lapply(ranef(model),uniquify)
            out
            }
        """
        ranef_func = robjects.r(rstring)
        ranefs = ranef_func(self.model_obj)
        if len(ranefs) > 1:
            self.ranef = [pandas2ri.ri2py(r) for r in ranefs]
        else:
            self.ranef = pandas2ri.ri2py(ranefs[0])

        #Save the design matrix
        #Make sure column names match population coefficients
        stats = importr('stats')
        self.design_matrix = pandas2ri.ri2py(stats.model_matrix(self.model_obj))
        self.design_matrix = pd.DataFrame(self.design_matrix, columns = self.coefs.index[:])

        #Model residuals
        rstring = """
            function(model){
            out <- resid(model)
            out
            }
        """
        resid_func = robjects.r(rstring)
        self.resid = pandas2ri.ri2py(resid_func(self.model_obj))
        self.data['residuals'] = copy(self.resid)

        #Model fits
        rstring = """
            function(model){
            out <- fitted(model)
            out
            }
        """
        fit_func = robjects.r(rstring)
        self.fits = pandas2ri.ri2py(fit_func(self.model_obj))
        self.data['fits'] = copy(self.fits)

        if summarize:
            return self.summary()


    def simulate(self,num_datasets,use_rfx=True):
        """
        Simulate new responses based upon estimates from a fitted model. By default group/cluster means for simulated data will match those of the original data. Unlike predict, this is a non-deterministic operation because lmer will sample random-efects values for all groups/cluster and then sample data points from their respective conditional distributions.

        Args:
            num_datasets (int): number of simulated datasets to generate. Each simulation always generates a dataset that matches the size of the original data
            use_rfx (bool): match group/cluster means in simulated data?; Default True

        Returns:
            ndarray: simulated data values
        """

        if isinstance(num_datasets,float):
            num_datasets = int(num_datasets)
        if not isinstance(num_datasets, int):
            raise ValueError("num_datasets must be an integer")

        if use_rfx:
            re_form = 'NULL'
        else:
            re_form = 'NA'

        rstring = """
            function(model){
            out <- simulate(model,""" + str(num_datasets) + """,allow.new.levels=TRUE,re.form="""+re_form+""")
            out
            }
        """
        simulate_func = robjects.r(rstring)
        sims = pandas2ri.ri2py(simulate_func(self.model_obj))
        return sims

    def predict(self,data,use_rfx=False,pred_type='response'):
        """
        Make predictions given new data. Input must be a dataframe that contains the same columns as the model.matrix excluding the intercept (i.e. all the predictor variables used to fit the model). If using random effects to make predictions, input data must also contain a column for the group identifier that were used to fit the model random effects terms. Using random effects to make predictions only makes sense if predictions are being made about the same groups/clusters.

        Args:
            data (pandas.core.frame.DataFrame): input data to make predictions on
            use_rfx (bool): whether to condition on random effects when making predictions
            pred_type (str): whether the prediction should be on the 'response' scale (default); or on the 'link' scale of the predictors passed through the link function (e.g. log-odds scale in a logit model instead of probability values)

        Returns:
            ndarray: prediction values

        """
        required_cols = self.design_matrix.columns[1:]
        if not all([col in data.columns for col in required_cols]):
            raise ValueError("Column names do not match all fixed effects model terms!")

        if use_rfx:
            required_cols = set(list(required_cols) + list(self.grps.keys()))
            if not all([col in data.columns for col in required_cols]):
                raise ValueError("Column names are missing random effects model grouping terms!")

            re_form = 'NULL'
        else:
            re_form = 'NA'

        rstring = """
            function(model,new){
            out <- predict(model,new,allow.new.levels=TRUE,re.form="""+re_form+""",type='"""+pred_type+"""')
            out
            }
        """

        predict_func = robjects.r(rstring)
        preds = pandas2ri.ri2py(predict_func(self.model_obj,data))
        return preds

    def summary(self):
        """
        Summarize the output of a fitted model.

        """

        if not self.fitted:
            raise RuntimeError("Model must be fitted to generate summary!")

        print("Formula: {}\n".format(self.formula))
        print("Family: {}\t Inference: {}\n".format(self.family,self.sig_type))
        print("Number of observations: %s\t Groups: %s\n" % (self.data.shape[0],self.grps))
        print("Log-likelihood: %.3f \t AIC: %.3f\n" % (self.logLike,self.AIC))
        print("Random effects:\n")
        print("%s\n" % (self.ranef_var.round(3)))
        if self.ranef_corr is not None:
            print("%s\n" % (self.ranef_corr.round(3)))
        else:
            print("No random effect correlations specified\n")
        print("Fixed effects:\n")
        return self.coefs.round(3)

    def post_hoc(self,marginal_vars, grouping_vars=None,p_adjust='tukey',summarize=True):
        """
        Post-hoc pair-wise tests corrected for multiple comparisons (Tukey method) implemented using the lsmeans package. This method provide both marginal means/trends along with marginal pairwise differences. More info can be found at: https://cran.r-project.org/web/packages/lsmeans/lsmeans.pdf

        Args:
            marginal_var (str/list): what variable(s) to compute marginal means/trends for; unique combinations of factor levels of these variable(s) will determine family-wise error correction
            grouping_vars (str/list): what variable(s) to group on. Trends/means/comparisons of other variable(s), will be computed at each level of these variable(s)
            p_adjust (str): multiple comparisons adjustment method. One of: tukey, bonf, fdr, hochberg, hommel, holm, dunnet, mvt (monte-carlo multi-variate T, aka exact tukey/dunnet). Default tukey
            summarize (bool): output effects and contrasts or don't (always stored in model object as model.marginal_estimates and model.marginal_contrasts); default True

        Returns:
            marginal_estimates (pd.Dataframe): unique factor level effects (e.g. means/coefs)
            marginal_contrasts (pd.DataFrame): contrasts between factor levels

        Examples:

            Pairwise comparison of means of A at each level of B

            >>> model.post_hoc(marginal_vars='A',grouping_vars='B')

            Pairwise differences of slopes of C between levels of A at each level of B

            >>> model.post_hoc(marginal_vars='C',grouping_vars=['A','B'])

            Pairwise differences of each unique A,B cell

            >>> model.post_hoc(marginal_vars=['A','B'])

        """

        if not marginal_vars:
            raise ValueError("Must provide marginal_vars")

        if not self.fitted:
            raise RuntimeError("Model must be fitted to generate post-hoc comparisons")

        if not isinstance(marginal_vars, list):
            marginal_vars = [marginal_vars]

        if grouping_vars and not isinstance(grouping_vars, list):
            grouping_vars = [grouping_vars]
            # Conditional vars can only be factor types
            if not all([elem in self.factors.keys() for elem in grouping_vars]):
                raise ValueError("All grouping_vars must be existing categorical variables (i.e. factors)")

        # Need to figure out if marginal_vars is continuous or not to determine lstrends or lsmeans call
        cont,factor = [],[]
        for var in marginal_vars:
            if not self.factors or var not in self.factors.keys():
                cont.append(var)
            else:
                factor.append(var)

        if cont:
            if factor:
                raise ValueError("With more than one marginal variable, all variables must be categorical factors. Mixing continuous and categorical variables is not supported. Try passing additional categorical factors to grouping_vars""")
            else:
                if len(cont) > 1:
                    raise ValueError("Marginal variables can only contain one continuous variable")
                elif len(cont) == 1:
                    if grouping_vars:
                        # Lstrends
                        cont = cont[0]
                        if len(grouping_vars) > 1:
                            g1 = grouping_vars[0]
                            _conditional = '+'.join(grouping_vars[1:])

                            rstring = """
                                function(model){
                                suppressMessages(library(lsmeans))
                                out <- lstrends(model,pairwise ~ """ + g1 + """|""" + _conditional + """,var='""" + cont + """',adjust='""" + p_adjust + """')
                                out
                                }"""
                        else:
                            rstring = """
                                function(model){
                                suppressMessages(library(lsmeans))
                                out <- lstrends(model,pairwise ~ """ + grouping_vars[0] + """,var='""" + cont + """',adjust='""" + p_adjust + """')
                                out
                                }"""

                    else:
                        raise ValueError("grouping_vars are required with a continuous marginal_vars")
        else:
            if factor:
                _marginal = '+'.join(factor)
                if grouping_vars:
                    # Lsmeans with pipe
                    _conditional = '+'.join(grouping_vars)
                    rstring = """
                        function(model){
                        suppressMessages(library(lsmeans))
                        out <- lsmeans(model,pairwise ~ """ + _marginal + """|""" + _conditional + """, adjust='""" + p_adjust + """')
                        out
                        }"""
                else:
                    # Lsmeans without pipe
                    rstring = """
                        function(model){
                        suppressMessages(library(lsmeans))
                        out <- lsmeans(model,pairwise ~ """ + _marginal + """,adjust='""" + p_adjust + """')
                        out
                        }"""
            else:
                raise ValueError("marginal_vars are not in model!")


        func = robjects.r(rstring)
        res = func(self.model_obj)
        base = importr('base')
        lsmeans = importr('lsmeans')

        # Marginal estimates
        self.marginal_estimates = pandas2ri.ri2py(base.summary(res.rx2('lsmeans')))
        # Resort columns
        effect_names = list(self.marginal_estimates.columns[:-4])
        effname = effect_names[-1] # this column name changes depending on whether we're doing post-hoc trends or means
        sorted = effect_names[:-1] + ['Estimate','2.5_ci','97.5_ci','SE','DF']
        self.marginal_estimates = self.marginal_estimates.rename(columns={effname:'Estimate','df':'DF','lower.CL':'2.5_ci','upper.CL':'97.5_ci'})[sorted]

        # Marginal Contrasts
        self.marginal_contrasts = pandas2ri.ri2py(base.summary(res.rx2('contrasts'))).rename(columns={'t.ratio':'T-stat','p.value':'P-val','estimate':'Estimate','df':'DF','contrast':'Contrast'})
        # Need to make another call to lsmeans to get confidence intervals on contrasts
        confs = pandas2ri.ri2py(base.unclass(lsmeans.confint_ref_grid(res))[1]).iloc[:,-2:].rename(columns={'lower.CL':'2.5_ci','upper.CL':'97.5_ci'})
        self.marginal_contrasts = pd.concat([self.marginal_contrasts,confs],axis=1)
        # Resort columns
        effect_names = list(self.marginal_contrasts.columns[:-7])
        sorted = effect_names + ['Estimate','2.5_ci','97.5_ci','SE','DF','T-stat','P-val']
        self.marginal_contrasts = self.marginal_contrasts[sorted]
        self.marginal_contrasts['Sig'] = self.marginal_contrasts['P-val'].apply(_sig_stars)


        if p_adjust == 'tukey' and self.marginal_contrasts.shape[0] >= self.marginal_estimates.shape[0]:
            print("P-values adjusted by tukey method for family of {} estimates".format(self.marginal_contrasts['Contrast'].nunique()))
        elif p_adjust != 'tukey':
            print("P-values adjusted by {} method for {} comparisons".format(p_adjust,self.marginal_contrasts['Contrast'].nunique()))
        if summarize:
            return self.marginal_estimates.round(3), self.marginal_contrasts.round(3)

    def plot_summary(self,figsize=(12,6),error_bars= 'ci',ranef=True,xlim=None,intercept=True,ranef_alpha=.5,coef_fmt='o',**kwargs):
        """
        Create a forestplot overlaying estimated coefficients with random effects (i.e. BLUPs). By default display the 95% confidence intervals computed during fitting.

        Args:
            error_bars (str): one of 'ci' or 'se' to change which error bars are plotted; default 'ci'
            ranef (bool): overlay BLUP estimates on figure; default True
            xlim (tuple): lower and upper xlimit of plot; default min and max of BLUPs
            intercept (bool): plot the intercept estimate; default True
            ranef_alpha (float): opacity of random effect points; default .5
            coef_fmt (str): matplotlib marker style for population coefficients

        Returns:
            matplotlib axis handle
        """

        if not self.fitted:
            raise RuntimeError("Model must be fit before plotting!")

        if isinstance(self.fixef,list):
            ranef_idx = kwargs.pop("ranef_idx",0)
            print("Multiple random effects clusters specified in model. Plotting the {} one. This can be changed by passing 'ranef_idx = number'".format(ranef_idx+1))
            m_ranef = self.fixef[ranef_idx]
        else:
            m_ranef = self.fixef
        m_fixef = self.coefs

        if not intercept:
            m_ranef = m_ranef.drop('(Intercept)',axis=1)
            m_fixef = m_fixef.drop('(Intercept)',axis=0)

        if error_bars == 'ci':
            col_lb = m_fixef['Estimate'] - m_fixef['2.5_ci']
            col_ub = m_fixef['97.5_ci'] - m_fixef['Estimate']
        elif error_bars == 'se':
            col_lb,col_ub = m_fixef['SE'], m_fixef['SE']

        # For seaborn
        m = pd.melt(m_ranef)

        if not xlim:
            xlim = (m['value'].min()-1, m['value'].max()+1)

        f,ax = plt.subplots(1,1,figsize=figsize)

        if ranef:
            alpha_plot=ranef_alpha
        else:
            alpha_plot=0

        sns.stripplot(x='value',y='variable',data=m,ax=ax,size=6,alpha=alpha_plot,color='grey');

        ax.errorbar(x=m_fixef['Estimate'],y=range(m_fixef.shape[0]),xerr=[col_lb,col_ub],fmt=coef_fmt,capsize=0,elinewidth=4,color='black',ms=12,zorder=9999999999);

        ax.vlines(x=0,ymin=-1,ymax=self.coefs.shape[0],linestyles='--',color='grey')

        ax.set(ylabel='',xlabel='Estimate',xlim=xlim);
        sns.despine(top=True,right=True,left=True);
        return ax

    def plot(self,param,figsize=(8,6),xlabel='',ylabel='',plot_fixef=True,plot_ci=True,grps=[],ax=None):
        """
        Plot random and group level parameters from a fitted model

        Args:
            param (str): model parameter (column name) to plot
            figsize (tup): matplotlib desired figsize
            xlabel (str): x-axis label
            ylabel (str): y-axis label
            plot_fixef (bool): plot population effect fit of param?; default True
            plot_ci (bool): plot computed ci's of population effect?; default True
            grps (list): plot specific group fits only; must correspond to index values in model.fixef
            ax (matplotlib.axes.Axes): axis handle for an existing plot; if provided will ensure that random parameter plots appear *behind* all other plot objects.

        Returns:
            matplotlib axis handle

        """

        if not self.fitted:
            raise RuntimeError("Model must be fit before plotting!")
        if self.factors:
            raise NotImplementedError("Plotting can currently only handle models with continuous predictors!")
        if isinstance(self.fixef, list) or isinstance(self.ranef,list):
            raise NotImplementedError("Plotting can currently only handle models with 1 random effect grouping variable!")
        if not ax:
            f,ax = plt.subplots(1,1,figsize=figsize);

        #Get range of unique values for desired parameter
        x_vals = self.design_matrix[param].unique()
        #Sort order to handle bug in matplotlib plotting
        idx = np.argsort(x_vals)

        #Get desired parameter part of the prediction
        fixef_pred = self.coefs.loc['(Intercept)','Estimate'] + self.coefs.loc[param,'Estimate']*x_vals
        fixef_pred_upper = self.coefs.loc['(Intercept)','97.5_ci'] + self.coefs.loc[param,'97.5_ci']*x_vals
        fixef_pred_lower = self.coefs.loc['(Intercept)','2.5_ci'] + self.coefs.loc[param,'2.5_ci']*x_vals

        if grps:
            if all(isinstance(x, int) for x in grps):
                ran_dat = self.fixef.iloc[grps,:]
            elif all(isinstance(x, str) for x in grps):
                ran_dat = self.fixef.loc[grps,:]
            else:
                raise TypeError('grps must be integer list for integer-indexing (.iloc) of fixed effects, or label list for label-indexing (.loc) of fixed effects')
        else:
            ran_dat = self.fixef

        #Now generate random effects predictions
        for i, row in ran_dat.iterrows():

            ranef_desired = row['(Intercept)'] + row[param]*x_vals
            #ranef_other = np.dot(other_vals_means, row.loc[other_vals])
            pred = ranef_desired #+ ranef_other

            ax.plot(x_vals[idx],pred[idx],'-',linewidth=2);

        if plot_fixef:
            ax.plot(x_vals[idx],fixef_pred[idx],'--',color='black',linewidth=3,zorder=9999999);

        if plot_ci:
            ax.fill_between(x_vals[idx],fixef_pred_lower[idx],fixef_pred_upper[idx],facecolor='black',alpha=.25,zorder=9999998);

        ax.set(ylim = (self.data.fits.min(), self.data.fits.max()),
               xlim = (x_vals.min(),x_vals.max()),
               xlabel = param,
               ylabel = self.formula.split('~')[0].strip());
        if xlabel:
            ax.set_xlabel(xlabel);
        if ylabel:
            ax.set_ylabel(ylabel);
        return ax

class Lm(object):

    """
    Model class to perform OLS regression. Formula specification works just like in R based on columns of a dataframe. Formulae are parsed by patsy which makes it easy to utilize specifiy columns as factors. This is **different** from Lmer. See patsy for more information on the different use cases.

    Args:
        formula (str): Complete lm-style model formula
        data (pandas.core.frame.DataFrame): input data
        family (string): what distribution family (i.e.) link function to use for the generalized model; default is gaussian (linear model)

    Attributes:
        fitted (bool): whether model has been fit
        formula (str): model formula
        data (pandas.core.frame.DataFrame): model copy of input data
        grps (dict): groups and number of observations per groups recognized by lmer
        AIC (float): model akaike information criterion
        logLike (float): model Log-likelihood
        family (string): model family
        ranef (pandas.core.frame.DataFrame/list): cluster-level differences from population parameters, i.e. difference between coefs and fixefs; returns list if multiple cluster variables are used to specify random effects (e.g. subjects and items)
        fixef (pandas.core.frame.DataFrame/list): cluster-level parameters; returns list if multiple cluster variables are used to specify random effects (e.g. subjects and items)
        coefs (pandas.core.frame.DataFrame/list): model summary table of population parameters
        resid (numpy.ndarray): model residuals
        fits (numpy.ndarray): model fits/predictions
        model_obj(lm model): rpy2 lmer model object
        factors (dict): factors used to fit the model if any

    """

    def __init__(self,formula,data,family='gaussian'):

        self.family = family
        #implemented_fams = ['gaussian','binomial']
        if self.family != 'gaussian':
            raise NotImplementedError("Currently only linear (family ='gaussian') models supported! ")
        self.fitted = False
        self.formula = formula
        self.data = copy(data)
        self.AIC = None
        self.logLike = None
        self.warnings = []
        self.resid = None
        self.coefs = None
        self.model_obj = None
        self.factors = None
        self.ci_type = None
        self.se_type = None
        self.sig_type = None

    def __repr__(self):
        out = "{}.{}(fitted={}, formula={}, family={})".format(
        self.__class__.__module__,
        self.__class__.__name__,
        self.fitted,
        self.formula,
        self.family)
        return out

    def fit(self,robust=False,conf_int='standard',permute=None,summarize=True,verbose=False,n_boot=500,n_jobs=-1,n_lags=1,cluster=None):
        """
        Fit a variety of OLS models. By default will fit a model that makes parametric assumptions (under a t-distribution) replicating the output of software like R. 95% confidence intervals (CIs) are also estimated parametrically by default. However, empirical bootstrapping can also be used to compute CIs; this procedure resamples with replacement from the data themselves, not residuals or data generated from fitted parameters.

        Alternatively, OLS robust to heteroscedasticity can be fit by computing sandwich standard error estimates. This is similar to Stata's robust routine.
        Robust estimators include:

        - 'hc0': Huber (1980) original sandwich estimator

        - 'hc3': MacKinnon and White (1985) HC3 sandwich estimator; provides more robustness in smaller samples than hc0, Long & Ervin (2000)

        - 'hac': Newey-West (1987) estimator for robustness to heteroscedasticity as well as serial auto-correlation at given lags.

        - 'cluster' : cluster-robust standard errors (see Cameron & Miller 2015 for review). Provides robustness to errors that cluster according to specific groupings (e.g. repeated observations within a person/school/site). This acts as post-modeling "correction" for what a multi-level model explicitly estimates and is popular in the econometrics literature. DOF correction differs slightly from stat/statsmodels which use num_clusters - 1, where as pymer4 uses num_clusters - num_coefs

        Args:
            robust (bool/str): whether to use heteroscedasticity robust s.e. and optionally which estimator type to use ('hc0','hc3','hac','cluster'). If robust = True, default robust estimator is 'hc0'; default False
            conf_int (str): whether confidence intervals should be computed through bootstrap ('boot') or assuming a t-distribution ('standard'); default 'standard'
            permute (int): if non-zero, computes parameter significance tests by permuting t-stastics rather than parametrically; works with robust estimators
            summarize (bool): whether to print a model summary after fitting; default True
            verbose (bool): whether to print which model, standard error, confidence interval, and inference type are being fitted
            n_boot (int): how many bootstrap resamples to use for confidence intervals (ignored unless conf_int='boot')
            n_jobs (int): number of cores for parallelizing bootstrapping or permutations; default all cores
            n_lags (int): number of lags for robust estimator type 'hac' (ignored unless robust='hac'); default 1
            cluster (str): column name identifying clusters/groups for robust estimator type 'cluster' (ignored unless robust='cluster')

        Returns:
            DataFrame: R style summary() table

        """
        if permute and permute < 500:
            w = 'Permutation testing < 500 permutations is not recommended'
            warnings.warn(w)
            self.warnings.append(w)
        if robust:
            if isinstance(robust,bool):
                robust = 'hc0'
            self.se_type = 'robust' + ' (' + robust + ')'
            if cluster:
                if cluster not in self.data.columns:
                    raise ValueError("cluster identifier must be an existing column in data")
                else:
                    cluster = self.data[cluster]
        else:
            self.se_type = 'non-robust'

        if self.family == 'gaussian':
            if verbose:
                if not robust:
                    print_robust = 'non-robust'
                else:
                    print_robust = 'robust ' + robust

                if conf_int == 'boot':
                    print("Fitting linear model with "+print_robust+ " standard errors and \n" + str(n_boot) + " bootstrapped 95% confidence intervals...\n")
                else:
                    print("Fitting linear model with "+print_robust+ " standard errors\nand 95% confidence intervals...\n")

                if permute:
                    print("Using {} permutations to determine significance...".format(permute))

        self.ci_type = conf_int + ' (' + str(n_boot) + ')' if conf_int == 'boot' else conf_int
        self.sig_type = 'parametric' if not permute else 'permutation' + ' (' + str(permute) + ')'

        # Parse formula using patsy to make design matrix
        y,x = dmatrices(self.formula,self.data,1,return_type='dataframe')
        self.design_matrix = x

        # Compute standard estimates
        b, se, t, res = _ols(x,y,robust,all_stats=True,n_lags=n_lags,cluster=cluster)
        if cluster is not None:
            # Cluster corrected dof (num clusters - num coef)
            # Differs from stats and statsmodels which do num cluster - 1
            # Ref: http://cameron.econ.ucdavis.edu/research/Cameron_Miller_JHR_2015_February.pdf
            df = cluster.nunique() - x.shape[1]
        else:
            df = x.shape[0] - x.shape[1]
        p = 2*(1-t_dist.cdf(np.abs(t), df))
        df = np.array([df]*len(t))
        sig = np.array([_sig_stars(elem) for elem in p])

        if conf_int == 'boot':

            # Parallelize bootstrap computation for CIs
            par_for = Parallel(n_jobs=n_jobs,backend='multiprocessing')

            # To make sure that parallel processes don't use the same random-number generator pass in seed (sklearn trick)
            seeds = np.random.randint(np.iinfo(np.int32).max,size=n_boot)

            # Since we're bootstrapping coefficients themselves we don't need the robust info anymore
            boot_betas = par_for(delayed(                          _chunk_boot_ols_coefs)(dat=self.data,formula=self.formula,seed=seeds[i]) for i in range(n_boot))

            boot_betas = np.array(boot_betas)
            ci_u = np.percentile(boot_betas,97.5,axis=0)
            print(ci_u)
            ci_l = np.percentile(boot_betas,2.5,axis=0)

        else:
            # Otherwise we're doing parametric CIs
            ci_u = b + t_dist.ppf(.975,df)*se
            ci_l = b + t_dist.ppf(.025,df)*se

        if permute:
            # Permuting will change degrees of freedom to num_iter and p-values
            # Parallelize computation
            # Unfortunate monkey patch that robust estimation hangs with multiple processes; maybe because of function nesting level??
            # _chunk_perm_ols -> _ols -> _robust_estimator
            if robust:
                n_jobs = 1
            par_for = Parallel(n_jobs=n_jobs,backend='multiprocessing')
            seeds = np.random.randint(np.iinfo(np.int32).max,size=permute)
            perm_ts = par_for(delayed(                          _chunk_perm_ols
            )(x=x,y=y,robust=robust,n_lags=n_lags,cluster=cluster,seed=seeds[i]) for i in range(permute))
            perm_ts = np.array(perm_ts)

            p = []
            for col, fit_t in zip(range(perm_ts.shape[1]),t):
                p.append(_perm_find(perm_ts[:,col],fit_t))
            p = np.array(p)
            df = np.array([permute]*len(p))
            sig = np.array([_sig_stars(elem) for elem in p])


        #Make output df
        results = np.column_stack([b,ci_l,ci_u,se,df,t,p,sig])
        results = pd.DataFrame(results)
        results.index = x.columns
        results.columns = ['Estimate','2.5_ci','97.5_ci','SE','DF','T-stat','P-val','Sig']
        results[['Estimate','2.5_ci','97.5_ci','SE','DF','T-stat','P-val']] = results[['Estimate','2.5_ci','97.5_ci','SE','DF','T-stat','P-val']].apply(pd.to_numeric)

        if permute:
            results = results.rename(columns={'DF':'Num_perm','P-val':'Perm-P-val'})

        self.coefs = results
        self.fitted = True
        self.resid = res
        self.data['fits'] = y.squeeze() - res
        self.data['residuals'] = res

        #Fit statistics
        self.rsquared = np.corrcoef(np.dot(x,b),y.squeeze())**2
        self.rsquared = self.rsquared[0,1]
        self.rsquared_adj = 1.-(len(res)-1.)/(len(res)-x.shape[1]) * (1.- self.rsquared)
        half_obs = len(res)/2.0
        ssr = np.dot(res,res.T)
        self.logLike = (-np.log(ssr)* half_obs) - ((1+np.log(np.pi/half_obs))*half_obs)
        self.AIC = 2*x.shape[1] - 2*self.logLike
        self.BIC = np.log((len(res)))*x.shape[1] - 2*self.logLike

        if summarize:
            return self.summary()


    def summary(self):
        """
        Summarize the output of a fitted model.

        """

        if not self.fitted:
            raise RuntimeError("Model must be fitted to generate summary!")

        print("Formula: {}\n".format(self.formula))
        print("Family: {}\n".format(self.family))
        print("Std-errors: {}\tCIs: {} 95%\tInference: {} \n".format(self.se_type,self.ci_type,self.sig_type))
        print("Number of observations: %s\t R^2: %.3f\t R^2_adj: %.3f\n" % (self.data.shape[0],self.rsquared,self.rsquared_adj))
        print("Log-likelihood: %.3f \t AIC: %.3f\t BIC: %.3f\n" % (self.logLike,self.AIC,self.BIC))
        print("Fixed effects:\n")
        return self.coefs.round(3)

    def post_hoc(self):
        raise NotImplementedError("Post-hoc tests are not yet implemented for linear models.")

    def predict(self,data):
        """
        Make predictions given new data. Input must be a dataframe that contains the same columns as the model.matrix excluding the intercept (i.e. all the predictor variables used to fit the model). Will automatically use/ignore intercept to make a prediction if it was/was not part of the original fitted model.

        Args:
            data (pandas.core.frame.DataFrame): input data to make predictions on

        Returns:
            ndarray: prediction values

        """

        required_cols = self.design_matrix.columns[1:]
        if not all([col in data.columns for col in required_cols]):
            raise ValueError("Column names do not match all fixed effects model terms!")
        X = data[required_cols]
        coefs = self.coefs.loc[:,'Estimate'].values
        if self.coefs.index[0] == 'Intercept':
            preds = np.dot(np.column_stack([np.ones(X.shape[0]),X]),coefs)
        else:
            preds = np.dot(X,coefs[1:])
        return preds
