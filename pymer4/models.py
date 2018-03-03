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
                          _perm_find
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
        implemented_fams = ['gaussian','binomial']
        if self.family not in implemented_fams:
            raise NotImplementedError("Currently only linear (family ='gaussian') and logisitic (family='binomial') models supported! ")
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
        self.marginal_effects = None
        self.marginal_contrasts = None

    def __repr__(self):
        out = "{}.{}(fitted={}, formula={}, family={})".format(
        self.__class__.__module__,
        self.__class__.__name__,
        self.fitted,
        self.formula,
        self.family)
        return out

    def _make_factors(self,factor_dict, ordered=False):
        """
        Covert specific columns to R-style factors. Default scheme is dummy coding where reference is 1st level provided. Alternative is orthogonal polynomial contrasts

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

        rstring = """
            function(df,f,lv){
            df[,f] <- factor(df[,f],lv,ordered=F)
            df
            }
        """

        factorize = robjects.r(rstring)
        df = copy(self.data)
        for k in factor_dict.keys():
            df[k] = df[k].astype(str)

        r_df = pandas2ri.py2ri(df)
        for k,v in factor_dict.items():

            r_df = factorize(r_df,k,v)

        return r_df

    def anova(self):

        raise NotImplementedError(" ANOVA tables coming soon!")

        #TODO Make sure factors are set with orthogonal contrasts first
        """Compute a Type III ANOVA table on a fitted model."""

        if not self.fitted:
            raise RuntimeError("Model hasn't been fit! Call fit() method before computing ANOVA table.")

        #See rpy2 for building contrasts or loop and construct rstring
        rstring = """
            function(model){
            df<- anova(model)
            df
            }
        """
        anova = robjects.r(rstring)
        self.anova = pandas2ri.ri2py(anova)
        if self.anova.shape[1] == 6:
                self.anova.columns = ['SS','MS','NumDF','DenomDF','F-stat','P-val']
                self.anova['Sig'] = self.anova['P-val'].apply(lambda x: _sig_stars(x))
        elif self.anova.shape[1] == 4:
            warnings.warn("MODELING FIT WARNING! Check model.warnings!! P-value computation did not occur because lmerTest choked. Possible issue(s): ranefx have too many parameters or too little variance...")
            self.anova.columns = ['DF','SS','MS','F-stat']
        print("Analysis of variance Table of type III with Satterthwaite approximated degrees of freedom:\n")
        return self.anova

    def fit(self,conf_int='Wald',factors=None,ordered=False,summarize=True,verbose=False):
        """
        Main method for fitting model object. Will modify the model's data attribute to add columns for residuals and fits for convenience.

        Args:
            conf_int (str): which method to compute confidence intervals; 'profile', 'Wald' (default), or 'boot' (parametric bootstrap)
            factors (dict): col names (keys) to treat as dummy-coded factors with levels specified by unique values (vals). First level is always reference, e.g. {'Col1':['A','B','C']}
            ordered (bool): whether factors should be treated as ordered polynomial contrasts; this will parameterize a model with K-1 orthogonal polynomial regressors beginning with a linear contrast based on the factor order provided; default is False
            summarize (bool): whether to print a model summary after fitting; default is True
            verbose (bool): whether to print when and which model and confidence interval are being fitted

        Returns:
            DataFrame: R style summary() table

        """

        if factors:
            dat = self._make_factors(factors,ordered)
            self.factors = factors
        else:
            dat = self.data

        if self.family == 'gaussian':
            if verbose:
                print("Fitting linear model using lmer with "+conf_int+" confidence intervals...\n")
            lmer = importr('lmerTest')
            self.model_obj = lmer.lmer(self.formula,data=dat)
        else:
            if verbose:
                print("Fitting generalized linear model using glmer with "+conf_int+" confidence intervals...\n")
            lmer = importr('lme4')
            self.model_obj = lmer.lmer(self.formula,data=dat,family=self.family)

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
        if self.family == 'gaussian':

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

            if df.shape[1] == 7:
                df.columns = ['Estimate','SE','DF','T-stat','P-val','2.5_ci','97.5_ci']
                df = df[['Estimate','2.5_ci','97.5_ci','SE','DF','T-stat','P-val']]

            elif df.shape[1] == 5:
                #Incase lmerTest chokes it won't return p-values
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

        if 'P-val' in df.columns:
            df['Sig'] = df['P-val'].apply(lambda x: _sig_stars(x))
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

        # TODO Parse and store fixed effects correlation matrix +enhancement id:2 gh:11

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

    def post_hoc(self,marginal_vars, grouping_vars=None,p_adjust='tukey'):
        """
        Post-hoc pair-wise tests corrected for multiple comparisons (Tukey method) implemented using the lsmeans package. This method provide both marginal means/trends along with marginal pairwise differences. More info can be found at: https://cran.r-project.org/web/packages/lsmeans/lsmeans.pdf

        Args:
            marginal_var (str): what variable(s) to compute marginal means/trends for; unique combinations of factor levels of these variable(s) will determine family-wise error correction
            grouping_vars (str/list): what variable(s) to group on. Trends/means/comparisons of other variable(s), will be computed at each level of these variable(s)
            p_adjust (str): multiple comparisons adjustment method. One of: tukey, bonf, fdr, hochberg, hommel, holm, dunnet, mvt (monte-carlo multi-variate T, aka exact tukey/dunnet). Default tukey

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
            if var not in self.factors.keys():
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
                        _conditional = '+'.join(grouping_vars)
                        rstring = """
                            function(model){
                            suppressMessages(library(lsmeans))
                            out <- lstrends(model,pairwise ~ """ + _conditional + """,var='""" + cont + """',adjust='""" + p_adjust + """')
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
                        out <- lsmeans(model,pairwise ~ """ + _marginal + """|""" + _conditional + """, adjust=''""" + p_adjust + """')
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

        self.marginal_effects = pandas2ri.ri2py(base.summary(res.rx2('lsmeans')))
        self.marginal_contrasts = pandas2ri.ri2py(base.summary(res.rx2('contrasts')))

        if p_adjust == 'tukey' and self.marginal_contrasts.shape[0] >= self.marginal_effects.shape[0]:
            print("P-values adjusted by tukey method for family of {} estimates".format(self.marginal_effects.shape[0]))
        elif p_adjust != 'tukey':
            print("P-values adjusted by {} method for {} comparisons".format(p_adjust,self.marginal_contrasts.shape[0]))

        return self.marginal_effects.round(3), self.marginal_contrasts.round(3)

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
            matplotlib figure handle
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
        return f,ax

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
        implemented_fams = ['gaussian','binomial']
        if self.family not in implemented_fams:
            raise NotImplementedError("Currently only linear (family ='gaussian') and logisitic (family='binomial') models supported! ")
        self.fitted = False
        self.formula = formula
        self.data = copy(data)
        self.grps = None
        self.AIC = None
        self.logLike = None
        self.warnings = []
        self.fixef = None
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

    def fit(self,robust=False,conf_int='standard',permute=None,summarize=True,verbose=False,n_boot=500,n_jobs=-1,n_lags=1):
        """
        Fit a variety of OLS models. By default will fit a model that makes parametric assumptions (under a t-distribution) replicating the output of software like R. 95% confidence intervals (CIs) are also estimated parametrically by default. However, empirical bootstrapping can also be used to compute CIs; this procedure resamples with replacement from the data themselves, not residuals or data generated from fitted parameters.

        Alternatively, OLS robust to heteroscedasticity can be fit by computing sandwich standard error estimates. This is similar to Stata's robust routine.
        Robust estimators include:
        - 'hc0': Huber (1980) original sandwich estimator
        - 'hc3': MacKinnon and White (1985) HC3 sandwich estimator; provides more robustness in smaller samples than hc0, Long & Ervin (2000)
        - 'hac': Newey-West (1987) estimator for robustness to heteroscedasticity as well as serial auto-correlation at given lags.

        Args:
            robust (bool/str): whether to use heteroscedasticity robust s.e.and optionally which estimator type to use ('hc0','hc3','hac'). If robust = True, default robust estimator is 'hc0'; default False
            conf_int (str): whether confidence intervals should be computed through bootstrap ('boot') or assuming a t-distribution ('standard'); default 'standard'
            permute (int): if non-zero, computes parameter significance tests by permuting t-stastics rather than parametrically; works with robust estimators
            summarize (bool): whether to print a model summary after fitting; default True
            verbose (bool): whether to print which model, standard error, confidence interval, and inference type are being fitted
            n_boot (int): how many bootstrap resamples to use for confidence intervals (ignored unless conf_int='boot')
            n_jobs (int): number of cores for parallelizing bootstrapping or permutations; default all cores
            n_lags (int): number of lags for robust estimator type 'hac' (ignored unless robust='hac'); default 1

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
                    print("Using " +str(permute)+ " permutations to determine significance...")

        self.ci_type = conf_int + ' (' + str(n_boot) + ')' if conf_int == 'boot' else conf_int
        self.sig_type = 'parametric' if not permute else 'permute' + ' (' + str(permute) + ')'

        # Parse formula using patsy to make design matrix
        y,x = dmatrices(self.formula,self.data,1,return_type='dataframe')
        self.design_matrix = x

        # Compute standard estimates
        b, se, t, res = _ols(x,y,robust,all_stats=True,n_lags=n_lags)
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
            )(x=x,y=y,robust=robust,n_lags=n_lags,seed=seeds[i]) for i in range(permute))
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
        print("Std-errors: {}\tCIs: {} 95%\tInference: {} \n".format(self.se_type,self.ci_type,self.sig_type))
        print("Number of observations: %s\t R^2: %.3f\t R^2_adj: %.3f\n" % (self.data.shape[0],self.rsquared,self.rsquared_adj))
        print("Log-likelihood: %.3f \t AIC: %.3f\t BIC: %.3f\n" % (self.logLike,self.AIC,self.BIC))
        print("Fixed effects:\n")
        return self.coefs.round(3)

    def post_hoc(self):
        raise NotImplementedError("Post-hoc tests are not yet implemented for linear models.")
