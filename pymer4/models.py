import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
import rpy2
from copy import copy
import pandas as pd
import numpy as np
pandas2ri.activate()

class Lmer(object):

    """Linear model class to hold data outputted from fitting lmer in R and converting to Python object. This class stores as much information as it can about a merMod object computed using lmer and lmerTest in R. Most attributes will not be computed until the fit method is called

    Args:
        formula (string): Complete lmer-style model formula
        data (dataframe): pandas dataframe of input data

    Attributes:
        fitted: (bool) whether fit method has been called yet
        formula: (string) model formula
        data: (dataframe) pandas dataframe used to initialize instance
        ngrps: (float) number of groups recognized by lmer
        AIC: (float) model AIC
        logLike: (float) model Log-likelihood
        family: (string) what model family; currently only linear implemented
        vconv_ranef: (dataframe) pandas dataframe of random effect variances and standard deviations
        fixef: (dataframe) cluster-level fixed effect coefficients, lmer BLUPs
        ranef: (dataframe) cluster-level fixed effect deviations from group fit
        resid: (ndarray) model residuals
        fits: (ndarray) model fits/predictions
        coefs: (dataframe) dataframe of R style summary() table


    """

    def __init__(self,formula,data):

        self.fitted = False
        self.formula = formula
        self.data = data

    def _sig_stars(self,val):
        star = ''
        if 0 < val < .001:
            star = '***'
        elif .001 <= val < 0.01:
            star = '**'
        elif .01 <= val < .05:
            star = '*'
        elif .05 <= val < .1:
            star = '.'
        return star

    def fit(self,conf_int='Wald'):
        """Main method for fitting model object. Will modify the model's data attribute to add columns for residuls and fits for convenience.

        Args:
            conf_int: (string) which method to compute confidence intervals: profile, Wald (default), or boot (parametric bootstrap)

        Returns:
            coefs: (dataframe) dataframe of R style summary() table
            model: (Lmer) model object with numerous additional computed attributions

        """

        print("Fitting model using lmer with "+conf_int+" confidence intervals...\n")
        self.fitted = True
        lmer = importr('lmerTest')
        base = importr('base')

        model = lmer.lmer(self.formula,data=self.data)
        summary = base.summary(model)
        unsum = base.unclass(summary)

        #Do scalars first cause they're easier
        self.ngrps = unsum.rx2('ngrps')[0]
        self.AIC = unsum.rx2('AICtab')[0]
        self.logLike = unsum.rx2('logLik')[0]
        if type(unsum.rx2('family')) is rpy2.rinterface.RNULLType:
            self.family = 'linear'
        else:
            self.family = unsum.rx2('family')[0]

        if len(unsum.rx2('optinfo').rx2('warnings')) == 0:
            self.warnings = None
        else:
            self.warnings = unsum.rx2('optinfo').rx2('warnings')
            for warning in self.warnings:
                print(warning + ' \n')

        #Vcovs
        df = pandas2ri.ri2py(base.data_frame(unsum.rx2('varcor'))).drop('var2',axis=1)
        df.columns = ['Groups','Name','Var','Std']
        self.vcov_ranef = df

        #self.vcov_fixef = Gotta figure out how to parse this

        #Cluster (e.g subject) level coefficients
        rstring = """
            function(model){
            out <- coef(model)
            out
            }
        """
        fixef_func = robjects.r(rstring)
        self.fixef = pandas2ri.ri2py(fixef_func(model)[0])

        #Cluster (e.g subject) level random deviations
        rstring = """
            function(model){
            out <- ranef(model)
            out
            }
        """
        ranef_func = robjects.r(rstring)
        self.ranef = pandas2ri.ri2py(ranef_func(model)[0])

        #Model residuals
        rstring = """
            function(model){
            out <- resid(model)
            out
            }
        """
        resid_func = robjects.r(rstring)
        self.resid = pandas2ri.ri2py(resid_func(model))
        self.data['residuals'] = copy(self.resid)

        #Model fits
        rstring = """
            function(model){
            out <- fitted(model)
            out
            }
        """
        fit_func = robjects.r(rstring)
        self.fits = pandas2ri.ri2py(fit_func(model))
        self.data['fits'] = copy(self.fits)

        #Coefficients, and inference statistics
        rstring = """
            function(model){
            out.df <- data.frame(unclass(summary(model))$coefficients)
            out.df
            }
        """
        estimates_func = robjects.r(rstring)
        df = pandas2ri.ri2py(estimates_func(model))
        df.columns = ['Estimate','Std','DF','T-stat','P-val']

        rstring = """
            function(model){
            out.df <- data.frame(confint(model,method='"""+conf_int+"""'))
            out.df
            }
        """
        conf_func = robjects.r(rstring)
        df_c = pandas2ri.ri2py(conf_func(model))
        df_c = df_c.loc['(Intercept)':,:]
        df_c.columns = ['2.5_ci','97.5_ci']
        df = pd.concat([df,df_c],axis=1)
        df['Sig'] = df['P-val'].apply(lambda x: self._sig_stars(x))
        self.coefs = df

        print("Number of groups: %s\n" % (self.ngrps))
        print("Log-likelihood: %s \t AIC: %s\n" % (self.logLike,self.AIC))
        print("Random effects:\n")
        print self.vcov_ranef
        print("Fixed effects:\n")
        return self.coefs
