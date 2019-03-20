[![Build Status](https://travis-ci.org/ejolly/pymer4.svg?branch=master)](https://travis-ci.org/ejolly/pymer4)
[![Package versioning](https://img.shields.io/pypi/v/pymer4.svg)](https://pypi.python.org/pypi?name=pymer4&version=0.6.0&:action=display)
[![Python Versions](https://img.shields.io/pypi/pyversions/pymer4.svg)](https://pypi.python.org/pypi?name=pymer4&version=0.2.2&:action=display)
[![DOI](http://joss.theoj.org/papers/10.21105/joss.00862/status.svg)](https://doi.org/10.21105/joss.00862)
[![DOI](https://zenodo.org/badge/90598701.svg)](https://zenodo.org/record/1523205)

# pymer4

Love multi-level-modeling using lme4 in R but prefer to work in the scientific Python ecosystem? Then this package has got you covered! It's a small convenience package wrapping the basic functionality of [lme4](https://github.com/lme4/lme4)\*.  

This package can also estimate standard, robust, and permuted regression models\*  
\* *Currently this only includes linear regression models*

## Documentation and Citing
Current documentation and usage examples can be found **[here](http://eshinjolly.com/pymer4/)**.  

If you use this software please cite as:  
Jolly, (2018). Pymer4: Connecting R and Python for Linear Mixed Modeling. *Journal of Open Source Software*, *3*(31), 862, https://doi.org/10.21105/joss.00862

## Installation  

`pymer4` since version 0.6.0 is only compatible with Python 3. Versions 0.5.0 and lower will work with Python 2, but will not contain any new features. `pymer4` also requires a working R installation with specific packages installed and it will *not* install R or these packages for you. However, you can follow either option below to easily handle these dependencies.

## Option 1 (simpler but slower model fitting)

If you don't have R installed and you use the Anaconda Python distribution simply run the following commands to have Anaconda install R and the required packages for you. This is fairly painless installation, but model fitting will be slower than if you install R and `pymer4` separately and configure them (option 2).

1. `conda install -c conda-forge r r-base r-lmertest r-lsmeans rpy2=2.9.4`  
2. `pip install pymer4`
3. Test the installation to see if it's working by running: `python -c "from pymer4.test_install import test_install; test_install()"`
4. If there are errors follow the guide below

## Option 2 (potentially trickier, but faster model fitting)  

This method assumes you already have R installed. If not install first install it from the [R Project website](https://www.r-project.org/). Then complete the following steps:

1. Install the required R packages by running the following command from within R: `install.packages(c('lme4','lmerTest','lsmeans'))`
2. Install pymer4: `pip install pymer4`
3. Test the installation to see if it's working by running: `python -c "from pymer4.test_install import test_install; test_install()"`
4. If there are errors follow the guide below  

### Installation issues

If you run into issues using either option above, it's likely due to compiler issues that give `rpy2` (a package dependency of `pymer4`) some issues during install. The instructions below should fix that on macOS:

1. Install [homebrew](https://brew.sh/) if you don't have it already, by running the command at the link (it's a great pacakage manager for macOS). To check if you already have it, do `which brew` in your Terminal. If nothing pops up you don't have it.
2. Fix brew permissions: `sudo chown -R $(whoami) $(brew --prefix)/*` (this is **necessary** on macOS Sierra or higher (>= macOS 10.12))
3. Update homebrew `brew update`
4. Install the xz utility `brew install xz`
5. At this point you can try to re-install `pymer4` and re-test the installation. If it still doesn't work follow the next few steps below
6. Install an updated compiler: `brew install gcc`, or if you have homebrew already, `brew upgrade gcc`
7. Enable the new compiler for use:
    ```
    export CC="$(find `brew info gcc | grep usr | sed 's/(.*//' | awk '{printf $1"/bin"}'` -name 'x86*gcc-?')"
    export CFLAGS="-W"
    ```
8. If this doesn't work for you might need to manually find out where the new compiler is installed. To do so use `brew info gcc` and `cd` into the directory that begins with `/usr` in the output of that command. From there `cd` into `bin` and look for a file that begins with `x86` and ends with `gcc-7`. It's possible that the directory ends with `gcc-8` or a higher number based on how recently you installed from homebrew. In that case, just use the latest version. Copy the *full path* to that file and run the following:
    ```
    export CC= pathYouCopiedInQuotes
    export CFLAGS="-W"
    ```
9. Finally install `rpy2` using the new compiler you just installed: `conda install -c conda-forge rpy2` if you followed Option 1 above or `pip install rpy2` if you followed Option 2
10. Now you should be able to `pip install pymer4` :)

## Testing  

`git clone https://github.com/ejolly/pymer4.git`

#### Option 1 (Using [Tox](https://tox.readthedocs.io/en/latest/))  

`tox` in project root directory for all tests  

#### Option 2 (Using [Pytest](https://docs.pytest.org/en/latest/))  

`pytest -s --capture=no` in project root for all tests  

`pytest pymer4/tests/test_models.py -k "test_gaussian_lm"` for specific tests


#### Change-log  
**0.7.0**
- Addition of a `pymer4.stats` module for various parametrics and non-parametric statistics functions (i.e. permutation testing and bootstrapping)
- Addition of `Lm2` models that can perform multi-level modeling by first estimating `Lm` separately for each group and then performing inference on those estimates. Can perform inference on first-level semi-partial and partial correlation coefficients instead of betas too.
- `Lm` models can also perform inference on partial or semi-partial correlation coefficients
- All model clases now have the ability to rank transform data prior to estimation 
- Automated testing on travis now pins specific r and r-package versions. This is because of recent changes to the `lme4` and other R packages which result in slightly different estimates that the associated `conda-forge` versions see [this issue](https://github.com/ejolly/pymer4/issues/37). Additionally obtaining references to elements via `rpy2` seems to have changed in some cases (e.g. recent version of `lsmeans` on `conda-forge`). If users are installing via option 1 and are running into issues, it's suggested they try the following installation command instead `conda install -c conda-forge r-base=3.4.1 r-lme4=1.1_13 r-lmertest=3.0_1 r-lsmeans=2.27_62 rpy2=2.9.4`  

**0.6.0**  
- Upgraded to latest version of `rpy2`, meaning that from this version onwards `pymer4` is **only compatible with Python 3**.  
- This has the direct benefit of making installation *substantially easier* by using Anaconda and the less problematic recent versions of `rpy2`  

**0.5.0**
- `Lmer` models now support all generalized linear model family types supported by lme4 (e.g. poisson, gamma, etc)
- `Lmer` models now support ANOVA tables with support for auto-orthogonalizing factors
- Test statistic inference for `Lmer` models can now be performed via non-parametric permutation tests that shuffle observations within clusters
- `Lmer.fit(factors={})` arguments now support custom arbitrary contrasts
- New forest plots for visualizing model estimates and confidence intervals via the `Lmer.plot_summary()` method
- More comprehensive documentation with examples of new features

**0.4.0**  
- Added `post_hoc` tests to `Lmer` models
- Added `simulate` data from fitted `Lmer` models
- Numerous bug fixes for python 3 compatibility

**0.3.2**
- Addition of `simulate` module

**0.2.2**
- **Pypi release**
- Better versioning system

**0.2.1**
- Support for standard linear regression models
- Models include support for robust standard errors, boot-strapped CIs, and permuted inference

**0.2.0**
- Support for categorical predictors, model predictions, and plotting

**0.1.0**
- Linear and Logit multi-level models
