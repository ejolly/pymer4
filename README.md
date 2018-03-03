[![Build Status](https://travis-ci.org/ejolly/pymer4.svg?branch=master)](https://travis-ci.org/ejolly/pymer4)
[![Package versioning](https://img.shields.io/pypi/v/pymer4.svg)](https://pypi.python.org/pypi?name=pymer4&version=0.2.2&:action=display)
[![Python Versions](https://img.shields.io/pypi/pyversions/pymer4.svg)](https://pypi.python.org/pypi?name=pymer4&version=0.2.2&:action=display)

# pymer4

Love multi-level-modeling using lme4 in R but prefer to work in the scientific Python ecosystem? Then this package has got you covered! It's a small convenience package wrapping the basic functionality of [lme4](https://github.com/lme4/lme4)\*.  
\* *Currently this only includes linear and logit models*

This package can also estimate standard, robust, and permuted regression models\*  
\* *Currently this only includes linear models*

## Documentation
Current documentation and usage examples can be found **[here](http://eshinjolly.com/pymer4/)**.

## Installation

#### Requirements <a name="requirements"></a>
You need *both* Python (2.7 or 3.6) and R (>= 3.2.4) on your system to use this package in addition to the following R packages (*pymer4 will NOT install R or R packages for you!*):
```
lme4>=1.1.12
lmerTest>=2.0.33
lsmeans>=2.25
```

#### Instructions

1. Method (stable)

    ```
    pip install pymer4
    ```

2. Method 2 (latest)

    ```
    pip install git+https://github.com/ejolly/pymer4
    ```

#### Installation issues

Some users have issues installing `pymer4` on recent versions of macOS. This is due to compiler issues that give `rpy2` (a package dependency of `pymer4`) some issues during install. Here's a fix that should work for that:

1. Install [homebrew](https://brew.sh/) if you don't have it already, by running the command at the link (it's a great pacakage manager for macOS). To check if you already have it, do `which brew` in your Terminal. If nothing pops up you don't have it.
2. Fix brew permissions: `sudo chown -R $(whoami) $(brew --prefix)/*` (this is **necessary** on macOS Sierra or higher (>= macOS 10.12))
3. Update homebrew `brew update`
4. Install an updated compiler: `brew install gcc`, or if you have homebrew already, `brew upgrade gcc`
5. Enable the new compiler for use:
    ```
    export CC="$(find `brew info gcc | grep usr | sed 's/(.*//' | awk '{printf $1"/bin"}'` -name 'x86*gcc-7')"
    export CFLAGS="-W"
    ```
6. If this doesn't work for you might need to manually find out where the new compiler is installed. To do so use `brew info gcc` and `cd` into the directory that begins with `/usr` in the output of that command. From there `cd` into `bin` and look for a file that begins with `x86` and ends with `gcc-7`. Copy the *full path* to that file and run the following:
    ```
    export CC= pathYouCopiedInQuotes
    export CFLAGS="-W"
    ```
7. Finally install `rpy2` using the new compiler you just installed: `pip install rpy2==2.8.6`
8. Now you should be able to `pip install pymer4`:)

#### Change-log
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
