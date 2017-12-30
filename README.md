[![Build Status](https://travis-ci.org/ejolly/pymer4.svg?branch=master)](https://travis-ci.org/ejolly/pymer4)
[![Package versioning](https://img.shields.io/pypi/v/pymer4.svg)](https://pypi.python.org/pypi?name=pymer4&version=0.2.2&:action=display)
[![Python Versions](https://img.shields.io/pypi/pyversions/pymer4.svg)](https://pypi.python.org/pypi?name=pymer4&version=0.2.2&:action=display)

# pymer4

Love multi-level-modeling using lme4 in R but prefer to work in the scientific Python ecosystem? Then this package has got you covered! It's a small convenience package wrapping the basic functionality of [lme4](https://github.com/lme4/lme4)\*.  
\* *Currently this only includes linear and logit models*

This package can also estimate standard, robust, and permuted regression models\*  
\* *Currently this only includes linear models*

#### Documentation
Current documentation and usage examples can be found **[here](http://eshinjolly.com/pymer4/)**.

#### Requirements <a name="requirements"></a>
You need *both* Python (2.7 or 3.6) and R (>= 3.2.4) on your system to use this package in addition to the following R packages (*pymer4 will NOT install R or R packages for you!*):
```
lme4>=1.1.12
lmerTest>=2.0.33
```

#### Installation  

1. Method (stable)

    ```
    pip install pymer4
    ```

2. Method 2 (latest)

    ```
    pip install git+https://github.com/ejolly/pymer4
    ```


#### Change-log
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
