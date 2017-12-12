[![Build Status](https://travis-ci.org/ejolly/pymer4.svg?branch=master)](https://travis-ci.org/ejolly/pymer4)
# pymer4

Love multi-level-modeling using lme4 in R but prefer to work in the scientific Python ecosystem? Then this package has got you covered! It's a small convenience package wrapping the basic functionality of [lme4](https://github.com/lme4/lme4)\*.  
\* *Currently this only includes linear and logit models*

This package can also estimate standard, robust, and permuted regression models\*  
\* *Currently this only includes linear models*

#### Documentation
Current documentation and usage examples can be found [here](http://eshinjolly.com/pymer4/).

#### Installation

```
pip install git+https://github.com/ejolly/pymer4
```

#### Requirements <a name="requirements"></a>
You need *both* Python and R on your system to use this package. In addition to the following Python and R packages:
```
# Python
pandas>=0.19.1
numpy>=1.12.0
rpy2==2.8.5
seaborn>=0.8.0
matplotlib>=2.0
patsy>=0.4.1
joblib>=0.11
scipy>=1.0.0


# R
lme4>=1.1.12
lmerTest>=2.0.33
```

#### Change-log
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
