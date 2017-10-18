# pymer4

Love mixed-modeling using lme4 in R but prefer to work in the scientific Python ecosystem? Then this package has got you covered! It's a small convenience package wrapping the basic functionality of [lme4](https://github.com/lme4/lme4)\*.  
\* *Currently this only includes linear and logit models*

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
seaborn>=0.8.0
matplotlib>=2.0
rpy2==2.8.5

# R
lme4>=1.1.12
lmerTest>=2.0.33
```
