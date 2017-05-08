# pymer4

This is a small convenience package wrapping much of the basic functionality of [lme4](https://github.com/lme4/lme4). It's main purpose is to provide an interface that hides the back-and-forth code required when analyzing data in R and Python simultaneously. In other words a user can write completely in Python, never having to deal with R, but get all the mixed-effects model goodness and computation of lme4. All inferential statistics are computed using Satterthwaite approximation via the [lmerTest](https://cran.r-project.org/web/packages/lmerTest/index.html) package.

Behind the scenes this package simply uses [rpy2](https://rpy2.readthedocs.io/en/version_2.8.x/) to pass objects to R from Python, compute what's needed, parse and convert them into native Python types (e.g. pandas dataframes).

### Requirements:
You need *both* Python and R on your system to use this package. In addition to the following Python and R packages:
```
pandas>=0.19.1
numpy>=1.12.0
rpy2>=2.8.5
lme4>=1.1.12
lmerTest>=2.0.33
```

Example usage:
```
import pandas as pd
from pymer4.models import Lmer

df = pandas.read_csv('mydata.csv')

#Initialize a model object using plain old R-style formulae
#Because this is just running R behind the scenes, any model specification that lme4 can handle will work here
model = Lmer('DV ~ IV1 (IV1 | Subject)',data=df)

#Fit it and voila! R-style summary statistics
model.fit(method = 'Wald')
```

![](/misc/output.png)


This table is actually a pandas dataframe stored in model.coefs. In fact the model object also contains a bunch of other goodies such as:
model.AIC, model.vcov_ranef... which are also native python types (e.g. numpy arrays, bools, strings, etc). See the docstring for more info.

Finally, for convenience the model object also modifies its own dataframe (stored in the model.data attribute) to append a column of model residuals and fits.
