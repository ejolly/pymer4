---
name: Bug report
about: Please raise issues using the following guidelines
title: 'Issue with...'
labels: bug
assignees: ''

---

# Describe the issue
A clear and concise description of what the bug is, specifically:
- What you expected to happen and what actually happened.
- Any thing you tried to solve the issue
- Steps to reproduce it

## If you installed using Anaconda 
- Does installing in a clean environment solve the issue?
  - `conda create --name pymer4 -c ejolly -c defaults -c conda-forge pymer4`
- Does using conda forge for all requirement solve the issue?
  - `conda install -c ejolly -c conda-forge pymer4`
- Does using a development version of `pymer4` solve the issue?
  - `conda install -c ejolly/label/pre-release -c conda-forge pymer4`
### Please provide the following
- The install command you used (e.g. `conda install ...`)
- Other packages in the environment you installed `pymer4` (i.e. the output of `conda env list`)

## If you installed using pip
Debugging pip installation issues is unfortunately infeasible because of how finicky `rpy2` can be on various platforms. For this reason please try [installing via Anaconda](http://eshinjolly.com/pymer4/installation.html#using-anaconda-recommended) to see if that solves your problem. 

If you raise an issue without first trying a conda install it's highly likely that your issue will not be solved and will eventually be closed. 

### Please provide the following
- The install command you used (e.g. `pip install ...`)
- Other packges in your pip "environment" (i.e. the output of `pip freeze`)

# Please provide the following additional info

 - OS: [e.g. Mac OS 10.15.6]
 - Python version
