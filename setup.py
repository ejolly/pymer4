from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

with open("pymer4/version.py") as f:
    exec(f.read())

extra_setuptools_args = dict(
    tests_require=['pytest']
)

setup(
    name ='pymer4',
    version = __version__,
    author ='Eshin Jolly',
    author_email ='eshin.jolly.GR@dartmouth.edu',
    url = 'http://eshinjolly.com/pymer4/',
    install_requires = requirements,
    package_data = {'pymer4':['resources/*']},
    packages = find_packages(exclude=['pymer4/tests']),
    license = 'MIT',
    description='pymer4: all the convenience of lme4 in python',
    long_description= "pymer4 is a Python package to make it simple to perform multi-level modeling by interfacing with the popular R package lme4. pymer4 is also capable of fitting a variety of standard regression models with robust, bootstrapped, and permuted estimators",
    keywords = ['statistics','multi-level-modeling','regression','analysis'],
    classifiers = [
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3.6",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License"
    ],
    **extra_setuptools_args
)
