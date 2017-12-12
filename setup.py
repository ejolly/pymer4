from setuptools import setup

__version__ = '0.0.3'

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
name='pymer4',
version= __version__,
author='Eshin Jolly',
author_email='eshin.jolly.GR@dartmouth.edu',
install_requires=requirements,
package_data={'pymer4':['resources/*']},
packages=['pymer4'],
license='LICENSE.txt',
description='pymer4: all the convenience of lme4 in python'
)
