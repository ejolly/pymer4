from setuptools import setup

__version__ = '0.0.1'

setup(
name='pymer4',
version= __version__,
author='Eshin Jolly',
author_email='eshin.jolly.GR@dartmouth.edu',
install_requires=['pandas>=0.19.1','numpy>=1.12.0','rpy2>=2.8.5'],
package_data={'pymer4':['resources/*']},
packages=['pymer4'],
license='LICENSE.txt',
description='pymer4: all the convenience of lme4 in python'
)
