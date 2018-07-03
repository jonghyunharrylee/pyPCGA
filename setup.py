#!/usr/bin/env python
"""pyPCGA : Prinicipal Component Geostatistical Approach
PCGA description wil be provided later
"""
import os
import sys
from setuptools import setup
from pyPCGA import __name__, __author__

if sys.argv[-1].endswith('setup.py'):
    print('\nTo install, run "python setup.py install"\n')
    sys.exit(-1)

setup(name=__name__,
      description='pyPCGA is a Python package to run Principal Component Geostatistical Approach',
      long_description='',
      author=__author__,
      author_email='jonghyun.harry.lee@hawaii.edu',
      url='https://github.com/jonghyunharrylee/pyPCGA/',
      license='New BSD',
      install_requires=['numpy>=1.9.0','scipy>=0.18'],
      platforms='Windows, Mac OS-X, Linux',
      packages=['pyPCGA','pyPCGA.covariance'],
      version='0.1.0')

