#!/usr/bin/env python
"""pyPCGA : Prinicipal Component Geostatistical Approach
PCGA description wil be provided later
"""
from setuptools import setup, find_packages

with open('README.md') as readme_file:
    readme = readme_file.read()

with open('CHANGELOG.rst') as history_file:
    history = history_file.read()

setup(name='pyPCGA',
      description='pyPCGA is a Python package to run Principal Component Geostatistical Approach',
      long_description=readme + '\n\n' + history,
      author='Jonghyun Harry Lee',
      author_email='jonghyun.harry.lee@hawaii.edu',
      url='https://github.com/jonghyunharrylee/pyPCGA/',
      license='New BSD',
      install_requires=['numpy>=1.9.0', 'scipy>=0.18'],
      platforms='Windows, Mac OS-X, Linux',
      packages=find_packages(include=['pyPCGA',
                                      'pyPCGA.*']),
      include_package_data=True,
      version='0.1.0')

