#!/usr/bin/env python
"""pyPCGA : Prinicipal Component Geostatistical Approach

PCGA description


"""
from __future__ import print_function
import os
import sys
import numpy
# BEFORE importing setuptools, remove MANIFEST.
# Otherwise it may not be properly updated
# when the contents of directories change
# (true for distutils, not sure about setuptools).
if os.path.exists('MANIFEST'):
    os.remove('MANIFEST')

if sys.argv[-1].endswith('setup.py'):
    print('\nTo install, run "python setup.py install"\n')
    sys.exit(-1)

try:
    import numpy
    if int(numpy.__version__.split('.')[0]) < 1:
        print(('pyOpt requires NumPy version 1.0 or later (%s detected).' %numpy.__version__))
        sys.exit(-1)
except ImportError:
    print('NumPy version 1.0 or later must be installed to build pyPCGA')
    sys.exit(-1)

PROJECT_ROOT = os.path.dirname(os.path.realpath(__file__))
REQUIREMENTS_FILE = os.path.join(PROJECT_ROOT, 'requirements.txt')

with open(REQUIREMENTS_FILE) as f:
    install_reqs = f.read().splitlines()

'''
def configuration(parent_package=None, top_path=None):
    from numpy.distutils.misc_util import Configuration

    config = Configuration(None, parent_package, top_path)
    config.set_options(ignore_setup_xxx_py=True,
                       assume_default_configuration=True,
                       delegate_options_to_subpackages=True,
                       quiet=True)
    config.add_subpackage('PCGA')
    return config
'''

if __name__ == '__main__':
    #from numpy.distutils.core import setup
    #from setuptools import setup, find_packages
    from distutils.core import setup
    setup(
        name = 'pyPCGA',
        version = '0.1.0',
        author = 'Jonghyun Harry Lee, Peter K. Kitanidis, Eric Darve, Hojat, Matthew, Ty',
        author_email = 'jonghyun.harry.lee@hawaii.edu',
        maintainer = 'Jonghyun Harry Lee, Matthew, Ty, Hojat',
        maintainer_email ='jonghyun.harry.lee@hawaii.edu',
        url = 'https://github.com/jonghyunharrylee/pyPCGA',
        download_url ='https://github.com/jonghyunharrylee/pyPCGA',
        description = 'PCGA',
        long_description = 'Principal Component Geostatistical Approach',
        keywords = ['inverse modeling', 'UQ'],
        license = 'TBD',
        platforms = ['Windows', 'Linux', 'Solaris', 'Mac OS-X', 'Unix'],
        packages=['pyPCGA','pyPCGA.covariance'],
        #packages=find_packages(),
        #install_requires=install_reqs,
        #zip_safe=True,
    )
