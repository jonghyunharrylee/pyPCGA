"""
The pyPCGA package enables you to run matrix-free geostatistical inversion
"""
__name__ = 'pyPCGA'
__author__ = 'Jonghyun Harry Lee'

# import
from .pcga import PCGA
from . import covariance

#!/usr/bin/env python
#import os, sys
#dir = os.path.dirname(os.path.realpath(__file__))
#for f in os.listdir(dir):
#    if f.startswith('py') and os.path.isdir(os.path.join(dir,f)):
#        try:
#            exec('from .%s import %s' %(f,f.strip('py')))
#            __all__.extend(sys.modules['pyPCGA.'+f].__all__)
#        except Exception as e:
#            continue
