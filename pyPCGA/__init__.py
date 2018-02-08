#!/usr/bin/env python
import os, sys
from .pcga import PCGA
__all__ = ['PCGA']

dir = os.path.dirname(os.path.realpath(__file__))
for f in os.listdir(dir):
    if f.startswith('py') and os.path.isdir(os.path.join(dir,f)):
        try:
            exec('from .%s import %s' %(f,f.strip('py')))
            __all__.extend(sys.modules['pyPCGA.'+f].__all__)
        except Exception as e:
            continue
