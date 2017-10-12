#Add directory ../ - Temporary hack
import os as _os
import sys as _sys

parent = _os.path.abspath('../')
if parent not in _sys.path:
        _sys.path.append(parent)
del _sys, _os


from covariance import *
from eigen import RandomizedHEP, Nystrom
from scipy.sparse.linalg import eigsh

if __name__ == '__main__':
	import numpy as np
	n = 1000
	pts = np.random.rand(n,2)
	def kernel(R):	return np.exp(-R/2.)
	
	Q = CovarianceMatrix('Hmatrix',pts,kernel,verbose = False)

	k = 20
	
	from matplotlib import pyplot as plt
	plt.close('all')
	plt.figure()

	
	l, v = Nystrom(Q, k = k)
	plt.semilogy(l, 'm+-', label = 'Nystrom')
	print l

	l, v = RandomizedHEP(Q, k = k, twopass = False)
	plt.semilogy(l, 'ko-', label = 'Single pass')
	print l
	
	l, v = RandomizedHEP(Q, k = k, twopass = True)
	plt.semilogy(l, 'rx-', label = 'Double pass')
	print l

	l, v = eigsh(Q, k = k, which = 'LM')
	plt.semilogy(l[::-1], 'gs-', label = 'Arpack')
	print l[::-1]

	plt.legend()
	plt.show()	
