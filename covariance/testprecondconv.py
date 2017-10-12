import numpy as np
from covariance import CovarianceMatrix

from time import time
from scipy.sparse.linalg import minres, gmres, aslinearoperator, LinearOperator

import sys
nu = sys.argv[1]
k = int(sys.argv[2])




if nu == '12':
	def kernel(R): return np.exp(-R)
elif nu == '32':
	def kernel(R): return (1+np.sqrt(3)*R)*np.exp(-np.sqrt(3)*R)
elif nu == '52':
	def kernel(R): return (1+np.sqrt(5)*R + 5.*R/3.)*np.exp(-np.sqrt(5)*R)



class Residual:
        def __init__(self):
                self.res = []

        def __call__(self, rk):
                self.res.append(rk)

        def itercount(self):
                return len(self.res)

        def clear(self):
                self.res = []



for n in [128, 256, 512, 1024]:

	N = np.array([n,n])
	Q = CovarianceMatrix(method = 'FFT', pts = None, kernel = kernel, xmin = np.zeros(2), xmax = np.ones(2), N = N, theta = np.ones(2), nugget = 1.e-4)

	Qop = aslinearoperator(Q)


	start = time()
	Q.BuildPreconditioner(k = k)
	print "Time for building preconditioner is %g" %(time()-start)


	b = np.random.randn(n*n,1)


	callback = Residual()
	x, info = gmres(Qop, b, tol = 1.e-10, maxiter = 1000, callback = callback, M = Q.P)
	print "Number of iterations are %g" %(callback.itercount())
