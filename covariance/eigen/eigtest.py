
import numpy as np
from scipy.sparse import csr_matrix, csc_matrix, rand, issparse
from scipy.sparse.linalg import factorized, spsolve, splu
from randomized import RandomizedGHEP, RandomizedHEP

class MyMatrix:
        def __init__(self, M, solve = False):
        
		assert issparse(M)
	        self.M = M if isinstance(M,csc_matrix) else csc_matrix(M)
	
                n = M.shape[0]
                self.shape = (n,n)
                self.dtype = 'd'

		if solve:
			self.factor = splu(M) 
	
                self.mvcount = 0
                self.scount  = 0
        def matvec(self, x):
                self.mvcount += 1
                return self.M*x

        def solve(self, x):
                self.scount  += 1
                return self.factor.solve(x)

        def clear(self):
                self.mvcount = 0
                self.scount  = 0
                return


class MyMatrixInverse(MyMatrix):
        def __init__(self, M, solve = False):
		MyMatrix.__init__(self, M,solve)
        
        def matvec(self, x):
                self.mvcount += 1
                return self.factor.solve(x)

        def solve(self, x):
                self.scount  += 1
                return self.M*x


def comparespectrum(Bname, k):
	mat = mmread(Bname)
	mat = csc_matrix(mat)	

	B = MyMatrix(mat, solve = True)

	n = B.shape[0]
	
	from scipy.sparse import diags
	A = MyMatrix(diags(0.95**np.arange(n), 0, format = "csc", shape = (n,n)) ) 
	Aop = aslinearoperator(A)
	

	print "Size of A is ", A.shape

	start = time()
	lg, v = RandomizedGHEP(Aop, B, k = k, twopass = True, error = True) 
	print "Time for Randomized is %g" %(time()-start)
	print "Ax %d, Bx %d, B^{-1}x %d"%(A.mvcount, B.mvcount, B.scount)

	A.clear()
	B.clear()

	Binv = MyMatrixInverse(mat,solve = True)
	start = time()
	l, v = eigsh(Aop, M = aslinearoperator(B), which = 'LM', k = k, tol = 1.e-15, Minv = aslinearoperator(Binv)) 
	l =  l[::-1]
	print "Time for eigsh is %g" %(time()-start)
	print "Ax %d, Bx %d, B^{-1}x %d"%(A.mvcount, B.mvcount, Binv.mvcount)
                   

	return np.max(np.abs(l-lg)/np.abs(l)), np.max(np.abs(l-lg))
	

if __name__ == '__main__':
	from time import time
	import sys
	from scipy.sparse.linalg import eigsh, LinearOperator, aslinearoperator
	from scipy.io import loadmat, mmread

	k = 25
	
	errr, erra = [], []

	Bname = ['Chem97ZtZ', 'Trefethen_2000', 'plbuckle', 'nasa1824', '1138_bus', 't2dal_e', 'bcsstk12', 'ex3', 'mhd3200b' ]


	Bname = ['mat/' + item + '.mtx' for item in Bname]

	for name in Bname:
		errr_, erra_ = comparespectrum(name, k)

		errr.append(errr_)
		erra.append(erra_)	

	
		
