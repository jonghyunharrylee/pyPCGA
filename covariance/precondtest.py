from dolfin import *
from covariance import CovarianceMatrix
import numpy as np

class Residual:
        def __init__(self):
                self.res = []

        def __call__(self, rk):
                self.res.append(rk)

        def itercount(self):
                return len(self.res)

        def clear(self):
                self.res = []



def LaplacePreconditioner(mesh, V):
	u = TrialFunction(V)
	v = TestFunction(V)

	a = inner(grad(u),grad(v))*dx
	L = Constant(0.0)*v*dx

	bc = None

	M = uBLASSparseMatrix()
        assemble(a, tensor = M)


        #Convert into 
        (row, col, data) = M.data()   # get sparse data
        col = np.intc(col)
        row = np.intc(row)
        n = M.size(0)

        import scipy.sparse as sp
        M = sp.csc_matrix( (data,col,row), shape=(n,n), dtype='d')

	return M


if __name__ == '__main__':
	import sys
	n = 500
	if len(sys.argv) > 1:
		n = int(sys.argv[1])

	N = np.array([n+1,n+1])

	mesh = UnitSquare(n,n)
	V = FunctionSpace(mesh,"CG",1)
	
	A = LaplacePreconditioner(mesh,V)

	def kernel(r):	return np.exp(-r)

	#def kernel(r): 	return (1+np.sqrt(3)*r)*np.exp(-np.sqrt(3)*r)
	#def kernel(r):
	#	d = np.sqrt(5)*r
	#	return (1+d+d**2./3.)*np.exp(-d)

	from time import time
	start = time()
	#Q = CovarianceMatrix('Hmatrix',mesh.coordinates(), kernel)
	Q = CovarianceMatrix(method = 'FFT', pts = mesh.coordinates(), kernel = kernel, xmin = np.zeros(2), xmax = np.ones(2), N = N, theta = np.ones(2))
	print "Covariance matrix setup", time()-start
	

	from scipy.sparse.linalg import minres, gmres, aslinearoperator, LinearOperator
	Qop = aslinearoperator(Q)

	nn = mesh.coordinates().shape[0]
	b = np.random.randn(nn,1)

	Aop = LinearOperator(A.shape, matvec = lambda x: A*x, dtype = 'd')

	callback = Residual()
	x, info = minres(Qop, b, tol = 1.e-10, callback = callback, M = Aop)

	print callback.itercount()
