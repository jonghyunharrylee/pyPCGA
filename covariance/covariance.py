import numpy as np
# no explicit exception model 
#from exceptions import NotImplementedError, KeyError 

class Residual:
    def __init__(self):
        self.res = []

    def __call__(self, rk):
        self.res.append(rk)

    def itercount(self):
        return len(self.res)

    def clear(self):
        self.res = []

class CovarianceMatrix:
	def __init__(self, method, pts, kernel, nugget = 0.0, **kwargs):
		self.method = method
		self.kernel = kernel
		self.pts    = pts
	
			
		try:
			verbose = kwargs['verbose']
		except KeyError:
			verbose = False
		self.verbose = verbose

		if method == 'Dense':
		        from dense import GenerateDenseMatrix
			self.mat = GenerateDenseMatrix(pts, kernel)
			
			self.pts = pts
	
		
		elif method == 'FFT':
		        from toeplitz import CreateRow, ToeplitzProduct
			xmin 	= kwargs['xmin']
			xmax 	= kwargs['xmax']	
			N    	= kwargs['N']
			theta	= kwargs['theta']		
			
			self.N      = N
			self.row, pts  = CreateRow(xmin,xmax,N,kernel,theta)		
		
			self.pts    = pts
	
		elif method == 'Hmatrix':
		        
                        from hmatrix import Hmatrix	
			n   	= np.size(pts,0)
			ind 	= np.arange(n)
			rkmax 	= 32 if 'rkmax' not in kwargs else kwargs['rkmax']
			self.H 	= Hmatrix(pts, kernel, ind, verbose = verbose, rkmax = rkmax)	
			
		else:
			raise NotImplementedError

		self.P		= None
		n = pts.shape[0]
		self.shape 	= (n,n)
		self.nugget	= nugget
		self.dtype      = 'd'
		self.count = 0

		self.solvmatvecs = 0

	def matvec(self, x):
		method = self.method
		if method == 'Dense':
			y = np.dot(self.mat,x)

		elif method == 'FFT':
		        from toeplitz import CreateRow, ToeplitzProduct
			y = ToeplitzProduct(x, self.row, self.N)

		elif method == 'Hmatrix':
			y = np.zeros_like(x, dtype = 'd')
			self.H.mult(x, y, self.verbose)
		
		y += self.nugget*y	


		self.count += 1
		
		return y

	def rmatvec(self, x):
		method = self.method
		if method == 'Dense':
			y = np.dot(self.mat.T,x)

		elif method == 'FFT':
			y = ToeplitzProduct(x, self.row, self.N)

		elif method == 'Hmatrix':
			y = np.zeros_like(x, dtype = 'd')
			self.H.transpmult(x, y, self.verbose)
		
		y += self.nugget*y	
		self.count += 1
		
		return y

	def reset(self):
		self.count = 0
		self.solvmatvecs = 0

	def itercount(self):
		return self.count

	def BuildPreconditioner(self, k = 100, view = False):
		from time import time
		
		from scipy.spatial import cKDTree
		from scipy.spatial.distance import pdist, cdist
		
		from scipy.linalg import solve
		from scipy.sparse import csr_matrix	
	
		if self.P != None:
			return
	
	
		pts = self.pts
		kernel = self.kernel
	
		N = pts.shape[0]

	        #Build the tree
       	 	start = time()
        	tree = cKDTree(pts, leafsize = 32)
        	end = time()

		if self.verbose:
        		print "Tree building time = %g" % (end-start)

        	#Find the nearest neighbors of all the points
        	start = time()
        	dist, ind = tree.query(pts,k = k)
        	end = time()
		if self.verbose:
	        	print "Nearest neighbor computation time = %g" % (end-start)

        	Q = np.zeros((k,k),dtype='d')
        	y = np.zeros((k,1),dtype='d')

        	row = np.tile(np.arange(N), (k,1)).transpose()
       		col = np.copy(ind)
        	nu = np.zeros((N,k),dtype='d')

        	y[0] = 1.
        	start = time()
        	for i in np.arange(N):
                	Q = kernel(cdist(pts[ind[i,:],:],pts[ind[i,:],:]))
                	nui = np.linalg.solve(Q,y)
                	nu[i,:] = np.copy(nui.transpose())
        	end = time()
		
		if self.verbose:
       			print "Elapsed time = %g" % (end-start)
	
		ij = np.zeros((N*k,2), dtype = 'i')
        	ij[:,0] = np.copy(np.reshape(row,N*k,order='F').transpose() )
        	ij[:,1] = np.copy(np.reshape(col,N*k,order='F').transpose() )

        	data = np.copy(np.reshape(nu,N*k,order='F').transpose())
        	self.P = csr_matrix((data,ij.transpose()),shape=(N,N), dtype = 'd')

		if view == True:
			from matplotlib import pyplot as plt
			plt.spy(self.P,markersize = 0.05)
			print float(self.P.getnnz())/N**2.
			plt.savefig('sp.eps')
	
	def solve(self, b, maxiter = 1000, tol = 1.e-10):
				
		if self.method == 'Dense':
			from scipy.linalg import solve
			return solve(self.mat, b)

		else:
			from scipy.sparse.linalg import gmres, aslinearoperator, minres

			P = self.P
			Aop = aslinearoperator(self)
		
			residual = Residual()
			if P != None:
				x, info = gmres(Aop, b, tol = tol, restart = 30, maxiter = maxiter, callback = residual, M = P)
			else:
				x, info = minres(Aop, b, tol = tol, maxiter = maxiter, callback = residual )
			self.solvmatvecs += residual.itercount()
			if self.verbose:	
				print "Number of iterations is %g and status is %g"% (residual.itercount(), info)
		return x

	def realizations(self):

		import os as _os
		import sys as _sys


		parent = _os.path.abspath('../') 
		if parent not in _sys.path:
			_sys.path.append(parent)
		del _sys, _os


		from fAb import sqrtm
		b = np.random.randn(self.shape[0], 1)
		if self.method == 'Dense':	
			eigen = np.linalg.eig(self.mat)
			y = sqrtm(self.mat, b, eps = 1.e-6, precond = None, restol = None)
		else:
			from scipy.sparse.linalg import eigsh, LinearOperator
			Qop = LinearOperator(self.shape, lambda x : self.matvec(x), dtype = 'd')
			m = self.nugget
			M,_ = eigsh(Qop, k = 1, which = 'LM')
			eigen = np.array([m,M])

			print m, M 			
			if self.P == None:
				self.BuildPreconditioner()
			
			P = self.P
			y = sqrtm(Qop, b, eigen, eps = 1.e-6, precond = P, restol = 1.e-7)

			print np.dot(y.T,y), np.dot(b.T, Q.matvec(b))

		return y 
		 	
	
if __name__ == '__main__':
	n = 5000
	pts = np.random.rand(n, 2)

	def kernel(R):
		return np.exp(-R)

	Q = CovarianceMatrix('Hmatrix',pts,kernel,verbose = True, nugget = 1.e-4)	


	x = np.ones((n,), dtype = 'd')
	y = Q.matvec(x)


	Q.BuildPreconditioner(k = 30, view = False)
	Q.verbose = False
	xd = Q.solve(y)

	print np.linalg.norm(x-xd)/np.linalg.norm(x)
	y = Q.realizations()
	
