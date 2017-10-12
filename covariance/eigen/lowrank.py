import numpy as np
from randomized import *
from orth import *
from scipy.linalg import qr
from scipy.sparse.linalg import LinearOperator, aslinearoperator
__all__ = ['LowRankGHEP', 'LowRankAddition']



def matmat(A, X, transp = False):
	"""
		Returns LinearOperator * Dense Tall-skinny matrix
	"""
	Y = np.zeros_like(X)
	for i in np.arange(X.shape[1]):
		if not transp:
			Y[:,i] = A.matvec(X[:,i]) 
		else:
			Y[:,i] = A.rmatvec(X[:,i])

	return Y


def LowRankConversion(W,B):
	"""
	Convert low rank matrix WW^T = UDU^T, where U^TBU = I
	Returns U,D

	if inverse = True, assumes that B^{-1} is provided instead of B but still 
	computes U^TBU = I
	"""
	
	from scipy.linalg import svd
	

	w, _, r = mgs_stable(B,W, verbose = False)
	u,s,_ = svd(r, full_matrices = False, compute_uv = True)

	v = np.dot(w,u)

	return  s**2., v


def LowRankGHEP(U, d1, V, d2, B, inverse = False, tol = 1.e-10):
       	"""
       	A = UD_1U^T + VD_2V^T = WDW^T
       	Truncates the singular values smaller than tol
       	Returns W,D
       	"""	  	
       	       
	if inverse:
		matvec = lambda x: np.dot(U,np.dot(np.diag(d1), np.dot(U.T,x))) + np.dot(V,np.dot(np.diag(d2), np.dot(V.T,x)))
		A = LinearOperator(shape = B.shape, matvec = matvec, dtype = 'd')
	
		k = U.shape[1] + V.shape[1]
		d, w = RandomizedGHEP2(A, B, k, p = 0, error = False)
	else:
		U = np.dot(U,np.diag(np.sqrt(d1)))
       		V = np.dot(V,np.diag(np.sqrt(d2)))
       		W = np.hstack((U,V))
		
		d, w = LowRankConversion(W, B)


       	ind = np.flatnonzero(np.abs(d)/np.max(d) > tol)						
        
       	return  d[ind], w[:,ind]

class Inverse:
	def __init__(self, A):
		self.A = A
	def matvec(self,x):
		return self.A.solve(x)
	def solve(self,x):
		return self.A.matvec(x)
	

def LowRankAddition(U, d1, V, d2, B, BV = None, inverse = False, tol = 1.e-10):


	#Perform Block MGS
	BV = matmat(B,V) if BV == None else BV
	Vh = V - np.dot(U,np.dot(U.T,BV))
	Vh, _, _ = mgs_stable(B,Vh)	
	W = np.hstack((U,Vh))	#Orthonormal basis for [U,V]

		

	WU = np.vstack((np.eye(U.shape[1]),np.zeros((V.shape[1],U.shape[1]),dtype='d')))
	WV = np.vstack((np.dot(U.T,BV),np.dot(Vh.T,BV)))


	M = np.dot(WU, np.dot(np.diag(d1), WU.T))+ np.dot(WV, np.dot(np.diag(d2), WV.T))

	d,v = np.linalg.eigh(M)
	ind = np.flatnonzero(np.abs(d)/np.max(d) > tol)
		

	print "Rank  after low rank addition is %i and before it was %i" %(ind.size,d1.size+d2.size)
	return d[ind], np.dot(W,v[:,ind]) 	
