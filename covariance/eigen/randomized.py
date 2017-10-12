"""Randomized eigenvalue calculation"""

from scipy.sparse.linalg import LinearOperator, aslinearoperator 
from scipy.sparse import identity
from scipy.linalg import qr, eig, eigh 

import numpy as np

from time import time
from orth import mgs, mgs_stable, cholqr, precholqr 

__all__ = ['RandomizedHEP', 'Nystrom', 'RandomizedSVD', 'RandomizedGSVD', 'RandomizedGHEP', 'Nystrom_GHEP', 'RandomizedGHEP2']

def RandomizedHEP(A, k, p = 20, twopass = False):
	"""Randomized algorithm for Hermitian eigenvalue problems
	
	Parameters:
	
	
	A 	= LinearOperator n x n
			hermitian matrix operator whose eigenvalues need to be estimated
	k	= int, 
			number of eigenvalues/vectors to be estimated
	twopass = bool, 
			determines if matrix-vector product is to be performed twice
	
	Returns:
	
	w	= double, k
			eigenvalues
	u 	= n x k 
			eigenvectors
	
	"""


	#Get matrix sizes
	m, n = A.shape
	
	#For square matrices only
	assert m == n

	#Oversample
	k = k + p 

	#Generate gaussian random matrix 
	Omega = np.random.randn(n,k)
	
	Y = np.zeros((m,k), dtype = 'd')
	for i in np.arange(k):
		Y[:,i] = A.matvec(Omega[:,i])
	
	q,_ = qr(Y, mode = 'economic')

	if twopass == True:
		B = np.zeros((k,k),dtype = 'd')
		for i in np.arange(k):
			Aq = A.matvec(q[:,i])	
			for j in np.arange(k):
				B[i,j] = np.dot(q[:,j].transpose(),Aq)
			
	else:
		from scipy.linalg import inv, pinv,svd, pinv2
		#B = np.dot( np.dot(q.T, Y), inv(np.dot(q.T, Omega)) )
		

		temp  = np.dot(Omega.T, Y)
		temp2 = np.dot(q.T,Omega)
		temp3 = np.dot(q.T,Y)
		
		B = np.dot(pinv2(temp2.T), np.dot(temp, pinv2(temp2)))
		Binv = np.dot(pinv(temp3.T),np.dot(temp, pinv2(temp3)))	

		B = inv(Binv)
		
	#Eigen subproblem
	w, v = eigh(B)
	#Reverse eigenvalues in descending order
	w = w[::-1]
	#Compute eigenvectors		
	u = np.dot(q, v[:,::-1])	

	k = k - p
	return w[:k], u[:,:k]

def Nystrom(A, k, p = 20, twopass = False):
	"""Randomized algorithm for Hermitian eigenvalue problems
	
	Parameters:
	
	A 	= LinearOperator n x n
			hermitian matrix operator whose eigenvalues need to be estimated
	k	= int, 
			number of eigenvalues/vectors to be estimated
	twopass = bool, 
			determines if matrix-vector product is to be performed twice
	
	Returns:
	
	w	= double, k
			eigenvalues
	u 	= n x k 
			eigenvectors
	
	"""


	#Get matrix sizes
	m, n = A.shape
	
	#For square matrices only
	assert m == n

	#Oversample
	k = k + p 

	#Generate gaussian random matrix 
	Omega = np.random.randn(n,k)
	
	Y = np.zeros((m,k), dtype = 'd')
	for i in np.arange(k):
		Y[:,i] = A.matvec(Omega[:,i])
	
	q,_ = qr(Y, mode = 'economic')

	Aq = np.zeros((n,k), dtype = 'd')
	for i in np.arange(k):
		Aq[:,i] = A.matvec(q[:,i])	
			
	T = np.dot(q.T, Aq)

	from scipy.linalg import cholesky, svd, inv

	R = cholesky(inv(T), lower = True)
	B = np.dot(Aq, R)
	u, s, _ = svd(B) 
		

	k = k - p
	return s[:k]**2., u[:,:k]

def RandomizedSVD(A, k, p = 20):
	"""
	Randomized algorithm for Hermitian eigenvalue problems
	
	Parameters:
	
	A 	= LinearOperator n x n
			operator whose singular values need to be estimated
	k	= int, 
			number of eigenvalues/vectors to be estimated
	
	Returns:
	
	
	"""


	#Get matrix sizes
	m, n = A.shape
	
	#For square matrices only
	assert m == n

	#Oversample
	k = k + p 

	#Generate gaussian random matrix 
	Omega = np.random.randn(n,k)
	
	Y = np.zeros((m,k), dtype = 'd')
	for i in np.arange(k):
		Y[:,i] = A.matvec(Omega[:,i])
	
	q,_ = qr(Y, mode = 'economic')


	Atq = np.zeros((n,k), dtype = 'd')
	for i in np.arange(k):
		Atq[:,i] = A.rmatvec(q[:,i])

	from scipy.linalg import svd
	u, s, vt = svd(Atq.T, full_matrices = False)
	
	diff = Y - np.dot(q, np.dot(q.T,Y))	
	err = np.max(np.apply_along_axis(np.linalg.norm, 0, diff))

	from math import pi
	print "A posterior error is ", 10.*np.sqrt(2./pi)*err

	k = k - p
	return np.dot(q, u[:,:k]), s[:k], vt[:k,:].T


def RandomizedGSVD(A, S, T, k, p = 20, verbose = False):
	
	m, n = A.shape
	r = k + p
	
	Omega1 = np.random.randn(n,r)
	Omega2 = np.random.randn(m,r)

	Y1   	= np.zeros_like(Omega2, dtype = 'd')
	Y2   	= np.zeros_like(Omega1, dtype = 'd')


	for i in np.arange(r):
		Y1[:,i] = A.matvec(Omega1[:,i])
		Y2[:,i] = A.rmatvec(Omega2[:,i])	

	q1, Sq1, _ = mgs_stable(S, Y1, verbose = False)
	q2, Tq2, _ = mgs_stable(T, Y2, verbose = False)



	ATQ = np.zeros((m,r), dtype = 'd') 
	for i in np.arange(r):
		ATQ[:,i] = A.matvec(Tq2[:,i])
	
	mat = np.dot(Sq1.T,ATQ)
	ut,s,vth = np.linalg.svd(mat)	
	

	u = np.dot(q1,ut)[:,:k]
	s = s[:k]		
	vh = np.dot(q2,vth.T)[:,:k].T
	

	return u, s, vh



def RandomizedGHEP(A, B, k, p = 20, BinvA = None, orth = 'mgsr', twopass = True, verbose = False, error = True):
	"""
		Randomized algorithm for Generalized Hermitian Eigenvalue problem
		A approx (BU) * Lambda *(BU)^*

		Computes k largest eigenvalues and eigenvectors
		
		Modified from randomized algorithm for EV/SVD of A

	"""

	m, n = A.shape
	assert m == n	
	
	#Oversample
	k = k + p

	#Initialize quantities
	Omega 	= np.random.randn(n,k)
	Yh   	= np.zeros_like(Omega, dtype = 'd')
	Y   	= np.zeros_like(Omega, dtype = 'd')

	start = time()
	#Form matrix vector products with C = B^{-1}A
	if BinvA is None:
		for i in np.arange(k):
			Yh[:,i] = A.matvec(Omega[:,i])
			Y[:,i]  = B.solve(Yh[:,i])	
	else:
		for i in np.arange(k):
			Y[:,i]  = BinvA.matvec(Omega[:,i])


	matvectime = time()-start	
	if verbose:
		print "Matvec time in eigenvalue calculation is %g " %(matvectime) 

	#Compute Y = Q*R such that Q'*B*Q = I, R can be discarded
	start = time()
	if orth == 'mgs':
		q, Bq, _  = mgs(B, Y, verbose = False)
	elif orth == 'mgsr':
		q, Bq, _  = mgs_stable(B, Y, verbose = False)
	elif orth == 'cholqr':
		q, Bq, _  = cholqr(B, Y, verbose = False)
	elif orth == 'precholqr':
		q, Bq, _  = precholqr(B, Y, verbose = False)
	else:
		raise NotImplementedError
		
	Borthtime = time()-start
	
	if verbose:	
		print "B-orthonormalization time in eigenvalue calculation is %g " \
			%(Borthtime) 
	T = np.zeros((k,k), dtype = 'd')	

	
	start = time()
	
	if twopass == True:
		for i in np.arange(k):
			Aq = A.matvec(q[:,i])
			for j in np.arange(k):
				if not j < i:	T[i,j] = np.dot(Aq,q[:,j])

		T = T+T.T-np.diag(np.diag(T))
	else:
		if BinvA is not None:
			for i in np.arange(k):
				Yh[:,i] = B.matvec(Y[:,i])

		from scipy.linalg import inv
		OAO = np.dot(Omega.T, Yh)
		QtBO = np.dot(Bq.T, Omega)
		T = np.dot(inv(QtBO.T), np.dot(OAO, inv(QtBO)))

	eigcalctime = time()-start
	if verbose:	print "Calculating eigenvalues took %g" %(eigcalctime)
	if verbose:	print "Total time taken for Eigenvalue calculations is %g" % \
				(matvectime + Borthtime + eigcalctime)


	#Eigen subproblem
	w, v = eigh(T)

	#Reverse eigenvalues in descending order
	w = w[::-1]

	#Compute eigenvectors		
	u = np.dot(q, v[:,::-1])	
	k = k - p

	if error:
		#Compute error estimate
		r = 15 
		O = np.random.randn(n,r)
		err = np.zeros((r,), dtype = 'd')
		AO = np.zeros((n,r), dtype = 'd')
		BinvAO = np.zeros((n,r), dtype = 'd')
		for i in np.arange(r):
			AO[:,i]     = A.matvec(O[:,i])
			BinvAO[:,i] = B.solve(AO[:,i])
			diff =	BinvAO[:,i] - np.dot(q,np.dot(q.T,AO[:,i]))		
			err[i] = np.sqrt(np.dot(diff.T, B.matvec(diff)) )
	
		BinvNorm = np.max(np.apply_along_axis(np.linalg.norm, 0, q))
		
		alpha = 10.
	

		from math import pi
		print "Using r = %i and alpha = %g" %(r,alpha)
		print "Error in B-norm is %g" %\
				(alpha*np.sqrt(2./pi)*BinvNorm*np.max(err)/np.max(w))

	
	return w[:k], u[:,:k]


def Nystrom_GHEP(A,B, k, p = 20, orth = 'mgsr', verbose=False):

	m, n = A.shape
	assert m == n	
	
	#Oversample
	k = k + p

	#Initialize quantities
	Omega 	= np.random.randn(n,k)
	Yh   	= np.zeros_like(Omega, dtype = 'd')
	Y   	= np.zeros_like(Omega, dtype = 'd')

	for i in np.arange(k):
		Yh[:,i] = A.matvec(Omega[:,i])
		Y[:,i]  = B.solve(Yh[:,i])	


	if verbose:	print "Matvec time in eigenvalue calculation is %g " \
				%(matvectime) 

	#q, Bq, R  = mgs_stable(B, Y, verbose = False)
	start = time()
	if orth == 'mgs':
		q, Bq, _  = mgs(B, Y, verbose = False)
	elif orth == 'mgsr':
		q, Bq, _  = mgs_stable(B, Y, verbose = False)
	elif orth == 'cholqr':
		q, Bq, _  = cholqr(B, Y, verbose = False)
	elif orth == 'precholqr':
		q, Bq, _  = precholqr(B, Y, verbose = False)
	else:
		raise NotImplementedError
		
	Borthtime = time()-start
	
	if verbose:
		print "B-orthonormalization time in eigenvalue calculation is %g " \
			%(Borthtime) 


	T = np.zeros((k,k), dtype = 'd')
	AQ = np.zeros_like(q)

	for i in np.arange(k):
		Aq = A.matvec(q[:,i]);	AQ[:,i] = Aq;
		for j in np.arange(k):
			T[i,j] = np.dot(Aq,q[:,j])

	from scipy.linalg import cholesky, svd, inv, pinvh
	invT = pinvh(T)

	l, v = eigh(invT)
	ind = np.flatnonzero( l > 1.e-15)
	R = np.dot(np.dot(v[:,ind],np.diag(np.sqrt(l[ind]))),v[:,ind].T)

	M = np.dot(AQ, R)

	Binv = LinearOperator((n,n), matvec = lambda x: B.solve(x), dtype = 'd')	
	qt, Bt, rt = mgs_stable(Binv, M, verbose = False)
	u, s, _ = svd(rt)
	
	k -= p

	return s[:k]**2., np.dot(qt,u[:,:k])




def RandomizedGHEP2(A, B, k, p = 20, twopass = True, orth = 'mgsr', error = True, verbose = False, both = False):
	"""
		Randomized algorithm for Generalized Hermitian Eigenvalue problem
		A * U = B * U * Lambda

		Computes k largest eigenvalues and eigenvectors
		
		Modified from randomized algorithm for EV/SVD of A

	"""

	m,n = A.shape
	assert m == n	
	
	#Oversample
	k = k + p

	#Initialize quantities
	Omega 	= np.random.randn(n,k)
	Y   	= np.zeros_like(Omega, dtype = 'd')
	
	from time import time

	start = time()
	#Form matrix vector products with C = B^{-1}A
	for i in np.arange(k):
		Y[:,i] = A.matvec(Omega[:,i])

	matvectime = time()-start	
	if verbose:	print "Matvec time in eigenvalue calculation is %g " %(matvectime) 
	
	matvec = lambda x: B.solve(x)
	Binv = LinearOperator((n,n), matvec = matvec, dtype = 'd')		

	#Compute Y = Q*R such that Q'*B*Q = I, R can be discarded
	start = time()
	if orth == 'mgs':
		Bq, q, _  = mgs(Binv, Y, verbose = False)
	elif orth == 'mgsr':
		Bq, q, _  = mgs_stable(Binv, Y, verbose = False)
	elif orth == 'cholqr':
		Bq, q, _  = cholqr(Binv, Y, verbose = False)
	elif orth == 'precholqr':
		Bq, q, _  = precholqr(Binv, Y, verbose = False)
	else:
		raise NotImplementedError
		
	Borthtime = time()-start
	
	if verbose:	print "B-orthonormalization time in eigenvalue calculation is %g " %(Borthtime) 


	start = time()
	T = np.zeros((k,k), dtype = 'd')	
	if twopass == True:
		for i in np.arange(k):
			Aq = A.matvec(q[:,i])
			for j in np.arange(k):
				T[i,j] = np.dot(Aq,q[:,j])

		#Eigen subproblem
		w, v = eigh(T)
		#Reverse eigenvalues in descending order
		w = w[::-1]
	else:
		from scipy.linalg import inv, svd, pinv, pinv2
		#T = np.dot(np.dot(q.T, Y),inv(np.dot(Bq.T, Omega)))
	
		OAO = np.dot(Omega.T, Y)
		QtBO = np.dot(Bq.T, Omega)
		T = np.dot(inv(QtBO.T), np.dot(OAO, inv(QtBO)))

		w, v = eigh(T); w = w[::-1]

	eigcalctime = time()-start
	if verbose:	print "Calculating eigenvalues took %g" %(eigcalctime)

	print "Total time taken for Eigenvalue calculations is %g" % \
			 (matvectime + Borthtime + eigcalctime)
	
	if error:
		#Compute error estimate
		r = 5
		O = np.random.randn(n,r)
		err = np.zeros((r,), dtype = 'd')
		AO = np.zeros((n,r), dtype = 'd')
		BinvAO = np.zeros((n,r), dtype = 'd')
		for i in np.arange(r):
			AO[:,i]     = A.matvec(O[:,i])
			BinvAO[:,i] = B.solve(AO[:,i])
			diff =	BinvAO[:,i] - np.dot(q,np.dot(q.T,AO[:,i]))		
			err[i] = np.sqrt(np.dot(diff.T, B.matvec(diff)) )
	
		BinvNorm = np.max(np.apply_along_axis(np.linalg.norm, 0, q))

		from math import pi
		print "Error in B-norm is %g" %(10.*np.sqrt(2./pi)*BinvNorm*np.max(err))
	
		
	#Compute eigenvectors		
	u = np.dot(q, v[:,::-1])	
	Bu = np.dot(Bq, v[:,::-1])


	k = k - p
	w = w[:k]
	u = u[:,:k]
	
	if not both:
		return w, u
	else:	
		return w, u, Bu[:,:k]



        
if __name__ == '__main__':

	n = 100
	x = np.linspace(0,1,n)
	X, Y = np.meshgrid(x, x)
	Q = np.exp(-np.abs(X-Y))
	
	class LowRank:
		def __init__(self, Q, n):
			self.Q = Q 
			self.shape = (n,n)

		def matvec(self, x):
			return np.dot(Q,x)
	

	mat = LowRank(Q,n)
	matop = aslinearoperator(mat)
	
	l,v = RandomizedHEP(matop, k = 10, twopass = False)	

	le, ve = eig(Q)
	le = np.real(le)	

	#Test A-orthonormalize
	z = np.random.randn(n,10)
	q,_,r = Aorthonormalize(matop,z,verbose = False)



	class Identity:
		def __init__(self, n):
			self.shape = (n,n)

		def matvec(self, x):
			return x
		def solve(self, x):
			return x

	id_ = Identity(n)
	
	l_, v_ = RandomizedGHEP(matop, id_, 10)

	#print l_, le[:10]


	V = np.random.randn(n,10)
	I = id_
	u,d = LowRankConversion(V,mat)
	d = np.diag(d)   
	print np.linalg.norm(np.dot(V,V.T)-np.dot(u,np.dot(d,u.T)),2)

        U = np.random.randn(n,10);   V = np.random.randn(n,10);
	u1,d1 = LowRankConversion(U,mat);   u2, d2 = LowRankConversion(V,mat);
	
	w, d = AddSymmetricLowRankMatrices(u1,d1,u2,d2,mat)
	print np.linalg.norm(np.dot(w,np.dot(np.diag(d),w.T)) -np.dot(U,U.T) - np.dot(V,V.T),2)
    
    

