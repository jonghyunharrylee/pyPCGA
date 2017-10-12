import numpy as np
from scipy.linalg import qr

__all__ = ['mgs','mgs_stable','cholqr','precholqr']

def mgs(A, Z, verbose = False):
	""" Produce q'*A*q = I with inner product <x,y>_A = y'* A * x 
	    using Modified Gram-Schmidt
	"""
	
	#Get sizes
	n = np.size(Z,0);	k = np.size(Z,1)
	
	#Initialize
	Aq = np.zeros_like(Z, dtype  = 'd')
	q  = np.zeros_like(Z, dtype = 'd')
	r  = np.zeros((k,k), dtype = 'd')
		

	z  = Z[:,0]
	Aq[:,0] = A.matvec(z)	


	r[0,0] = np.sqrt(np.dot(z.T, Aq[:,0]))
	q[:,0] = Z[:,0]/r[0,0]

	Aq[:,0] /= r[0,0]

	for j in np.arange(1,k):
		q[:,j] = Z[:,j]
		for i in np.arange(j):
			r[i,j] = np.dot(q[:,j].T,Aq[:,i])
			q[:,j] -= r[i,j]*q[:,i]


		Aq[:,j] = A.matvec(q[:,j])
		r[j,j]  = np.sqrt(np.dot(q[:,j].T,Aq[:,j]))

		#If element becomes too small, terminate
		if np.abs(r[j,j]) < 1.e-14:
			k = j;	
			
			q = q[:,:kt]
			Aq = Aq[:,:kt]
			r = r[:kt,:kt]

			print "A-orthonormalization broke down"
			break
		
		q[:,j]  /= r[j,j]	
		Aq[:,j] /= r[j,j]	

	q = q[:,:k]
	Aq = Aq[:,:k]
	r = r[:k,:k]

	if verbose:
		#Verify Q*R = Y
		print "||QR-Y|| is ", np.linalg.norm(np.dot(q,r) - Z[:,:k], 2)
		
		#Verify Q'*A*Q = I
		T = np.dot(q.T, Aq)
		print "||Q^TAQ-I|| is ", np.linalg.norm(T - np.eye(k, dtype = 'd'), ord = 2)		

		#verify Q'AY = R 
		print "||Q^TAY-R|| is ", np.linalg.norm(np.dot(Aq.T,Z[:,:k]) - r,2)

		#Verify YR^{-1} = Q
		print "||YR^{-1}-Q|| is ", np.linalg.norm(np.linalg.solve(r.T,Z[:,:k].T).T-q,2)



	return q, Aq, r 


def mgs_stable(A, Z, verbose = False):
	""" Produce q'*A*q = I with inner product <x,y>_A = y'* A * x 
	    using Modified Gram-Schmidt
	"""
	#Get sizes
	m = np.size(Z,0);	n = np.size(Z,1)
	
	#Initialize
	Aq = np.zeros_like(Z, dtype  = 'd')
	q  = np.zeros_like(Z, dtype = 'd')
	r  = np.zeros((n,n), dtype = 'd')
		
	reorth = np.zeros((n,), dtype = 'd')
	eps = np.finfo(np.float64).eps

	q = np.copy(Z)

	for k in np.arange(n):
		Aq[:,k] = A.matvec(q[:,k])
		t = np.sqrt(np.dot(q[:,k].T,Aq[:,k]))
	
		nach = 1;	u = 0;
		while nach:
			u += 1
			for i in np.arange(k):
				s = np.dot(Aq[:,i].T,q[:,k])
				r[i,k] += s
				q[:,k] -= s*q[:,i];
			
			Aq[:,k] = A.matvec(q[:,k])	
			tt = np.sqrt(np.dot(q[:,k].T,Aq[:,k]))
			if tt > t*10.*eps and tt < t/10.:
				nach = 1;	t = tt;
			else:
				nach = 0;
				if tt < 10.*eps*t:	tt = 0.
			

		reorth[k] = u
		r[k,k] = tt
		tt = 1./tt if np.abs(tt*eps) > 0. else 0.
		q[:,k]  *= tt
		Aq[:,k] *= tt
	
	#print reorth	

	if verbose:
		#Verify Q*R = Y
		print "||QR-Y|| is ", np.linalg.norm(np.dot(q,r) - Z, 2)
		
		#Verify Q'*A*Q = I
		T = np.dot(q.T, Aq)
		print "||Q^TAQ-I|| is ", np.linalg.norm(T - np.eye(n, dtype = 'd'), ord = 2)		

		#verify Q'AY = R 
		print  "||Q^TAY-R|| is ", np.linalg.norm(np.dot(Aq.T,Z) - r,2)

		#Verify YR^{-1} = Q
		val = np.inf
		try:
			val = np.linalg.norm(np.linalg.solve(r.T,Z.T).T-q,2)
		except LinAlgError:
			print "Singular"
		print "||YR^{-1}-Q|| is ", val

	return q, Aq, r 


def cholqr(A, Z, verbose = False):
	B = np.apply_along_axis(lambda x: A.matvec(x), 0, Z)
	C = np.dot(Z.T, B)
	
	r = np.linalg.cholesky(C).T
	q = np.linalg.solve(r.T,Z.T).T
	Aq = np.linalg.solve(r.T,B.T).T

	if verbose:
		
		#Verify Q*R = Y
		print "||QR-Y|| is ", np.linalg.norm(np.dot(q,r) - Z, 2)
		
		#Verify Q'*A*Q = I
		T = np.dot(q.T, Aq)
		n = T.shape[1]
		print "||Q^TAQ-I|| is ", \
			np.linalg.norm(T - np.eye(n, dtype = 'd'), ord = 2)		

		#verify Q'AY = R 
		print "||Q^TAY-R|| is ", np.linalg.norm(np.dot(Aq.T,Z) - r,2)

		#Verify YR^{-1} = Q
		val = np.inf
		try:
			val = np.linalg.norm(np.linalg.solve(r.T,Z.T).T-q,2)
		except LinAlgError:
			print "||YR^{-1}-Q|| is ", "Singular"
		print "||YR^{-1}-Q|| is ", val

	return q, Aq, r 		

def precholqr(A, Z, verbose = False):

	y, s = qr(Z, mode = 'economic')
	q, Aq, u = cholqr(A, y, False)
	r = np.dot(u,s)

	if verbose:
		
		#Verify Q*R = Y
		print "||QR-Y|| is ", np.linalg.norm(np.dot(q,r) - Z, 2)
		
		#Verify Q'*A*Q = I
		T = np.dot(q.T, Aq)
		n = T.shape[1]
		print "||Q^TAQ-I|| is ", \
			np.linalg.norm(T - np.eye(n, dtype = 'd'), ord = 2)		

		#verify Q'AY = R 
		print  "||Q^TAY-R|| is ", np.linalg.norm(np.dot(Aq.T,Z) - r,2)

		#Verify YR^{-1} = Q
		val = np.inf
		try:
			val = np.linalg.norm(np.linalg.solve(r.T,Z.T).T-q,2)
		except LinAlgError:
			print "||YR^{-1}-Q|| is ", "Singular"
		print "||YR^{-1}-Q|| is ", val

	return q, Aq, r	
	

if __name__ == '__main__':

	m, n  = 100, 5

	from scipy.sparse import eye
	from scipy.sparse.linalg import aslinearoperator
	y = np.random.randn(m,n)

	A = aslinearoperator(2.*eye(m,m))

	#mgs_stable(A, y, verbose = True)
	#cholqr(A, y, verbose = True)
	precholqr(A, y, verbose = True)	
