import numpy as np



def LanczosGHEP(A, B, k = 10, p = 20):
	""" Solve B^{-1}A x = lambda * x with B-orthogonal inner products"""

	n = A.shape[0]

	k += p	

	#Initialize Lanczos vectors
	V = np.zeros((n,k), dtype = 'd')
	W = np.zeros((n,k), dtype = 'd')
	
	
	#Initialize various quantities	
	l = np.zeros((k,), dtype = 'd')
	beta = np.zeros((k+1,), dtype = 'd')
	alpha = np.zeros((k,), dtype = 'd')
	
	T = np.zeros((k,k), dtype = 'd')
	err = np.zeros((k,), dtype = 'd')

	q = np.random.randn(n)[:,np.newaxis]
	r = B.matvec(q)
	beta[0] = np.sqrt(np.dot(q.T, r))



	maxcount = 20

	for j in np.arange(k):
		
		W[:,j] = r.ravel(); 	W[:,j] /= beta[j]
		V[:,j] = q.ravel();	V[:,j] /= beta[j]	
	
		#Matrix vector product with A
		r = A.matvec(V[:,j])
		if j != 0:
			r -= beta[j-1]*W[:,j-1] 
		
		alpha[j] = np.dot(V[:,j].T,r)
		r -= alpha[j]*W[:,j] 

		#Reorthogonalization
		for l in np.arange(maxcount):
			Vtr = np.dot(V[:,:j].T,r)		
			if np.linalg.norm(Vtr) <= 1.e-10:
				break
			r -= np.dot(W[:,:j],Vtr)
		
		if l == maxcount:
			raise np.linalg.LinAlgError, "Reorthogonalization insufficient"
							
	
		q = B.solve(r)
		beta[j+1] = np.sqrt(np.dot(q.T, r))
		
		if np.abs(beta[j+1]) < 1.e-10:
			raise np.linalg.LinAllError, "Beta is too small %g" %(beta)


	#Construct T
	T = np.diag(alpha) + np.diag(beta[1:k],k=1) + np.diag(beta[1:k],k=-1)

	#Compute approximate Ritz values and vectors
	Theta, S = np.linalg.eig(T)

	#Sort eigenvalues largest to smallest
	ind  	= np.argsort(Theta)
	Theta 	= Theta[ind[::-1]];	
	S  	= S[:,ind[::-1]];


	#Approximate eigenvectors	
	X = np.dot(V, S)
	
	#Check for convergence of the eigenvalues using error estimates	
	for j in np.arange(k):
		err[j] = np.abs(beta[k]*S[k-1,j])		
	

	k -= p
	print "Number of converged eigenvalues are %g" %(np.count_nonzero(err[:k] < 1.e-7))

	return Theta[0:k], X[:,:k]


def LanczosGHEPExplicitRestarted(A, B, k = 10):
	""" Solve B^{-1}A x = lambda * x with B-orthogonal inner products"""

	n = A.shape[0]

	m = k 	

	#Initialize Lanczos vectors
	V = np.zeros((n,m), dtype = 'd')
	W = np.zeros((n,m), dtype = 'd')
	
	
	#Initialize various quantities	
	l = np.zeros((m,), dtype = 'd')
	beta = np.zeros((m+1,), dtype = 'd')
	alpha = np.zeros((m,), dtype = 'd')
	
	T = np.zeros((m,m), dtype = 'd')
	err = np.zeros((m,), dtype = 'd')

	q = np.random.randn(n)[:,np.newaxis]
	r = B.matvec(q)
	beta[0] = np.sqrt(np.dot(q.T, r))


        tol      = 1.e-7
	maxiter  = 5
	maxcount = 20

        k = 0

        for i in np.arange(maxiter):
	        for j in np.arange(k,m):
		
		        W[:,j] = r.ravel(); 	W[:,j] /= beta[j]
	        	V[:,j] = q.ravel();	V[:,j] /= beta[j]	
	
	        	#Matrix vector product with A
		        r = A.matvec(V[:,j])
		        if j != 0:
			        r -= beta[j-1]*W[:,j-1] 
		
		        alpha[j] = np.dot(V[:,j].T,r)
		        r -= alpha[j]*W[:,j] 

		        #Reorthogonalization
	        	for l in np.arange(maxcount):
			        Vtr = np.dot(V[:,:j].T,r)		
			        if np.linalg.norm(Vtr) <= 1.e-10:
				        break
			        r -= np.dot(W[:,:j],Vtr)
		
		        if l == maxcount:
                                from matplotlib import pyplot as plt
                                VtW = np.dot(V[:,:j].T,W[:,:j])
                            
                                plt.pcolor(np.log(np.abs(VtW-np.eye(j))))
                                plt.title(r'$\log(V^TW - I)$')
                                plt.savefig('VtW.png')    
                                print j, np.linalg.norm(Vtr)
			        raise np.linalg.LinAlgError, "Reorthogonalization insufficient"
							
	
		        q = B.solve(r)
		        beta[j+1] = np.sqrt(np.dot(q.T, r))


	        #Construct T
	        T = np.diag(alpha) + np.diag(beta[1:m],k=1) + np.diag(beta[1:m],k=-1)


	
	        #Compute approximate Ritz values and vectors
	        Theta, S = np.linalg.eig(T)

	        #Sort eigenvalues largest to smallest
	        ind  	= np.argsort(Theta)
	        Theta 	= Theta[ind[::-1]];	
	        S       = S[:,ind[::-1]];

	        #Check for convergence of the eigenvalues using error estimates	
	        for j in np.arange(m):
		        err[j] = np.abs(beta[m]*S[m-1,j])		

                ind = np.nonzero(err <= tol)[0]
                k = ind.size
                print ind, err, k, Theta[ind]
                
                #V[:,:k] = np.dot(V, S[:,ind])
                #W[:,:k] = np.dot(W, S[:,ind])



	#Approximate eigenvectors	
	X = np.dot(V, S)
	print np.count_nonzero(err[:k] > 1.e-7)

	return Theta[0:k], X[:,:k]




				
if __name__ == '__main__':
	
	print "Help"
