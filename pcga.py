#from linear import LinearInversion, FisherInverse, Residual
from scipy.sparse.linalg import aslinearoperator, LinearOperator
from covariance import *
from time import time

__all__ = ['PCGA']

class PCGAInversion:
    """
    Solve inverse problem with PCGA (approx to quasi-linear method)
    """

    def __init__(self, forward_model, params, pts, obs = None, X = None):
        
        # Forward solver
        self.forward_model = forward_model
        
        # Store parameters
        self.params = params		
        
        # Grid points
        self.pts = pts
        self.m   = np.size(pts,0) 

        #Observations
        self.obs = obs
        self.n = np.size(n,0)
        
        #Covariance matrix
        self.Q = None
        
        #Noise covariance
        self.R = self.params["R"]
        
        
        #CrossCovariance computed by Jacobian-free low-rank FD approximation
        self.Psi = None # HQH + R
        self.HQ = None 
        self.HX  = None	
        self.HZ = None
        self.Hs = None

        #Eigenvalues and eigenvectors of the prior covariance matrix
        self.priorU = None
        self.priorD = None

        #Eigenvalues and eigenvectors of the posterir covariance matrix
        # will be implemented later
        #self.postU = None
        #self.postD = None
       
        #Preconditioner
        self.P = None

        # Parallel support (smp, cluster)
        if 'parallel' in params:
            self.parallel = self.params['parallel']
        else:
            self.parallel = False

        # Drift or Prior structure
        # Drift functions
        if 'drift' in params and self.X != None:
            self.DriftFunctions(params['drift'])

        # PCGA parameters
        if 'eps' in params:
            self.eps = params['eps']
        else:
            self.eps = 1.e-4

        #Generate measurements - currently not available, but will be useful
        #if st != None:
        #    #self.CreateSyntheticData(st)
        #    raise NotImplementedError

		#Printing verbose - not implemented yet
        self.verbose = False if 'verbose' not in self.params else self.params['verbose']

    def DriftFunctions(self, method):
        if method == 'constant':
            self.p = 1
            self.X = np.ones((self.m,1),dtype ='d')/np.sqrt(self.m)
        elif method == 'linear':
            self.p = 1 + np.size(self.pts,1)
            self.X = np.ones((self.m,self.p), dtype = 'd')
            self.X[:,1:self.p] = np.copy(self.pts)
        elif method == 'none':
            self.p = 0
            self.X = np.zeros((self.m,1),dtype = 'd')
        else:
            raise NotImplementedError
        return

    def Covariance(self, method, kernel, **kwargs):
        self.Q = CovarianceMatrix(method, self.pts, kernel, **kwargs)
        return
    
    def EigCov(self,method,ker):
        
        return 
    
    def ForwardSolve(self,s):
        simul_obs = forward_model.run(s)
        return simul_obs

    def JacVect(self, x, s, simul_obs, eps, delta = None, dir_id = None):
        '''
        Jacobian times Vector
        perturbation interval delta determined following Brown and Saad [1990]
        '''
        if delta == None:
            mag = np.dot(s,x)
            absmag = np.dot(abs(s),abs(x))
            delta_cur = np.sign(mag)*eps*(max(abs(mag),absmag))/(np.linalg.norm(x)**2)
        
        forward_model = self.forward_model

        if dir_id == None:
            dir_id = 0
        # solve Hx HZ HQT
        Jx = (forward_model.Run(s+delta*x,dir_id) - simul_obs)/delta
        return Jx
    
    def CreateSyntheticData(self, s, noise = False):
        n = self.n
        
        #Generate measurements
        self.obs = self.forward_model.Run(s)
        
        #Add noise
        if noise:
            self.obs += np.sqrt(self.R)*np.random.randn(n,1)

        return

    def ObjectiveFunction(self, shat, beta, simul_obs):
        """
            0.5*(s-Xb)^TQ^{-1}(s-Xb) + 0.5(y-h(s))^TR^{-1}(y-h(s))
        """
        if simul_obs == None:
            simul_obs = self.forward_model.solve(shat)

        smxb = shat - np.dot(self.X,beta)
        Qinvs = self.Q.solve(smxb)
        
        ymhs = self.obs - simul_obs
        obj = 0.5*np.dot(ymhs.T,ymhs)/self.R + 0.5*np.dot(smxb.T,Qinvs)

        return obj

    def DirectSolve(self, save = False):
        """
        Solve the geostatistical system using a direct solver.
        Not to be used unless the number of measurements are small O(100)
        """
        n = self.n
        p = self.p
        
        #Construct HX
        HX = np.zeros((n,p), dtype='d')
        for i in range(p):
            HX[:,i] = self.JacVect(self.X[:,i],self.scur,simul_obs, self.eps, dir_id = i)

        #Construct HZ
        HZ = np.zeros((n,m), dtype='d')
        for i in range(p):
            HZ[:,i] = self.JacVect(self.Z[:,i],self.scur,simul_obs, self.eps, dir_id = i)
        
        HQ = np.dot(HZ,self.Z.T) 
        
        #Get Psi
        Psi = np.dot(HZ,HZ.T) + np.eye(n,dtype='d') 

        #Create matrix system and solve it
        A = np.zeros((n+p,n+p),dtype='d')
        b = np.zeros((n+p,1),dtype='d')

        A[0:n,0:n] = np.copy(Psi);   
        A[0:n,n:n+p] = np.copy(HX);   
        A[n:n+p,0:n] = np.copy(HX.T);
        
        b[:n] = self.obs[:]

        Hs = self.JacVect(self.scur,self.scur,simul_obs, self.eps)
        b[:n] = b[:n] - simul_obs + Hsu[:]

        x = np.linalg.solve(A, b)

        ##Extract components and return final solution
        
        xi = x[0:n]
        beta = x[n:n+p]
        shat = np.dot(self.X,beta) + np.dot(self.HQ.T,xi)

        if save: 
            self.HX = HX
            self.HZ = HZ
            self.Psi = Psi
            self.QHT = QHT
            self.Hs = Hs

        return shat, beta
    
    def LinearIteration(self, shat, scur):
        self.s = scur
        #Modification	
        obs_mod = np.zeros_like(self.obs, dtype = 'd')

        direct = self.direct
        precond = self.precond
            
        #Solve geostatistical system
        if direct:
            sc, betac = self.DirectSolve(recompute = True)
        else:
            #Construct preconditioner	
            if precond:	self.ConstructPreconditioner()

            sc, betac = self.IterativeSolve(precond = precond, recompute = True)
            
        return sc, betac	

    def ComputePosteriorDiagonalEntriesDirect(self, recompute = True):
        """		
		Works best for small measurements O(100)
		not implemented yet
        """
        return
    
    def FssInvDiag(self, recompute = True):
        """		
        Works best for small measurements O(100)
        """
        return
    
    def GeneralizedEigendecomposition(self):
        """
        Saibaba, Lee and Kitanidis, 2016
        """
        return
    
    def CompareSpectrum(self, filename):
        """
        Compare spectrum of Hred, Qinv, and the combination of the two.
        """
        return
    
    def SetupRealizations(self, k = 100):
        return

	def GenerateUnconditionalRealizations(self):
		m = self.m
		eps = np.random.randn((m,), dtype = 'd')
		y   = np.zeros_like(eps, dtype = 'd')


		#Decide which method to use if available in params
		method = 'eigen' if 'uncond' not in self.params else self.params['uncond']

		if method == 'eigen':
						
			if self.priorL == None or self.priorV == None:

				k = 100 if 'k' not in self.params else self.params['k']
				self.SetupRealiations(k = k)

			L, V = self.priorL, self.priorV
	
			y = self.dot(V, np.dot(np.diag(L), np.dot(V.T, eps))) 

		else:
			raise NotImplementedError	
		return y
	
	def ConstructPreconditioner(self):
        m = self.m
        n = self.n
        p = self.p
        
        k = 100 if 'k' not in self.params else self.params['k']
        
        #1. Factorize Q
        if self.priorL == None or self.priorV == None:
            self.SetupRealizations(k = k)		

        L = self.priorL
        V = self.priorV

		#2. Form HV
		HV = np.zeros((n,k),'d')
		for i in np.arange(k):
            HV[:,i] = self.HMult(V[:,i])*np.sqrt(L[i]/self.R)
	
		#3. Compute SVD		
		from scipy.linalg import svd
		u, s, _ = svd(HV, full_matrices = False)
		
		#4. Prepare for Sherman-Morrisson-Woodbury update
		d = s**2./(s**2.+1.)
		self.u = np.dot(u, np.diag(np.sqrt(d)))
	
		#5. Construct Sinv
		PhiInvHX = np.zeros((n,p), dtype = 'd')
			
		self.ConstructHX()
		HX = self.HX
		R = self.R
		for i in np.arange(p):
			PhiInvHX[:,i] = (HX[:,i] - np.dot(self.u, np.dot(self.u.T,HX[:,i])))/R 


		from scipy.linalg import inv
		if p != 0:
			self.Sinv = inv(np.dot(HX.T, PhiInvHX))
	

		#Create preconditioner context
		P = Preconditioner(self)
		self.P = aslinearoperator(P)

		return

	def IterativeSolve(self, precond = False, ymod = None, su = None, recompute = False):
        from scipy.sparse.linalg import gmres, minres
		
		n = self.n
		p = self.p
		
		#Construct HX
		if recompute or self.HX == None:
			self.ConstructHX()
		HX = self.HX
		
	
		#Create matrix context
		A = MatVec(inv = self)
		self.A = aslinearoperator(A)

		b = np.zeros((n+p,1), dtype = 'd')
		b[:n] = self.y[:]
		if ymod != None:
			b[:n] -= ymod
	

		callback = Residual()
        	#Residua and maximum iterations	
		itertol = 1.e-10 if not 'itertol' in self.params else self.params['itertol']
        	maxiter = n if not 'maxiter' in self.params else self.params['maxiter']
        	
		if self.P == None:
			x, info = minres(self.A, b, tol = itertol, maxiter = maxiter, callback = callback)
		else:	
			restart = 50 if 'gmresrestart' not in self.params else self.params['gmresrestart']
			x, info = gmres(self.A, b, tol = itertol, restart = restart, maxiter = maxiter, callback = callback, M = self.P)
		
		
		print "Number of iterations for geostatistical solver %g" %(callback.itercount())
		assert info == 0
	
		#Extract components and postprocess
		s, beta = self.PostProcess(x)
		
		return s, beta		
	
		
	def LinearInversionKnownMean(self, beta = 0.):
		#Currently only implemented for iterative solver

		m = self.m
		n = self.n

		assert self.p == 0


		if isinstance(self.H, csr_matrix) or isinstance(self.H, np.ndarray):
			H = aslinearoperator(self.H)
		else:
			H = self.H

		Q = self.Q
		R = self.R


		#Matrix operator
		Psi = LinearOperator(shape = (n,n), matvec = lambda x: H.matvec(Q.matvec(H.rmatvec(x))) + R*x, dtype = 'd') 
	
		#Construct preconditioner
		self.ConstructPreconditioner()

		U  = self.u
        P  = LinearOperator(shape = (n,n), matvec = lambda x:  (x - np.dot(U, np.dot(U.T,x)))/R , dtype = 'd')

        #Residual and maximum iterations	
        callback = Residual()
        itertol = 1.e-10 if not 'itertol' in self.params else self.params['itertol']
        maxiter = n if not 'maxiter' in self.params else self.params['maxiter']

		#Create right hand side
		mu = beta*np.ones((m,1), dtype = 'd')
		b  = self.y - H.matvec(mu)


		#Solve using iterative preconditioner Krylov solver
		restart = 50 if 'gmresrestart' not in self.params else self.params['gmresrestart']
		from scipy.sparse.linalg import gmres
		x, info = gmres(Psi, b, tol = itertol, restart = restart, maxiter = maxiter, callback = callback, M = P)	
		print("Number of iterations for geostatistical solver %g" % (callback.itercount()))
		assert info == 0

		#Postprocess solution 
		s = mu.flatten() + Q.matvec(H.rmatvec(x))
        return s
    
    def GaussNewton(self, filename):
        
        m = self.m
        sp = np.zeros((m,), dtype = 'd')
        sc = np.zeros((m,), dtype = 'd')
        
        graph = False if 'graph' not in self.params else self.params['graph']

        maxiter = self.params['maxiter']
        restol  = self.params['restol']	
        iterc   = maxiter

        obj = 0.0
        self.Q.BuildPreconditioner(k = 100)

        res = 1.
        
        for i in np.arange(maxiter):
            phic = self.pde.ForwardSolve(self.pde.sources)
            start = time()
            sc, betac = self.LinearIteration(sc, phic)
            print("Time for iteration %g is %g" %(i+1, time()-start))

            if i > 1:
                res = np.linalg.norm(sp-sc)/np.linalg.norm(sp)
            
            obj = -1.
        
            if self.params['obj']:	
                obj = self.ObjectiveFunction(sc, betac)
        
            err = np.linalg.norm(sc.ravel()-self.st)/np.linalg.norm(self.st)

            print("At iteration %g, relative residual is %g, objective function is %g and error is %g" %(i+1, res, obj, err))

            if res < restol:
                iterc = i + 1
                break
        
            sp = np.copy(sc)	

            #plot if necessary

        return sc, betac, iterc					

    def Uncertainty(self, **kwargs):

        m = self.pde.m
        diag = np.zeros((m,), dtype = 'd')

        start = time()	

        if self.params['direct'] == True:
            diag = self.ComputePosteriorDiagonalEntriesDirect()
        else:
            F = FisherInverse(self)
            prior, datalowrank, beta = F.diags()
            print("Uncertainty in beta is %g" %(beta))

        print("Time taken for uncertainty computation is ", time() - start)
        #plot
        return