from time import time
from math import isnan, sqrt
 

import numpy as np

from covariance.mat import *

from scipy.sparse.linalg import gmres, minres # IterativeSolve
from scipy.sparse.linalg import LinearOperator # Matrix-free IterativeSolve

from IPython.core.debugger import Tracer; debug_here = Tracer()

__all__ = ['PCGA']

class PCGA:
    """
    Solve inverse problem with PCGA (approx to quasi-linear method)
    tried to keep every array in 2D columns
    """

    def __init__(self, forward_model, s_init, pts, params, s_true = None, obs = None, obs_true = None, X = None):
        print('##### PCGA Inversion #####')
        print('##### 1. Initialize forward and inversion parameters')

        # forward solver setting should be done externally as a blackbox
        # currently I define model seperately, but might be able to  merge them into one function if needed
        self.forward_model = forward_model
        
        # Grid points (for Hmatrix and FMM)
        self.pts = pts
        self.m   = np.size(pts,0) 
 
        # Store parameters
        self.params = params		
        
        # inversion setting
        self.s_init = s_init
        self.s_true = s_true
        if s_true is not None:
            assert(self.m == np.size(s_true,0))
        
        # keep track of some values (best, init)
        self.s_best = None
        self.simul_obs_best = None
        self.iter_best = 0
        self.obj_best = 1.e+20
        self.simul_obs_init = None

        #Observations
        self.obs = obs 
        if obs is None:
            self.n = 0
        else:
            self.n = np.size(obs,0)

        self.obs_true = obs_true  # observation without noise for synthetic problems
        
        # drift parameter (determined )
        self.p   = 0
        
        #Covariance matrix
        self.Q = None
        self.covariance_matvec = self.params['covariance_matvec']
        self.xmin = params['xmin']
        self.xmax = params['xmax']
        self.N = params['N']
        self.theta = params['theta']
        self.kernel = params['kernel']
        
        if 'n_pc' in params:
            self.n_pc = params['n_pc']
        else:
            return ValueError('provide n_pc')


        assert(np.size(self.xmin,0) == np.size(self.xmax,0))
        assert(np.size(self.xmin,0) == np.size(self.N,0))
        assert(np.size(self.xmin,0) == np.size(self.theta,0))
        
        # objetive function evaluation
        if 'objeval' in params:
            self.objeval = params['objeval']
        else:
            self.params['objeval'] = False
            self.objeval = False


        #Noise covariance
        if 'R' in params:
            self.R = params['R']
        else:
            return ValueError('provide R')

        #Eigenvalues and eigenvectors of the prior covariance matrix
        self.priorU = None
        self.priord = None

        #Eigenvalues and eigenvectors of the posterir covariance matrix - will be implemented later
        #self.postU = None
        #self.postd = None

        # Parallel support (only for single machine, will add mpi later)
        if 'parallel' in params:
            self.parallel = params['parallel']
            if 'ncores' in params:
                self.ncores = params['ncores']
            else:
                # get number of physcial cores
                from psutil import cpu_count # physcial cpu counts
                self.ncores = cpu_count(logical=False)
        else:
            self.parallel = False
            self.ncores = 1

        #CrossCovariance computed by Jacobian-free low-rank FD approximation if you want to save..
        if 'JacSave' in params:
            self.JacSave =   params['JacSave']
            self.HX  = None	
            self.HZ = None
            self.Hs = None
        else:
            self.JacSave = False
        
        # Matrix solver - default : False (iterative)
        if 'direct' in params:
            self.direct = params['direct']
        else:
            self.direct = False 

        direct = self.direct

        # No preconditioner for now
        # default : False 
        if 'precond' in params: 
            self.precond = params['precond']
        else:
            self.precond = False 

        #Preconditioner
        self.P = None

        # 
        self.linesearch = False # not now
        if 'LM' in params:
            self.LM = params['LM'] # Levenberg Marquart
            if 'nopts_LM' in params:
                self.nopts_LM = params['nopts_LM']
            else:
                self.nopts_LM = self.ncores

            if 'alphamax_LM' in params:
                self.alphamax_LM = params['alphamax_LM']
            else:
                self.alphamax_LM = 10.**4. # does it sound ok?
            
        # Define Drift (or Prior) functions 
        if X is not None:
            self.X = X
            self.p = np.size(self.X,0)
        else:
            if 'drift' in params:
                self.DriftFunctions(params['drift'])
            else:
                # add constant drift by default
                params['drift'] = 'constant'
                self.DriftFunctions(params['drift'])
        
        # PCGA parameters
        if 'precision' in params:
            self.precision = params['precision']
        else:
            self.precision = 1.e-8 # assume single precision

        #Generate measurements when obs is not provided AND s_true is given
        if s_true is not None and obs is None:
            self.CreateSyntheticData(s_true, noise = True)
        
		#Printing verbose
        self.verbose = False if 'verbose' not in self.params else self.params['verbose']

    def DriftFunctions(self, method):
        if method == 'constant':
            self.p = 1
            self.X = np.ones((self.m,1),dtype ='d')/np.sqrt(self.m)
        elif method == 'linear':
            self.p = 1 + np.size(self.pts,1)
            self.X = np.ones((self.m,self.p), dtype = 'd')
            self.X[:,1:self.p] = np.copy(self.pts)
        elif method == 'none': # point prior
            self.p = 0
            self.X = np.zeros((self.m,1),dtype = 'd')
            return NotImplementedError
        else: # constant drift
            self.p = 1
            self.X = np.ones((self.m,1),dtype ='d')/np.sqrt(self.m)
        return

    def ConstructCovariance(self, method, kernel, **kwargs):
        print('##### 2. Construct Prior Covariance Matrix')
        start = time()
        self.Q = CovarianceMatrix(method, self.pts, kernel, **kwargs)
        print('- time for covariance matrix construction is %g sec' % round(time()-start))
        return
    
    def ComputePriorEig(self, n_pc=None):
        '''
        Compute Eigenmodes of Prior Covariance
		to be implemented
        '''
        print('##### 3. Eigendecomposition of Prior Covariance')
        m = self.m
        n = self.n
        p = self.p
        
        if n_pc is None:
            n_pc = self.n_pc
        else:
            # say 25?
            n_pc = 25 
        
        assert(self.Q is not None) # have to assign Q through Covariance before

        method = 'arpack' if 'precondeigen' not in self.params else self.params['precondeigen']
        
        #twopass = False if not 'twopass' in self.params else self.params['twopass']
        start = time()
        if method == 'arpack':
            from scipy.sparse.linalg import eigsh
            self.priord, self.priorU = eigsh(self.Q, k = n_pc)
            self.priord = self.priord[::-1]
            self.priord = self.priord.reshape(self.priord.shape[0],-1) # make a column vector
            self.priorU = self.priorU[:,::-1]

        #elif method == 'randomized':
        #    # randomized method to be implemented!
        #    from eigen import RandomizedEIG
        #    self.priorU, self.priorD = RandomizedEIG(self.Q, k =k, twopass = twopass)
        else:
            raise NotImplementedError

        print('- time for eigendecomposition is %g sec' % round(time()-start))

        return
    
    def ForwardSolve(self,s):
        '''
        provide additional settings for your function forward_model 
        '''
        par = False
        simul_obs = self.forward_model(s,par)
        
        return simul_obs

    def ParallelForwardSolve(self,s):
        '''
        provide additional settings for your function forward_model running in parallel
        '''
        par = True
        simul_obs_parallel = self.forward_model(s,par,ncores = self.ncores)
        
        return simul_obs_parallel

    def JacVect(self, x, s, simul_obs, precision, delta = None):
        '''
        Jacobian times Matrix (Vectors) in Parallel
        perturbation interval delta determined following Brown and Saad [1990]
        '''
        nruns = np.size(x,1)
        deltas = np.zeros((nruns,1),'d') 

        if delta is None or math.isnan(delta) or delta == 0:
            for i in range(nruns):
                mag = np.dot(s.T,x[:,i:i+1])
                absmag = np.dot(abs(s.T),abs(x[:,i:i+1]))
                if mag >= 0:
                    signmag = 1.
                else:
                    signmag = -1.

                deltas[i] = signmag*sqrt(precision)*(max(abs(mag),absmag))/((np.linalg.norm(x[:,i:i+1]))**2)
                if deltas[i] == 0:
                    print('%d-th delta: signmag %g, precision %g, max abs %g, norm %g' % (i,signmag, precision,(max(abs(mag),absmag)), (np.linalg.norm(x)**2)))
                    raise ValueError('delta is zero?')

                # reuse storage x by updating x
                x[:,i:i+1] = s + deltas[i]*x[:,i:i+1]

        else:
            for i in range(nruns):
                deltas[i] = delta
                # reuse storage x by updating x
                x[:,i:i+1] = s + deltas[i]*x[:,i:i+1]

        if self.parallel:
            simul_obs_purturbation = self.ParallelForwardSolve(x)
        else:
            for i in range(nruns):
                if i == 0:
                    simul_obs_purturbation = self.ForwardSolve(x)
                else:
                    simul_obs_purturbation = np.concatenate((simul_obs_purturbation, self.ForwardSolve(x)), axis=1)
        
        assert(np.size(simul_obs_purturbation,1) == nruns) # should satisfy this
        
        Jxs = np.zeros_like(simul_obs_purturbation)
        
        # solve Hx HZ HQT
        for i in range(nruns):
            Jxs[:,i:i+1] = np.true_divide((simul_obs_purturbation[:,i:i+1] - simul_obs),deltas[i])
        return Jxs

    def CreateSyntheticData(self, s = None, noise = False):
        '''
        when obs is not provided (and s_true is), create synthetic observations
        '''
        s_true = self.s_true
        R = self.R
        #Generate measurements
        if s is None:
            if s_true is None:
                raise ValueError('plz specify bathymetry')
            else:
                print('- generate noisy observations for this synthetic inversion problem')
                obs_true = self.ForwardSolve(s_true)
        else:
            print('- generate noisy observations for this synthetic inversion problem')
            obs_true = self.ForwardSolve(s)

        n = np.size(obs_true,0)
        self.n = n

        #Add noise
        if noise:
            obs = obs_true + np.sqrt(self.R)*np.random.randn(self.n,1)
        else:
            obs = obs_true
        
        self.obs = obs
        self.obs_true = obs_true

        return obs, obs_true

    def ObjectiveFunction(self, s_cur, beta_cur, simul_obs, approx = True):
        """
            0.5(y-h(s))^TR^{-1}(y-h(s)) + 0.5*(s-Xb)^TQ^{-1}(s-Xb)
        """
        if simul_obs is None:
            simul_obs = self.ForwardSolve(s_cur)

        smxb = s_cur - np.dot(self.X,beta_cur)
        ymhs = self.obs - simul_obs
        
        if approx:
            Zinvs = np.multiply(1./np.sqrt(self.priord),np.dot(self.priorU.T,smxb))
            obj = 0.5*np.dot(ymhs.T,ymhs)/self.R + 0.5*np.dot(Zinvs.T,Zinvs)
        else:
            Qinvs = self.Q.solve(smxb)
            obj = 0.5*np.dot(ymhs.T,ymhs)/self.R + 0.5*np.dot(smxb.T,Qinvs)
        return obj

    def JacMat(self, s_cur, simul_obs):
        
        m = self.m
        n = self.n
        p = self.p
        n_pc = self.n_pc
        precision = self.precision

        Z = np.zeros((m,n_pc), dtype ='d')
        for i in range(n_pc):
            Z[:,i:i+1] = np.dot(sqrt(self.priord[i]),self.priorU[:,i:i+1])

        temp = np.zeros((m,p+n_pc+1), dtype='d') # [HX, HZ, Hs]
        Htemp = np.zeros((n,p+n_pc+1), dtype='d') # [HX, HZ, Hs]
            
        temp[:,0:p] = np.copy(self.X)
        temp[:,p:p+n_pc] = np.copy(Z) 
        temp[:,p+n_pc:p+n_pc+1] = np.copy(s_cur)

        Htemp = self.JacVect(temp,s_cur,simul_obs,precision)
            
        HX = Htemp[:,0:p]
        HZ = Htemp[:,p:p+n_pc]
        Hs = Htemp[:,p+n_pc:p+n_pc+1]
        
        if self.JacSave: 
            self.HX = HX
            self.HZ = HZ
            self.Hs = Hs
        # constructing and returning Z are redundant indeed..
        return HX, HZ, Hs, Z

    def DirectSolve(self, s_cur, simul_obs = None):
        """
        Solve the geostatistical system using a direct solver.
        Not to be used unless the number of measurements are small O(100)
        """
        print("use direct solver for saddle-point (cokrigging) system")
        m = self.m
        n = self.n
        p = self.p
        n_pc = self.n_pc
        
        if simul_obs is None:
            simul_obs = self.ForwardSolve(s_cur)
        
        HX, HZ, Hs, Z = self.JacMat(s_cur, simul_obs)
        
        # Construct HQ directly 
        HQ = np.dot(HZ,Z.T) 
        
        if self.LM:
            print('Levenberg-Marquardt')
            nopts = self.nopts_LM 
            alpha = 10**(np.linspace(0.,np.log10(self.alphamax_LM),nopts))

            beta_all = np.zeros((p,nopts),'d')
            s_hat_all = np.zeros((m,nopts),'d')
            # this is sequential for now
            for i in range(nopts):
                # Construct Psi directly 
                Psi = np.dot(HZ,HZ.T) + np.dot(alpha(i)*self.R,np.eye(n,dtype='d'))

                #Create matrix system and solve it
                # cokriging matrix
                A = np.zeros((n+p,n+p),dtype='d')
                b = np.zeros((n+p,1),dtype='d')

                A[0:n,0:n] = np.copy(Psi);   
                A[0:n,n:n+p] = np.copy(HX);   
                A[n:n+p,0:n] = np.copy(HX.T);
        
                # Ax = b, b = obs - h(s) + Hs 
                b[:n] = self.obs[:] - simul_obs + Hs[:]

                x = np.linalg.solve(A, b)

                ##Extract components and return final solution
                xi = x[0:n,np.newaxis]
                beta_all[:,i:i+1] = x[n:n+p,np.newaxis]
                s_hat_all[:,i:i+1] = np.dot(self.X,beta_all[:,i:i+1]) + np.dot(HQ.T,xi)
            
            # parallel
            print('evaluate solutions')
            if self.parallel:
                simul_obs_all = self.ParallelForwardSolve(s_hat_all)
            else:
                for i in range(nopts):
                    if i == 0:
                        simul_obs_all = self.ForwardSolve(x)
                    else:
                        simul_obs_all = np.concatenate((simul_obs_all, self.ForwardSolve(x)), axis=1)

            assert(np.size(simul_obs_all,1) == nopts)

            obj_best = 1.e+20
            print('%d objective value evaluations' % nopts)
            for i in range(nopts):
                if self.objeval: # If true, we do accurate computation
                    obj = self.ObjectiveFunction(s_hat_all[:,i:i+1], beta_all[:,i:i+1], simul_obs_all[:,i:i+1],0) 
                else: # we compute through PCGA approximation
                    obj = self.ObjectiveFunction(s_hat_all[:,i:i+1], beta_all[:,i:i+1], simul_obs_all[:,i:i+1],1) 
                
                if obj < obj_best: 
                    print('%d-th solution obj %e (alpha %f)' % (i,obj,alpha[i]))
                    s_hat = s_hat_all[:,i:i+1]
                    beta = beta_all[:,i:i+1]
                    simul_obs_new = simul_obs_all[:,i:i+1]
                    obj_best = obj
        else:
            # Construct Psi directly 
            Psi = np.dot(HZ,HZ.T) + np.dot(self.R,np.eye(n,dtype='d'))

            #Create matrix system and solve it
            # cokriging matrix
            A = np.zeros((n+p,n+p),dtype='d')
            b = np.zeros((n+p,1),dtype='d')

            A[0:n,0:n] = np.copy(Psi);   
            A[0:n,n:n+p] = np.copy(HX);   
            A[n:n+p,0:n] = np.copy(HX.T);
        
            # Ax = b, b = obs - h(s) + Hs 
            b[:n] = self.obs[:] - simul_obs + Hs[:]

            x = np.linalg.solve(A, b)

            ##Extract components and return final solution
            xi = x[0:n,np.newaxis]
            beta = x[n:n+p,np.newaxis]
            s_hat = np.dot(self.X,beta) + np.dot(HQ.T,xi)
            simul_obs_new = self.ForwardSolve(s_hat)

        return s_hat, beta, simul_obs_new

    def IterativeSolve(self, s_cur, simul_obs = None, precond = False):
        m = self.m
        n = self.n
        p = self.p
        n_pc = self.n_pc

        if simul_obs is None:
            simul_obs = self.ForwardSolve(s_cur)

        HX, HZ, Hs, Z = self.JacMat(s_cur, simul_obs)

        if self.LM:
            print('Levenberg-Marquardt')
            nopts = self.nopts_LM 

            alpha = 10**(np.linspace(0.,np.log10(self.alphamax_LM),nopts))
            beta_all = np.zeros((p,nopts),'d')
            s_hat_all = np.zeros((m,nopts),'d')

            # this is sequential for now
            for i in range(nopts):

                #Create matrix context
                def mv(v):
                    return np.concatenate((np.dot(HZ,np.dot(HZ.T,v[0:n])) + np.dot(alpha[i]*self.R,v[0:n]) + np.dot(HX,v[n:n+p]), np.dot(HX.T,v[0:n])),axis = 0)
                # Matrix handle
                Afun = LinearOperator( (n+p,n+p), matvec=mv ,dtype = 'd')

                b = np.zeros((n+p,1), dtype = 'd')
                b[:n] = self.obs[:] - simul_obs + Hs[:]
        
                callback = Residual()
        
                #Residua and maximum iterations	
                itertol = 1.e-10 if not 'iterative_tol' in self.params else self.params['iterative_tol']
                maxiter = n if not 'iterative_maxiter' in self.params else self.params['iterative_maxiter']
        
                if self.P is None:
                    x, info = minres(Afun, b, tol = itertol, maxiter = maxiter, callback = callback)
                    if self.verbose: print("-- Number of iterations for minres %g" %(callback.itercount()))

                else:
                    restart = 50 if 'gmresrestart' not in self.params else self.params['gmresrestart']
                    x, info = gmres(Afun, b, tol = itertol, restart = restart, maxiter = maxiter, callback = callback, M = self.P)
                    if self.verbose: print("-- Number of iterations for gmres %g" %(callback.itercount()))

                assert(info == 0)
                #Extract components and postprocess
                # x.shape = (n+p,), so need to increase the dimension (n+p,1)
                xi = x[0:n,np.newaxis]
                beta_all[:,i:i+1] = x[n:n+p,np.newaxis]
                #from IPython.core.debugger import Tracer; debug_here = Tracer()
                s_hat_all[:,i:i+1] = np.dot(self.X,beta_all[:,i:i+1]) + np.dot(Z,np.dot(HZ.T,xi))

            # evaluate solutions from LM 
            print('evaluate solutions')
            if self.parallel:
                simul_obs_all = self.ParallelForwardSolve(s_hat_all)
            else:
                for i in range(nopts):
                    if i == 0:
                        simul_obs_all = self.ForwardSolve(x)
                    else:
                        simul_obs_all = np.concatenate((simul_obs_all, self.ForwardSolve(x)), axis=1)

            assert(np.size(simul_obs_all,1) == nopts)
            # evaluate objective values and select best value
            obj_best = 1.e+20
            print('%d objective value evaluations' % nopts)
                
            for i in range(nopts):
                if self.objeval: # If true, we do accurate computation
                    obj = self.ObjectiveFunction(s_hat_all[:,i:i+1], beta_all[:,i:i+1], simul_obs_all[:,i:i+1],0) 
                else: # we compute through PCGA approximation
                    obj = self.ObjectiveFunction(s_hat_all[:,i:i+1], beta_all[:,i:i+1], simul_obs_all[:,i:i+1],1) 
                
                if obj < obj_best: 
                    print('%d-th solution obj %e (alpha %f)' % (i,obj,alpha[i]))
                    s_hat = s_hat_all[:,i:i+1]
                    beta = beta_all[:,i:i+1]
                    simul_obs_new = simul_obs_all[:,i:i+1]
                    obj_best = obj
        else:
            #Create matrix context
            def mv(v):
                return np.concatenate((np.dot(HZ,np.dot(HZ.T,v[0:n])) + np.dot(self.R,v[0:n]) + np.dot(HX,v[n:n+p]), np.dot(HX.T,v[0:n])),axis = 0)
            # Matrix handle
            Afun = LinearOperator( (n+p,n+p), matvec=mv ,dtype = 'd')

            b = np.zeros((n+p,1), dtype = 'd')
            b[:n] = self.obs[:] - simul_obs + Hs[:]
        
            callback = Residual()
        
            #Residua and maximum iterations	
            itertol = 1.e-10 if not 'iterative_tol' in self.params else self.params['iterative_tol']
            maxiter = n if not 'iterative_maxiter' in self.params else self.params['iterative_maxiter']
        
            if self.P is None:
                x, info = minres(Afun, b, tol = itertol, maxiter = maxiter, callback = callback)
                if self.verbose: print("-- Number of iterations for minres %g" %(callback.itercount()))
            
            else:
                restart = 50 if 'gmresrestart' not in self.params else self.params['gmresrestart']
                x, info = gmres(Afun, b, tol = itertol, restart = restart, maxiter = maxiter, callback = callback, M = self.P)
                if self.verbose: print("-- Number of iterations for gmres %g" %(callback.itercount()))
            
            assert(info == 0)

            #Extract components and postprocess
            # x.shape = (n+p,) so make it to (n+p,1)
            xi = x[0:n,np.newaxis]
            beta = x[n:n+p,np.newaxis]
            #from IPython.core.debugger import Tracer; debug_here = Tracer()
            #debug_here()
            s_hat = np.dot(self.X,beta) + np.dot(Z,np.dot(HZ.T,xi))

            print('- evaluate new solution')
            simul_obs_new = self.ForwardSolve(s_hat)            

        return s_hat, beta, simul_obs_new

    def LinearInversionKnownMean(self, s_cur, beta = 0.):
		# will be implemented later.
        return NotImplementedError
    
    def LinearIteration(self, s_cur, simul_obs):
        
        direct = self.direct
        precond = self.precond
            
        #Solve geostatistical system
        if direct:
            s_hat, beta, simul_obs_new = self.DirectSolve(s_cur, simul_obs, recompute = True)
        else:
            #Construct preconditioner	
            if precond:	self.ConstructPreconditioner()

            s_hat, beta, simul_obs_new = self.IterativeSolve(s_cur, simul_obs, precond = precond)
        
        if self.objeval: # If true, we do accurate computation
            obj = self.ObjectiveFunction(s_hat, beta, simul_obs_new,0) 
        else: # we compute through PCGA approximation
            obj = self.ObjectiveFunction(s_hat, beta, simul_obs_new,1) 

        return s_hat, beta, simul_obs_new, obj
    
    def GaussNewton(self, savefilename = None):
        '''
        will save results if savefilename is provided
        '''
        m = self.m
        s_init = self.s_init
        #s_past = np.zeros((m,1), dtype = 'd')
        #s_cur = np.zeros((m,1), dtype = 'd')

        #plotting = False if 'plotting' not in self.params else self.params['plotting']
        
        maxiter = self.params['maxiter']
        restol  = self.params['restol']
        iter_cur   = maxiter

        obj = 1.0e+20
        
        #self.Q.BuildPreconditioner(k = 100)
        res = 1.
        
        print('##### 4. Start PCGA Inversion #####')
        print('-- evaluate initial solution')
        
        simul_obs_init = self.ForwardSolve(s_init)
        self.simul_obs_init = simul_obs_init
        
        print('norm(obsdiff): %g' % np.linalg.norm(simul_obs_init-self.obs))
        simul_obs = np.copy(simul_obs_init)
        s_cur = np.copy(s_init)
        s_past = np.copy(s_init)

        for i in range(maxiter):
            start = time()
            print("***** Iteration %d ******" % (i+1))
            s_cur, beta_cur, simul_obs_cur, obj = self.LinearIteration(s_past, simul_obs)
            
            print("- time for iteration %d is %g sec" %((i+1), round(time()-start)))
            
            res = np.linalg.norm(s_past-s_cur)/np.linalg.norm(s_past)

            if obj < self.obj_best:
                self.obj_best = obj
                self.s_best = s_cur
                self.simul_obs_best = simul_obs_cur

            obs_diff = np.linalg.norm(simul_obs_cur-self.obs)

            if self.s_true is not None:            
                err = np.linalg.norm(s_cur-self.s_true)/np.linalg.norm(self.s_true)
                print("- iteration %d: relative residual is %g, objective function is %e, error is %g, and norm(obs mismatch) is %g" %((i+1), res, obj, err, obs_diff))
            else:
                print("- iteration %d: relative residual is %g, objective function is %e, and norm(obs mismatch) is %g" %((i+1), res, obj, obs_diff))

            if res < restol:
                iter_cur = i + 1
                break

            s_past = np.copy(s_cur)
            simul_obs = np.copy(simul_obs_cur)

        #return s_cur, beta_cur, simul_obs, iter_cur
        return self.s_best, self.simul_obs_best, self.iter_best, iter_cur

    def Run(self):
        if self.Q is None:
            self.ConstructCovariance(method = self.covariance_matvec, kernel = self.kernel, xmin = self.xmin, xmax = self.xmax, N= self.N, theta = self.theta)
    
        if self.priorU is None or self.priord is None:
            self.ComputePriorEig()

        s_hat, simul_obs, iter_best, iter_final = self.GaussNewton()

        return s_hat, simul_obs, iter_best, iter_final

    """
        functions below have not been implmented yet
    """
    def ComputePosteriorDiagonalEntriesDirect(self, recompute = True):
        """		
		Works best for small measurements O(100)
		to be implemented
        """
        return
    
    def FssInvDiag(self, recompute = True):
        """		
        Works best for small measurements O(100)
		to be implemented
        """
        return
    
    def GeneralizedEig(self):
        """
        Saibaba, Lee and Kitanidis, 2016
		to be implemented
        """
        return
    
    def CompareSpectrum(self, filename):
        """
        Compare spectrum of Hred, Qinv, and the combination of the two.
		to be implemented
        """
        return
    
    def GenerateUnconditionalRealizations(self):
        return
    
    def ConstructPreconditioner(self):
        '''
        Lee et al., 2016
        '''
        m = self.m
        n = self.n
        p = self.p
        #Create preconditioner context
		#P = Preconditioner(self)
		#self.P = aslinearoperator(P)
        return

    def Uncertainty(self, **kwargs):
        return

#if __name__ == '__main__':
