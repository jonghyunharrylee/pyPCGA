from time import time
from math import isnan, sqrt
import numpy as np
import sys,os
from covariance.mat import *
from inspect import getsource

from scipy.sparse.linalg import gmres, minres, svds # IterativeSolve
from scipy.sparse.linalg import LinearOperator # Matrix-free IterativeSolve
from IPython.core.debugger import Tracer; debug_here = Tracer()
#from pdb import set_trace

# todo remove either Z or d*U
__all__ = ['PCGA']

class PCGA:
    """
    Solve inverse problem with PCGA (approx to quasi-linear method)
    every values are represented as 2D np array
    """

    def __init__(self, forward_model, s_init, pts, params, s_true = None, obs = None, obs_true = None, X = None):
        print('##### PCGA Inversion #####')
        print('##### 1. Initialize forward and inversion parameters')

        ##### Forward Model
        # forward solver setting should be done externally as a blackbox
        self.forward_model = forward_model
        
        # Grid points (for Dense, Hmatrix and FMM)
        self.pts = pts # no need for FFT. Will use this later
        
        # Store parameters
        self.params = params
        
        ##### Inversion Setting
        # inversion setting
        self.m   = np.size(s_init,0) 
        self.s_init = np.array(s_init)
        
        if s_true is None:
            self.s_true = None
        else:
            self.s_true = np.array(s_true)
            if self.m != np.size(s_true,0):
                raise ValueError("self.m == np.size(s_true,0)")

        # Observations
        if obs is None:
            self.n = 0
        else:
            if obs.ndim == 1:
                obs = np.array(obs)
                self.obs = obs.reshape(-1,1)
            elif obs.ndim == 2:
                if obs.shape[1] != 1:
                    raise ValueError("obs should be n by 1 array")
                self.obs = np.array(obs)
            else:
               raise ValueError("obs should be n by 1 array")
            self.n = self.obs.shape[0]

        self.obs_true = np.array(obs_true)  # observation without noise for synthetic problems

        # keep track of some values (best, init)
        self.s_best = None
        self.beta_best = None
        self.simul_obs_best = None
        self.iter_best = 0
        self.obj_best = 1.e+20
        self.simul_obs_init = None
        self.objvals = []
        self.Q2_cur = None
        self.cR_cur = None
        self.Q2_best = None
        self.cR_best = None
        self.i_best = None

        self.iter_save = False if 'iter_save' not in self.params else self.params['iter_save']

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
        
        ##### Prior Information
        #Covariance matrix
        self.Q = None
        self.matvec = params['matvec'] # method for mat-vector multiplication
        self.prior_std = params['prior_std'] # prior std

        #Eigenvalues and eigenvectors of the prior covariance matrix
        self.priorU = None
        self.priord = None

        if self.matvec == 'FFT':
            self.xmin = params['xmin']
            self.xmax = params['xmax']
            self.N = params['N']
            self.prior_cov_scale = params['prior_cov_scale']
            self.kernel = params['kernel']
            
            if np.size(self.xmin,0) != np.size(self.xmax,0):
                raise ValueError("np.size(xmin,0) != np.size(xmax,0)")
            if np.size(self.xmin,0) != np.size(self.N,0):
                raise ValueError("np.size(xmin,0) != np.size(N,0)")
            if np.size(self.xmin,0) != np.size(self.prior_cov_scale, 0):
                raise ValueError("np.size(xmin,0) != np.size(theta,0)")
        elif self.matvec == 'Dense':
            self.prior_cov_scale = params['prior_cov_scale']
            self.kernel = params['kernel']
        else: # currently not implemented - we will support Hmatrix and FMM
            self.prior_cov_scale = params['prior_cov_scale']
            self.kernel = params['kernel']
            self.priord = params['priord']
            self.priorU = params['priorU']
            # change below to one using LinearOperator
            self.Q = CovarianceMatrixbyUd(self.priord,self.priorU)

        # number of principal components, default 5 times # of cores
        self.n_pc = 5*self.ncores if 'n_pc' not in params else params['n_pc'] 
        
        #Noise covariance
        if 'R' in params:
            if not isinstance(params['R'],(float,np.ndarray)):
                raise ValueError('provide R as float or numpy array')
            
            self.R = np.array(params['R'])
            
            if self.R.ndim == 0 or self.R.ndim == 1:
                self.R = np.array(params['R']).reshape(-1) # convert to 1d array (1,)
            elif self.R.ndim == 2:
                self.R = np.array(params['R']).reshape(-1,1) # convert to 2d n x 1 array
            else:
                raise ValueError('R should be scalar or 2d array')

            if (self.R <= 0).all():
                raise ValueError('R should be positive (R>0)')
            
            self.sqrtR = np.sqrt(self.R)
            self.invsqrtR = 1./np.sqrt(self.R)
            self.invR = 1./self.R
        else:
            raise ValueError('You should provide R')

        # posterior cov computation
        # we only support the computation of post variance (diagonals of posterior covariance matrix) for now
        #Eigenvalues and eigenvectors of the posterir covariance matrix - will be implemented later
        #self.postU = None
        #self.postd = None

        self.post_diagv = self.prior_std**2 # will be updated if post_cov == "diag"

        #self.post_cov = 'gep'
        if 'post_cov' in params:
            self.post_cov = params['post_cov']
        else:
            self.post_cov = False

        if self.post_cov or self.post_cov == "diag":
            if self.n <= 1000: # choose arbitrary..
                self.post_diag_direct = True
            else:
                self.post_diag_direct = False
            if 'post_diag_direct' in params:
                self.post_diag_direct = params['post_diag_direct']
            if self.post_diag_direct:
                print("WARNING!! : you chose to perform direct posterior variance analysis, which would take forever! plz use post_diag_direct=True")
        #CrossCovariance computed by Jacobian-free low-rank FD approximation if you want to save..
        if 'JacSave' in params:
            self.JacSave = params['JacSave']
        else:
            self.JacSave = False

        if self.post_cov or self.post_cov == "diag":
            self.JacSave = True

        if self.JacSave:
            self.HX = None
            self.HZ = None
            self.Hs = None

        # Define Drift (or Prior) functions 
        self.p   = 0 # drift parameter (to be determined)
        if X is not None:
            self.X = X
            self.p = np.size(self.X,1)
        else:
            if 'drift' in params:
                self.DriftFunctions(params['drift'])
            else:
                # add constant drift by default
                params['drift'] = 'constant'
                self.DriftFunctions(params['drift'])
        
        ##### Matrix Solver (Saddle Point/Cokkriging System)
        # Matrix solver - default : False (iterative)
        self.direct = False if 'direct' not in self.params else self.params['direct']
        direct = self.direct

        # default : False!
        self.precond = False if 'precond' not in self.params else self.params['precond']

        #Define Preconditioner
        if self.precond:
            self.P = None
            self.Psi_U = None
            self.Psi_sigma = None

        ##### Optimization
        # objetive function evaluation, either exact or approximate
        if 'objeval' in params:
            self.objeval = params['objeval']
        else:
            self.params['objeval'] = False
            self.objeval = False

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

        self.linesearch = True if 'linesearch' not in self.params else self.params['linesearch']
        
        if self.linesearch:
            self.nopts_LS = self.ncores if 'nopts_LS' not in self.params else self.params['nopts_LS']

        self.maxiter = 10 if 'maxiter' not in self.params else self.params['maxiter'] # maximum iteration
        self.restol  = 1e-2 if 'restol'  not in self.params else self.params['restol'] # rel tol
        
        # PCGA parameters (purturbation size)
        self.precision = 1.e-8 if 'precision' not in self.params else self.params['precision']

		#Printing verbose
        self.verbose = False if 'verbose' not in self.params else self.params['verbose']
        self.forward_model_verbose = True if 'forward_model_verbose' not in self.params else self.params['forward_model_verbose']

        #Generate measurements when obs is not provided AND s_true is given
        if s_true is not None and obs is None:
            self.CreateSyntheticData(s_true, noise = True)

        print("------------ Inversion Parameters -------------------------")
        print("   Number of unknowns                            : %d" % (self.m))
        print("   Number of observations                        : %d" % (self.n))
        print("   Number of principal components (n_pc)         : %d" % (self.n_pc))
        print("   Prior model                                   : %s" % (getsource(self.kernel)))
        print("   Prior variance                                : %e" % (self.prior_std ** 2))
        print("   Prior scale (correlation) parameter           : %s" % (self.prior_cov_scale))
        print("   Posterior cov computation                     : %s" % (self.post_cov))
        if self.post_cov:
            if self.post_diag_direct:
                print("   Posterior posterior vaiance compuation        : Direct")
            else:
                print("   Posterior posterior vaiance compuation        : Fast")
        print("   Number of CPU cores (n_core)                  : %d" % (self.ncores))
        print("   Maximum GN iterations                         : %d" % (self.maxiter))
        print("   machine precision (delta = sqrt(precision)    : %e" % (self.precision))
        print("   Tol for iterations (norm(sol_diff)/norm(sol)) : %e" % (self.restol))
        print("   Levenberg-Marquardt                           : %s" % (self.LM))
        print("   Line search                                   : %s" % (self.linesearch))
        print("-----------------------------------------------------------")

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
            raise NotImplementedError
        else: # constant drift
            self.p = 1
            self.X = np.ones((self.m,1),dtype ='d')/np.sqrt(self.m)
        return

    def ConstructCovariance(self, method, kernel, **kwargs):
        print('##### 2. Construct Prior Covariance Matrix')
        start = time()
        self.Q = CovarianceMatrix(method, self.pts, kernel, **kwargs)
        print('- time for covariance matrix construction (m = %d) is %g sec' % (self.m, round(time()-start)))
        return
    
    def ComputePriorEig(self, n_pc=None):
        '''
        Compute Eigenmodes of Prior Covariance 
        '''
        print('##### 3. Eigendecomposition of Prior Covariance')
        m = self.m
        n = self.n
        p = self.p
        
        if n_pc is None:
            n_pc = self.n_pc
        
        if self.Q is None: # have to input Q 
            raise ValueError("Q should be assigned")
        
        method = 'arpack' if 'precondeigen' not in self.params else self.params['precondeigen']
        
        #twopass = False if not 'twopass' in self.params else self.params['twopass']
        start = time()
        if method == 'arpack':
            from scipy.sparse.linalg import eigsh
            self.priord, self.priorU = eigsh(self.Q, k = n_pc)
            self.priord = self.priord[::-1]
            self.priord = self.priord.reshape(-1,1) # make a column vector
            self.priorU = self.priorU[:,::-1]
        #elif method == 'randomized':
        #    # randomized method to be implemented!
        #    from eigen import RandomizedEIG
        #    self.priorU, self.priorD = RandomizedEIG(self.Q, k =k, twopass = twopass)
        else:
            raise NotImplementedError

        print('- time for eigendecomposition with k = %d is %g sec' % (n_pc, round(time()-start)))

        if (self.priord > 0).sum() < n_pc:
            self.n_pc = (self.priord > 0).sum()
            self.priord = self.priord[:self.n_pc,:]
            self.priorU = self.priorU[:,:self.n_pc]
            print("Warning: n_pc changed to %d for positive eigenvalues" % (self.n_pc))

        print('- 1st eigv : %g, %d-th eigv : %g, ratio: %g' % (self.priord[0],self.n_pc,self.priord[-1],self.priord[-1]/self.priord[0]))
        return

    def ForwardSolve(self,s):
        '''
        provide additional settings for your function forward_model 
        '''

        par = False
        if self.forward_model_verbose:
            simul_obs = self.forward_model(s, par)
        else:
            with HiddenPrints():
                simul_obs = self.forward_model(s,par)
        return simul_obs

    def ParallelForwardSolve(self,s):
        '''
        provide additional settings for your function forward_model running in parallel
        '''
        par = True
        if self.forward_model_verbose:
            simul_obs_parallel = self.forward_model(s, par, ncores=self.ncores)
        else:
            with HiddenPrints():
                simul_obs_parallel = self.forward_model(s,par, ncores = self.ncores)

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
                    raise ValueError('delta is zero? - plz check your s_init is within a reasonable range')

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
        
        if np.size(simul_obs_purturbation,1) != nruns:
            raise ValueError("size of simul_obs_purturbation (%d,%d) is not nruns %d" % (simul_obs_purturbation.shape[0], simul_obs_purturbation.shape[1], nruns))

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
                raise ValueError('plz specify true solution')
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
            obs = obs_true + np.multiply(self.sqrtR,np.random.randn(self.n,1))
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
            invZs = np.multiply(1./np.sqrt(self.priord),np.dot(self.priorU.T,smxb))
            obj = 0.5*np.dot(ymhs.T,np.divide(ymhs,self.R)) + 0.5*np.dot(invZs.T,invZs)
        else:
            invQs = self.Q.solve(smxb)
            obj = 0.5*np.dot(ymhs.T,np.divide(ymhs,self.R)) + 0.5*np.dot(smxb.T,invQs)
        return obj

    def ObjectiveFunctionNoBeta(self, s_cur, simul_obs, approx = True):
        """
            marginalized objective w.r.t. beta
            0.5(y-h(s))^TR^{-1}(y-h(s)) + 0.5*(s-Xb)^TQ^{-1}(s-Xb)
        """
        if simul_obs is None:
            simul_obs = self.ForwardSolve(s_cur)
        
        priord  = self.priord
        priorU  = self.priorU
        X = self.X
        p = self.p

        ymhs = self.obs - simul_obs
        
        if approx:
            invZs = np.multiply(1./np.sqrt(priord),np.dot(priorU.T,s_cur))
            invZX = np.multiply(1./np.sqrt(priord),np.dot(priorU.T,X))
            XTinvQs = np.dot(invZX.T,invZs)
            XTinvQX = np.dot(invZX.T,invZX)
            tmp = np.linalg.solve(XTinvQX, XTinvQs) # inexpensive solve p by p where p <= 3, usually p = 1 (scalar devision)
            obj = 0.5*np.dot(ymhs.T,np.divide(ymhs,self.R)) + 0.5*(np.dot(invZs.T,invZs) - np.dot(XTinvQs.T,tmp))
        else:
            invQs = self.Q.solve(s_cur) #size (m,)
            invQX = self.Q.solve(X) # size (m,p)
            
            XTinvQs = np.dot(X.T,invQs) # size (p,)
            XTinvQX = np.dot(X.T,invQX) # size (p,p)
            
            if p == 1:
                tmp = XTinvQs/XTinvQX
            else:
                tmp = np.linalg.solve(XTinvQX, XTinvQs) # inexpensive solve p by p where p <= 3, usually p = 1 (scalar devision)
            
            obj = 0.5*np.dot(ymhs.T,np.divide(ymhs,self.R)) + 0.5*(np.dot(s_cur.T,invQs) - np.dot(XTinvQs.T,tmp))
        return obj

    def JacMat(self, s_cur, simul_obs, Z):
        
        m = self.m
        n = self.n
        p = self.p

        n_pc = self.n_pc
        precision = self.precision

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

        # compute the pre-posterior data space
        if p == 1:
            U_data = HX/np.linalg.norm(HX) 
        elif p > 1:
            from scipy.linalg import svd
            U_data = svd(HX,full_matrices=False,compute_uv=True,lapack_driver='gesdd')[0]
        else: # point prior
            raise NotImplementedError
        return HX, HZ, Hs, U_data

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
        R = self.R

        if simul_obs is None:
            simul_obs = self.ForwardSolve(s_cur)
        
        Z = np.zeros((m,n_pc), dtype ='d')
        for i in range(n_pc):
            Z[:,i:i+1] = np.dot(sqrt(self.priord[i]),self.priorU[:,i:i+1]) # use sqrt to make it scalar

        # Compute Jacobian-Matrix products
        start1 = time()
        HX, HZ, Hs, U_data = self.JacMat(s_cur, simul_obs, Z)

        # Compute eig(P*HQHT*P) approximately by svd(P*HZ)
        start2 = time()
        sqrtGDCov = GeneralizedSQRTCovarianceMatrix(U_data,HZ) # sqrt of generalized data covariance
        sigma_cR = svds(sqrtGDCov, k= min(n-p,n_pc), which='LM', maxiter = n, return_singular_vectors=False)
        print("computed Jacobian-Matrix products in : %f secs, eig. val. of generalized data covariance : %f secs" % (start1 - start2,time()-start2))

        # Construct HQ directly 
        HQ = np.dot(HZ,Z.T) 
        
        if self.LM:
            print('Solve geostatistical inversion problem (co-kriging, saddle point systems) with Levenberg-Marquardt')
            nopts = self.nopts_LM 
            alpha = 10**(np.linspace(0.,np.log10(self.alphamax_LM),nopts))
        else:
            print('Solve geostatistical inversion problem (co-kriging, saddle point systems)')
            nopts = 1
            alpha = np.array([1.0])

        beta_all = np.zeros((p,nopts),'d')
        s_hat_all = np.zeros((m,nopts),'d')
        Q2_all = np.zeros((1,nopts),'d')
        cR_all = np.zeros((1,nopts),'d')
            
        for i in range(nopts): # sequential evaluation for now
            # Construct Psi directly 
            Psi = np.dot(HZ,HZ.T) + np.multiply(np.multiply(alpha[i],R),np.eye(n,dtype='d'))

            #Create matrix system and solve it
            # cokriging matrix
            A = np.zeros((n+p,n+p),dtype='d')
            b = np.zeros((n+p,1),dtype='d')

            A[0:n,0:n] = np.copy(Psi)
            A[0:n,n:n+p] = np.copy(HX)
            A[n:n+p,0:n] = np.copy(HX.T)

            # Ax = b, b = obs - h(s) + Hs 
            b[:n] = self.obs[:] - simul_obs + Hs[:]

            x = np.linalg.solve(A, b)

            ##Extract components and return final solution
            xi = x[0:n,np.newaxis]
            beta_all[:,i:i+1] = x[n:n+p,np.newaxis]
            s_hat_all[:,i:i+1] = np.dot(self.X,beta_all[:,i:i+1]) + np.dot(HQ.T,xi)
            Q2_all[:,i:i+1] = np.dot(b[:n].T,xi)/(n-p)

            tmp_cR = np.zeros((n-p,1),'d')
                
            tmp_cR[:] = np.multiply(alpha[i],self.R) 
            tmp_cR[:sigma_cR.shape[0]] = tmp_cR[:sigma_cR.shape[0]] + (sigma_cR[:,np.newaxis])**2
            cR_all[:,i:i+1] = Q2_all[:,i:i+1]*np.exp(np.log(tmp_cR).sum()/(n-p))
                
        # parallel
        if self.LM:
            print("evaluate LM solutions")
        else:
            print("evaluate the best solution")

        if self.parallel:
            simul_obs_all = self.ParallelForwardSolve(s_hat_all)
        else:
            for i in range(nopts):
                if i == 0:
                    simul_obs_all = self.ForwardSolve(s_hat_all[:,i:i+1])
                else:
                    simul_obs_all = np.concatenate((simul_obs_all, self.ForwardSolve(s_hat_all[:,i:i+1])), axis=1)

        if np.size(simul_obs_all,1) != nopts:
            raise ValueError("np.size(simul_obs_all,1) != nopts")

        obj_best = 1.e+20
        print('%d objective value evaluations' % nopts)
        for i in range(nopts):
            if self.objeval: # If true, we do accurate computation
                obj = self.ObjectiveFunction(s_hat_all[:,i:i+1], beta_all[:,i:i+1], simul_obs_all[:,i:i+1],approx = False) 
            else: # we compute through PCGA approximation
                obj = self.ObjectiveFunction(s_hat_all[:,i:i+1], beta_all[:,i:i+1], simul_obs_all[:,i:i+1],approx = True) 
                
            if obj < obj_best: 
                print('%d-th solution obj %e (alpha %f)' % (i,obj,alpha[i]))
                s_hat = s_hat_all[:,i:i+1]
                beta = beta_all[:,i:i+1]
                simul_obs_new = simul_obs_all[:,i:i+1]
                obj_best = obj
                self.Q2_cur = Q2_all[:,i:i+1]
                self.cR_cur = cR_all[:,i:i+1]

        return s_hat, beta, simul_obs_new

    def IterativeSolve(self, s_cur, simul_obs = None, precond = False):
        '''
        Iterative Solve
        '''
        m = self.m
        n = self.n
        p = self.p
        n_pc = self.n_pc
        R = self.R

        Z = np.zeros((m,n_pc), dtype ='d')
        for i in range(n_pc):
            Z[:,i:i+1] = np.dot(sqrt(self.priord[i]),self.priorU[:,i:i+1]) # use sqrt to make it scalar

        if simul_obs is None:
            simul_obs = self.ForwardSolve(s_cur)

        # Compute Jacobian-Matrix products
        start1 = time()
        HX, HZ, Hs, U_data = self.JacMat(s_cur, simul_obs, Z)

        start2 = time()
        
        # Compute Q2/cR for covariance model validation
        if R.shape[0] == 1: # Compute eig(P*(HQHT+R)*P) approximately by svd(P*HZ)**2 + R if R is single number
            def mv(v):
                # P*HZ*x = ((I-(U_data*U_data.T))*HZ)*x '''
                tmp = np.dot(HZ,v)
                y = tmp - np.dot(U_data,np.dot(U_data.T,tmp))
                return y
            def rmv(v):
                return np.dot(HZ.T,v) - np.dot(HZ.T,np.dot(U_data,np.dot(U_data.T,v)))

            # Matrix handle for sqrt of Generalized Data Covariance
            sqrtGDCovfun = LinearOperator( (n,n_pc), matvec=mv, rmatvec = rmv, dtype = 'd')
            sigma_cR = svds(sqrtGDCovfun, k= min(n-p,n_pc-1), which='LM', maxiter = n, return_singular_vectors=False)
            print("computed Jacobian-Matrix products in %f secs, eig. val. of generalized data covariance : %f secs (%8.2e, %8.2e, %8.2e)" % (start2 - start1, time()-start2,sigma_cR[0],sigma_cR.min(),sigma_cR.max()))
        else: # Compute eig(P*(HQHT+R)*P) approximately by svd(P*(HZ*HZ' + R)*P) # need to do for each alpha[i]*R
            print("computed Jacobian-Matrix products in %f secs" % (start2 - start1))

        # preconditioner construction
        if self.precond != False:
            # GHEP : HQHT u = lamdba R u => u = R^{-1/2} y 
            if R.shape[0] == 1:
                # n by n_pc
                def pmv(v):
                    return np.multiply(self.invsqrtR,np.dot(HZ,v))
                def prmv(v):
                    return np.dot(HZ.T,np.multiply(self.invsqrtR,v))
            else:
                # n by n
                def pmv(v):
                    return np.multiply(self.invsqrtR.reshape(v.shape),np.dot(HZ,np.multiply(self.invsqrtR.reshape(v.shape),v)))
                def prmv(v):
                    return pmv(v)

            # Matrix handle for sqrt of Data Covariance
            sqrtDataCovfun = LinearOperator( (n,n_pc), matvec=pmv, rmatvec = prmv, dtype = 'd')

            [Psi_U,Psi_sigma,Psi_V] = svds(sqrtDataCovfun, k= min(n,n_pc)-1, which='LM', maxiter = n, return_singular_vectors='u')

            Psi_U = np.multiply(self.invsqrtR,Psi_U)
            if R.shape[0] == 1:
                Psi_sigma = Psi_sigma**2 # because we use svd(HZ) instead of svd(HQHT+R)
            index_Psi_sigma = np.argsort(Psi_sigma)
            index_Psi_sigma = index_Psi_sigma[::-1]
            Psi_sigma = Psi_sigma[index_Psi_sigma]
            Psi_U = Psi_U[:,index_Psi_sigma]
            Psi_U = Psi_U[:,Psi_sigma > 0]
            Psi_sigma =  Psi_sigma[Psi_sigma > 0]

            self.Psi_sigma = Psi_sigma
            self.Psi_U = Psi_U

        if self.LM:
            print('Solve saddle point (co-kriging) systems with Levenberg-Marquardt')
            nopts = self.nopts_LM
            alpha = 10**(np.linspace(0.,np.log10(self.alphamax_LM),nopts))
        else:
            print('Solve saddle point (co-kriging) system')
            nopts = 1
            alpha = np.array([1.0])

        beta_all = np.zeros((p,nopts),'d')
        s_hat_all = np.zeros((m,nopts),'d')
        Q2_all = np.zeros((1,nopts),'d')
        cR_all = np.zeros((1,nopts),'d')
            
        for i in range(nopts): # this is sequential for now
            
            #Create matrix context
            if R.shape[0] == 1:
                def mv(v):
                    return np.concatenate(( (np.dot(HZ,np.dot(HZ.T,v[0:n])) + np.multiply(np.multiply(alpha[i],R),v[0:n]) + np.dot(HX,v[n:n+p])) , (np.dot(HX.T,v[0:n])) ),axis = 0)
            else:
                def mv(v):
                    return np.concatenate(( (np.dot(HZ,np.dot(HZ.T,v[0:n])) + np.multiply(np.multiply(alpha[i],R.reshape(v[0:n].shape)),v[0:n]) + np.dot(HX,v[n:n+p])) , (np.dot(HX.T,v[0:n])) ),axis = 0)

            # Matrix handle
            Afun = LinearOperator( (n+p,n+p), matvec=mv, rmatvec = mv, dtype = 'd')

            b = np.zeros((n+p,1), dtype = 'd')
            b[:n] = self.obs[:] - simul_obs + Hs[:]
        
            callback = Residual()
        
            #Residual and maximum iterations	
            itertol = 1.e-10 if not 'iterative_tol' in self.params else self.params['iterative_tol']
            solver_maxiter = m if not 'iterative_maxiter' in self.params else self.params['iterative_maxiter']

            # construction preconditioner
            if self.precond:

                if R.shape[0] == 1:

                    def invPsi(v):
                        Dvec = np.divide( (1./alpha[i] * Psi_sigma), ((1./alpha[i]) * Psi_sigma + 1))
                        Psi_U_i = np.multiply((1. / sqrt(alpha[i])),Psi_U)
                        Psi_UTv = np.dot(Psi_U_i.T, v)
                        return np.multiply(np.multiply((1./alpha[i]),self.invR),v) - np.dot(Psi_U_i, np.multiply(Dvec[:Psi_U_i.shape[1]].reshape(Psi_UTv.shape), Psi_UTv))

                    def Pmv(v):
                        invPsiv = invPsi(v[0:n])
                        S = np.dot(HX.T, invPsi(HX)) # p by p matrix
                        invSHXTinvPsiv = np.linalg.solve(S,np.dot(HX.T,invPsiv))
                        invPsiHXinvSHXTinvPsiv = invPsi(np.dot(HX,invSHXTinvPsiv))
                        #return np.concatenate( ((invPsiv - invPsiHXinvSHXTinvPsiv + invPsiHXinvSv1), (invSHXTinvPsiv - Sv1)),axis = 0)
                        return np.concatenate(
                            ((invPsiv - invPsiHXinvSHXTinvPsiv), (invSHXTinvPsiv)), axis=0)
                else:
                    return NotImplemented
                    
                P = LinearOperator( (n+p,n+p), matvec=Pmv, rmatvec = Pmv, dtype = 'd')

                restart = 50 if 'gmresrestart' not in self.params else self.params['gmresrestart']
                x, info = gmres(Afun, b, tol=itertol, restart=restart, maxiter=solver_maxiter, callback=callback, M=P)
                if self.verbose: print("-- Number of iterations for gmres %g" %(callback.itercount()))
                if info != 0: # if not converged
                    x, info = minres(Afun, b, x0 = x, tol=itertol, maxiter=solver_maxiter, callback=callback, M=P)
                    if self.verbose: print("-- Number of iterations for minres %g and info %d" %(callback.itercount(),info))
            else:
                x, info = minres(Afun, b, tol = itertol, maxiter = solver_maxiter, callback = callback)
                if self.verbose: print("-- Number of iterations for minres %g" %(callback.itercount()))

                if info != 0:
                    x, info = gmres(Afun, b, x0=x, tol=itertol, maxiter=solver_maxiter, callback=callback)
                    print("-- Number of iterations for gmres: %g, info: %d, tol: %f" % (callback.itercount(),info, itertol))

            #Extract components and postprocess
            # x.shape = (n+p,), so need to increase the dimension (n+p,1)
            xi = x[0:n,np.newaxis]
            beta_all[:,i:i+1] = x[n:n+p,np.newaxis]
            #from IPython.core.debugger import Tracer; debug_here = Tracer()
            s_hat_all[:,i:i+1] = np.dot(self.X,beta_all[:,i:i+1]) + np.dot(Z,np.dot(HZ.T,xi))
            if self.verbose: print("%d - min(s): %g, max(s) :%g" % (i,s_hat_all[:,i:i+1].min(),s_hat_all[:,i:i+1].max()))
            Q2_all[:,i:i+1] = np.dot(b[:n].T,xi)/(n-p)
                
            # model validation, predictive diagnostics cR/Q2
            if R.shape[0] == 1:
                tmp_cR = np.zeros((n-p,1),'d')
                tmp_cR[:] = np.multiply(alpha[i],R)
                tmp_cR[:sigma_cR.shape[0]] = tmp_cR[:sigma_cR.shape[0]] + (sigma_cR[:,np.newaxis])**2
            else:
                # need to find efficient way to compute cR once
                def mv(v):
                    # P*(HZ*HZ.T + R)*P*x = P = (I-(U_data*U_data.T))
                    Pv =  v - np.dot(U_data,np.dot(U_data.T,v))
                    RPv = np.multiply(R,Pv)
                    PRPv = np.dot(U_data,np.dot(U_data.T,RPv))
                    HQHTPv = np.dot(HZ,np.dot(HZ.T,RPv))
                    PHQHTPv = HQHTPv - np.dot(U_data,np.dot(U_data.T,HQHTPv))
                    return PHQHTPv + PRPv
                def rmv(v):
                    return mv(v) # symmetric matrix
                
                # Matrix handle for sqrt of Generalized Data Covariance
                sqrtGDCovfun = LinearOperator( (n,n_pc), matvec=mv, rmatvec = rmv, dtype = 'd')
                sigma_cR = svds(sqrtGDCovfun, k= min(n-p,n_pc-1), which='LM', maxiter = n, return_singular_vectors=False)
                tmp_cR[:sigma_cR.shape[0]] = sigma_cR[:,np.newaxis] # need to test this later
                
            cR_all[:,i:i+1] = Q2_all[:,i:i+1]*np.exp(np.log(tmp_cR).sum()/(n-p))
            #debug_here()
        
        # evaluate solutions from LM 
        if self.LM:
            print('evaluate LM solutions')
            if self.parallel:
                simul_obs_all = self.ParallelForwardSolve(s_hat_all)
            else:
                for i in range(nopts):
                    if i == 0:
                        simul_obs_all = self.ForwardSolve(s_hat_all[:,i:i+1])
                    else:
                        simul_obs_all = np.concatenate((simul_obs_all, self.ForwardSolve(s_hat_all[:,i:i+1])), axis=1)
        else:
            simul_obs_all = self.ForwardSolve(s_hat_all)

        if np.size(simul_obs_all,1) != nopts:
            return ValueError("np.size(simul_obs_all,1) should be nopts")

        # evaluate objective values and select best value
        obj_best = 1.e+20
        if self.LM:
            print('%d objective value evaluations' % nopts)

        i_best = 0
        
        for i in range(nopts):
            if self.objeval: # If true, we do accurate computation
                obj = self.ObjectiveFunction(s_hat_all[:,i:i+1], beta_all[:,i:i+1], simul_obs_all[:,i:i+1],approx = False) 
            else: # we compute through PCGA approximation
                obj = self.ObjectiveFunction(s_hat_all[:,i:i+1], beta_all[:,i:i+1], simul_obs_all[:,i:i+1],approx = True) 
                
            if obj < obj_best: 
                print("%d-th solution obj %e (alpha %f)" % (i,obj,alpha[i]))
                s_hat = s_hat_all[:,i:i+1]
                beta = beta_all[:,i:i+1]
                simul_obs_new = simul_obs_all[:,i:i+1]
                obj_best = obj
                self.Q2_cur = Q2_all[:,i:i+1]
                self.cR_cur = cR_all[:,i:i+1]
                i_best = i
            
        if self.post_cov or self.post_cov == "diag":
            self.HZ = HZ
            self.HX = HX
            self.R_LM = np.multiply(alpha[i_best],self.R)

        self.i_best = i_best # keep track of best LM solution
        return s_hat, beta, simul_obs_new

    def LinearInversionKnownMean(self, s_cur, beta = 0.):
		# will be implemented later.
        raise NotImplementedError
    
    def LinearIteration(self, s_cur, simul_obs):
        
        direct = self.direct
        precond = self.precond
            
        #Solve geostatistical system
        if direct:
            s_hat, beta, simul_obs_new = self.DirectSolve(s_cur, simul_obs, recompute = True)
        else:
            #Construct preconditioner	
            #if precond:	self.ConstructPreconditioner()
            s_hat, beta, simul_obs_new = self.IterativeSolve(s_cur, simul_obs, precond = precond)
        
        if self.objeval: # If true, we do accurate computation
            obj = self.ObjectiveFunction(s_hat, beta, simul_obs_new,approx = False) 
        else: # we compute through PCGA approximation
            obj = self.ObjectiveFunction(s_hat, beta, simul_obs_new,approx = True) 

        return s_hat, beta, simul_obs_new, obj
    
    def LineSearch(self,s_cur,s_past):
        nopts = self.nopts_LS
        m = self.m

        s_hat_all = np.zeros((m,nopts),'d')
        delta = np.linspace(-0.1,1.1,nopts) # need to remove delta = 0 and 1
        
        for i in range(nopts):
            s_hat_all[:,i:i+1] = delta[i]*s_past + (1.-delta[i])*s_cur

        # parallel
        print('evaluate Line Search solutions')
        if self.parallel:
            simul_obs_all = self.ParallelForwardSolve(s_hat_all)
        else:
            for i in range(nopts):
                if i == 0:
                    simul_obs_all = self.ForwardSolve(x)
                else:
                    simul_obs_all = np.concatenate((simul_obs_all, self.ForwardSolve(x)), axis=1)

        # will change assert to valueerror
        assert(np.size(simul_obs_all,1) == nopts)
        obj_best = 1.e+20
        
        for i in range(nopts):
            start0 = time()
            if self.objeval: # If true, we do accurate computation
                obj = self.ObjectiveFunctionNoBeta(s_hat_all[:,i:i+1], simul_obs_all[:,i:i+1],0) 
            else: # we compute through PCGA approximation
                obj = self.ObjectiveFunctionNoBeta(s_hat_all[:,i:i+1], simul_obs_all[:,i:i+1],1) 
            
            if obj < obj_best: 
                print('%d-th solution obj %e (delta %f)' % (i,obj,delta[i]))
                s_hat = s_hat_all[:,i:i+1]
                simul_obs_new = simul_obs_all[:,i:i+1]
                obj_best = obj
    
        return s_hat, simul_obs_new, obj_best

    def GaussNewton(self):
        '''
        will save results if savefilename is provided
        '''
        m = self.m
        n = self.n
        s_init = self.s_init
        maxiter = self.maxiter
        restol = self.restol

        iter_cur   = maxiter

        obj = 1.0e+20
        
        #self.Q.BuildPreconditioner(k = 100)
        res = 1.
        
        print('##### 4. Start PCGA Inversion #####')
        print('-- evaluate initial solution')
        
        simul_obs_init = self.ForwardSolve(s_init)
        self.simul_obs_init = simul_obs_init
        RMSE_init = np.linalg.norm(simul_obs_init-self.obs)/np.sqrt(n)
        nRMSE_init = np.linalg.norm( np.divide(simul_obs_init-self.obs,self.R) )/np.sqrt(n)
        print('RMSE (norm(obs. diff.)/sqrt(nobs)): %g, normalized RMSE (norm(obs. diff./R)/sqrt(nobs)): %g' % (RMSE_init, nRMSE_init))
        simul_obs = np.copy(simul_obs_init)
        s_cur = np.copy(s_init)
        s_past = np.copy(s_init)

        for i in range(maxiter):
            start = time()
            print("***** Iteration %d ******" % (i+1))
            s_cur, beta_cur, simul_obs_cur, obj = self.LinearIteration(s_past, simul_obs)

            print("- time for iteration %d is %g sec" %((i+1), round(time()-start)))
            print("- Q2:%e, cR: %e at iteration %d" %(self.Q2_cur,self.cR_cur,(i+1)))
            
            if obj < self.obj_best:
                self.obj_best = obj
                self.s_best = s_cur
                self.beta_best = beta_cur
                self.simul_obs_best = simul_obs_cur
                self.iter_best = i+1
                self.Q2_best = self.Q2_cur
                self.cR_best = self.cR_cur
            else: 
                if self.linesearch:
                    print('perform simple linesearch due to no progress in obj values')
                    s_cur, simul_obs_cur, obj = self.LineSearch(s_cur, s_past)
                    if obj < self.obj_best:
                        self.obj_best = obj
                        self.s_best = s_cur
                        self.simul_obs_best = simul_obs_cur
                        self.iter_best = i+1
                    else:
                        if i > 1:
                            print('no progress in obj values')
                            iter_cur = i+1
                            break
                        else:
                            print('no progress in obj values but wait for one more iteration..')
                            # allow first few iterations
                            pass # allow for 
                else:
                    print('no progress in obj values')
                    iter_cur = i +1
                    break

            res = np.linalg.norm(s_past-s_cur)/np.linalg.norm(s_past)
            RMSE_cur = np.linalg.norm(simul_obs_cur-self.obs)/np.sqrt(n)
            nRMSE_cur = np.linalg.norm( np.divide(simul_obs_cur-self.obs,self.R) )/np.sqrt(n)
        
            if self.s_true is not None:            
                err = np.linalg.norm(s_cur-self.s_true)/np.linalg.norm(self.s_true)
                print("- iteration %d: relative L2-norm diff (w.r.t prev. iter. sol.) is %g, objective function is %e, rel. L2-norm error (w.r.t truth) is %g, obs. RMSE is %g, obs. normalized RMSE is %g" %((i+1), res, obj, err, RMSE_cur, nRMSE_cur))
            else:
                print("- iteration %d: relative L2-norm diff (w.r.t prev. iter. sol.) is %g, objective function is %e, and RMSE is %g, normalized RMSE is %g" %((i+1), res, obj, RMSE_cur, nRMSE_cur))

            if res < restol:
                iter_cur = i + 1
                break

            s_past = np.copy(s_cur)
            simul_obs = np.copy(simul_obs_cur)
            self.objvals.append(float(obj))

            if self.iter_save:
                print("- save results at iteration %d" % (i+1))
                np.savetxt('./shat' + str(i+1) + '.txt', s_cur)
                np.savetxt('./simulobs' + str(i+1) + '.txt', simul_obs_cur)
                #self.post_diagv = self.ComputePosteriorDiagonalEntries(self.HZ, self.HX, self.i_best,self.R)
                #if self.post_cov or self.post_cov == "diag":
                #    np.savetxt('./diagv' + str(i + 1) + '.txt', self.post_diagv)

        if self.post_cov or self.post_cov == "diag": # assume linesearch result close to the current solution
            start = time()
            if self.post_diag_direct:
                print("start direct posterior variance computation ")
                self.post_diagv = self.ComputePosteriorDiagonalEntriesDirect(self.HZ, self.HX, self.i_best,self.R)
            else:
                print("start posterior variance computation")
                self.post_diagv = self.ComputePosteriorDiagonalEntries(self.HZ, self.HX, self.i_best,self.R)
            print("posterior diag. computed in %f secs" % (time()-start))
            if self.iter_save:
                np.savetxt('./postv.txt', self.post_diagv)

        #return s_cur, beta_cur, simul_obs, iter_cur
        print("------------ Inversion Summary ---------------------------")
        print("** Found solution at iteration %d" %(self.iter_best))
        print("** Solution RMSE %g , initial RMSE %g, where RMSE = (norm(obs. diff.)/sqrt(nobs)), Solution nRMSE %g, init. nRMSE %g" %(np.linalg.norm(self.simul_obs_best-self.obs)/np.sqrt(self.n),RMSE_init, np.linalg.norm( np.divide(self.simul_obs_best-self.obs,self.R) )/np.sqrt(n)
        , nRMSE_init))
        print("** Final objective function value is %e" %(self.obj_best))
        print("** Final predictive model checking Q2, cR is %e, %e" %(self.Q2_best, self.cR_best))

        return self.s_best, self.simul_obs_best, self.iter_best, self.post_diagv

    def Run(self):
        start = time()
        if self.Q is None:
            self.ConstructCovariance(method = self.matvec, kernel = self.kernel, xmin = self.xmin, xmax = self.xmax, N= self.N, theta = self.prior_cov_scale)
    
        if self.priorU is None or self.priord is None:
            self.ComputePriorEig()

        s_hat, simul_obs, iter_best, post_diagv = self.GaussNewton()
        #start = time()
        print("** Total elapsed time is %f secs" % (time()-start))
        print("----------------------------------------------------------")
        if self.post_cov or self.post_cov == "diag":
            return s_hat, simul_obs, iter_best, post_diagv
        else:
            return s_hat, simul_obs, iter_best

    """
        functions below have not been implmented yet
    """
    def ComputePosteriorDiagonalEntriesDirect(self,HZ,HX,i_best,R):
        """
        Computing posterior diagonal entries
        Don't use this for large number of measurements! 
        Works best for small measurements O(100)
        """
        
        m = self.m
        n = self.n
        p = self.p

        alpha = 10 ** (np.linspace(0., np.log10(self.alphamax_LM), self.nopts_LM))
        Ri = np.multiply(alpha[i_best],R)

        n_pc = self.n_pc
        priorvar = self.prior_std**2
        Z = np.zeros((m,n_pc), dtype ='d')
        for i in range(n_pc):
            Z[:,i:i+1] = np.dot(sqrt(self.priord[i]),self.priorU[:,i:i+1]) # use sqrt to make it scalar

        v = np.zeros((m,1),dtype='d')

        # Construct Psi directly
        if isinstance(Ri,float):
            Psi = np.dot(HZ,HZ.T)+ np.multiply(Ri,np.eye(n,dtype='d'))
        elif Ri.shape[0] == 1 and Ri.ndim == 1:
            Psi = np.dot(HZ,HZ.T)+ np.multiply(Ri,np.eye(n,dtype='d'))
        else:
            Psi = np.dot(HZ,HZ.T)+ np.diag(Ri)

        HQ = np.dot(HZ,Z.T)

        #Create matrix system and solve it
        # cokriging matrix
        A = np.zeros((n+p,n+p),dtype='d')
        b = np.zeros((n+p,1),dtype='d')

        A[0:n,0:n] = np.copy(Psi)   
        A[0:n,n:n+p] = np.copy(HX)   
        A[n:n+p,0:n] = np.copy(HX.T)

        for i in range(m):
            b = np.zeros((n+p,1),dtype='d')
            b[0:n] = HQ[:,i:i+1]
            b[n:n+p] = self.X[i:i+1,:].T
            tmp = np.dot(b.T, np.linalg.solve(A, b))
            v[i] = priorvar - tmp
            if i % 1000 == 0:
                print("%d-th element evaluated" % (i))

        v[v<priorvar] = priorvar

        #print("compute variance: %f sec" % (time() - start))
        return v

    def ComputePosteriorDiagonalEntries(self,HZ,HX,i_best,R):
        """
        Computing posterior diagonal entries using iterative approach
        """
        m = self.m
        n = self.n
        p = self.p

        alpha = 10 ** (np.linspace(0., np.log10(self.alphamax_LM), self.nopts_LM))

        n_pc = self.n_pc
        priorvar = self.prior_std ** 2

        v = np.zeros((m, 1), dtype='d')

        # Create matrix context
        #if R.shape[0] == 1:
        #    def mv(v):
        #        return np.concatenate(((np.dot(HZ, np.dot(HZ.T, v[0:n])) + np.multiply(np.multiply(alpha[i_best], R),v[0:n]) + np.dot(HX,v[n:n + p])),(np.dot(HX.T, v[0:n]))), axis=0)
        #else:
        #    def mv(v):
        #        return np.concatenate(((np.dot(HZ, np.dot(HZ.T, v[0:n])) + np.multiply(np.multiply(alpha[i_best], R.reshape(v[0:n].shape))) + np.dot(HX, v[n:n + p])),(np.dot(HX.T, v[0:n]))), axis=0)

        # Matrix handle
        #Afun = LinearOperator((n + p, n + p), matvec=mv, rmatvec=mv, dtype='d')

        #callback = Residual()
        # Residual and maximum iterations
        #itertol = 1.e-10 if not 'iterative_tol' in self.params else self.params['iterative_tol']
        #solver_maxiter = m if not 'iterative_maxiter' in self.params else self.params['iterative_maxiter']

        #Preconditioner
        def invPsi(v):
            Dvec = np.divide(((1. / alpha[i_best]) * self.Psi_sigma), ((1. / alpha[i_best]) * self.Psi_sigma + 1))
            Psi_U = np.multiply((1. / sqrt(alpha[i_best])),self.Psi_U)
            Psi_UTv = np.dot(Psi_U.T, v)
            return np.multiply(np.multiply((1. / alpha[i_best]), self.invR), v) - np.dot(Psi_U,np.multiply(Dvec[:Psi_U.shape[1]].reshape(Psi_UTv.shape), Psi_UTv))


        def Pmv(v):
            invPsiv = invPsi(v[0:n])
            S = np.dot(HX.T, invPsi(HX))  # p by p matrix
            invSHXTinvPsiv = np.linalg.solve(S, np.dot(HX.T, invPsiv))
            invPsiHXinvSHXTinvPsiv = invPsi(np.dot(HX, invSHXTinvPsiv))
            return np.concatenate(((invPsiv - invPsiHXinvSHXTinvPsiv), (invSHXTinvPsiv)), axis=0)

        P = LinearOperator((n + p, n + p), matvec=Pmv, rmatvec=Pmv, dtype='d')

        #start = time()
        #for i in range(m):
        #    b = np.zeros((n + p, 1), dtype='d')
        #    b[0:n] = np.dot(HZ,(np.multiply(np.sqrt(self.priord),self.priorU[i:i+1,:].T)))
        #    b[n:n + p] = self.X[i:i+1,:].T
        #    invAb, info = gmres(Afun, b, tol=itertol, maxiter=solver_maxiter, callback=callback, M=P)
        #    #invAb, info = minres(Afun, b, tol=itertol, maxiter=solver_maxiter, callback=callback, M=P)

        #    v[i] = priorvar - np.dot(b.T, invAb.reshape(-1,1))
        #    if i % 1000 == 0:
        #        print("%d-th element evalution done.." % (i))
        #        print("-- Number of iterations for gmres %g for this element" % (callback.itercount()))
        #v[v>priorvar] = priorvar
        #print("gmres compute variance: %f sec" % (time() - start))

        #start = time()
        v = np.zeros((m, 1), dtype='d')
        for i in range(m):
            b = np.zeros((n + p, 1), dtype='d')
            b[0:n] = np.dot(HZ, (np.multiply(np.sqrt(self.priord), self.priorU[i:i + 1, :].T)))
            b[n:n + p] = self.X[i:i + 1, :].T
            v[i] = priorvar - np.dot(b.T,P(b))
            if i % 10000 == 0:
                print("%d-th element evalution done.." % (i))
        v[v > priorvar] = priorvar

        #print("Pv compute variance: %f sec" % (time() - start))
        #print("norm(v-v1): %g" % (np.linalg.norm(v - v1)))
        #print("max(v-v1): %g, %g" % ((v - v1).max(),(v-v1).min()))

        return v

    def FssInvDiag(self, recompute = True):
        """		
        Works best for small measurements O(100)
		to be implemented
        """
        raise NotImplementedError
    
    def GEIGDiag(self):
        """
        Saibaba, Lee and Kitanidis, 2016
		to be implemented
        """
        m = self.m
        n = self.n
        p = self.p
        n_pc = self.n_pc
        priorvar = self.prior_std**2
        Z = np.zeros((m,n_pc), dtype ='d')
        for i in range(n_pc):
            Z[:,i:i+1] = np.dot(sqrt(self.priord[i]),self.priorU[:,i:i+1])  # use sqrt to make it scalar

        v = np.zeros((m,1),dtype='d')
        v1 = np.zeros((m,1),dtype='d')
        priorvar1 = np.zeros((m,1),dtype='d')

        # Construct Psi directly
        if isinstance(R,float):
            Psi = np.dot(HZ,HZ.T)+ np.multiply(R,np.eye(n,dtype='d'))
        elif R.shape[0] == 1 and R.ndim == 1:
            Psi = np.dot(HZ,HZ.T)+ np.multiply(R,np.eye(n,dtype='d'))
        else:
            Psi = np.dot(HZ,HZ.T)+ np.diag(R)

        HQ = np.dot(HZ,Z.T)

        #Create matrix system and solve it
        # cokriging matrix
        A = np.zeros((n+p,n+p),dtype='d')
        b = np.zeros((n+p,1),dtype='d')

        A[0:n,0:n] = np.copy(Psi)
        A[0:n,n:n+p] = np.copy(HX)
        A[n:n+p,0:n] = np.copy(HX.T)
        start = time()
        for i in range(m):
            priorvar1[i] = np.dot(Z[i,:],Z[i,:].T)

        for i in range(m):
            b = np.zeros((n+p,1),dtype='d')
            b[0:n] = HQ[:,i:i+1]
            b[n:n+p] = self.X[i:i+1,:].T
            tmp = np.dot(b.T,np.linalg.solve(A, b)) 
            v[i] =  priorvar1[i] - tmp
            v1[i] = priorvar - tmp
            
        print("compute variance: %f sec" % (time() - start))
        
        return v, v1
    
    def CompareSpectrum(self, filename):
        """
        Compare spectrum of Hred, Qinv, and the combination of the two.
		to be implemented
        """
        raise NotImplementedError
    
    def GenerateUnconditionalRealizations(self):
        raise NotImplementedError
    
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
        raise NotImplementedError

    def Uncertainty(self, **kwargs):
        m = self.m
        diag = np.zeros((m,1),dtype = 'd')
        start = time()

        #if self.params['direct'] == True:
        #    diag = self.ComputePosteriorDiagonalEntriesDirect()
        #else:
        #    F = FisherInverse(self)
        #    prior, datalowrank, beta - F.diags()
        #    print("Uncertainty in beta is %g" % (beta))
        
        print("Time for uncertainty computation is", time() - start)

        raise NotImplementedError

    def computeP(y,HX,HZ,R,rtol=1e-5):
        # not used.. 
        time_null1 = time()
        u, s, v = np.linalg.svd(HX.T)
        rank = (s > rtol*s[0]).sum()
        
        T = v[rank:].T.copy()
        T = T.T.copy()
        print('time for null operation %f sec with rank %d' % (time()-time_null1, myrank))
        
        # 
        if isinstance(R,float):
            Psi = np.dot(HZ,HZ.T)+ np.multiply(R,np.eye(n,dtype='d'))
        elif R.shape[0] == 1 and R.ndim == 1:
            Psi = np.dot(HZ,HZ.T)+ np.multiply(R,np.eye(n,dtype='d'))
        else:
            Psi = np.dot(HZ,HZ.T)+ np.diag(R)

        Pyy = np.dot(T.T, np.linalg.solve(np.dot(T,np.dot(PSI,T.T)),T))
        P, sP, VP = np.linalg.svd(Pyy,full_matrices=False,compute_uv=True)
        P = P[:,:n-p].T

        mydelta = np.dot(P,y)
        mydeltacov = np.dot(P,np.dot(PSI,P.T))
        myeps = mydelta**2/mydeltacov.diagonal()[:,np.newaxis]
        Q2 = myeps.sum()/(n-p)
        cR = Q21*np.exp(np.log(mydeltacov.diagonal()).sum()/(n-p))
        
        return Q2, cR, P 

class CovarianceMatrixbyUd:
    def __init__(self,priord,priorU):
        self.dtype = 'd'
        n = priord.shape[0]
        self.shape = (n,n)
        self.priord = priord
        self.priorU = priorU

    def matvec(self,x):
        #y = np.zeros_like(x,dtype = 'd')
        y = np.dot(self.priorU,np.multiply(self.priord,(np.dot(self.priorU.T,x))))
        return y

    def rmatvec(self,x):
        return matvec(self,x)

class DataCovarianceMatrix:
    '''
    (HQHT + R)x 
    '''
    def __init__(self,HZ):
        self.dtype = 'd'
        n = priord.shape[0]
        self.shape = (n,n)
        #self.priord = priord
        #self.priorU = priorU
        self.HZ = HZ

    def matvec(self,x):
        y = np.dot(self.HZ,np.dot(self.HZ.T,x)) + np.dot(R,x)
        return y

    def rmatvec(self,x):
        return matvec(self,x)

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self._original_stdout