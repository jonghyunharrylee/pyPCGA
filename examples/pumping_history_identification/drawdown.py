import datetime as dt
import os
import sys
from multiprocessing import Pool
import numpy as np
from scipy.io import loadmat, savemat

from shutil import copy2, rmtree
from subprocess import call, check_output
from time import time

"""
example from CEE362g Stanford, pumping history reconstruction for horizontal infinite aquier
aquifer properties
T = 0.02   # transmissivity [L^2/T]
S = 0.001  # storatotivity  [/]
phi0 = 50. # initial head
D = T/S    # Hydraulic diffusivity
x1 = 6, x2 = 8 # location of monitoring well (x1, x2)
"""

"""
this template foward model require three operations
1. write inputs (not used)
2. run simul
3. read input (not used)
"""

class Model:
    def __init__(self,params = None):
        self.idx = 0
        self.homedir = os.path.abspath('./')
        self.inputdir = os.path.abspath(os.path.join(self.homedir,"./input_files"))
        self.deletedir = True

        from psutil import cpu_count  # physcial cpu counts
        self.ncores = cpu_count(logical=False)

        if params is not None: 
            if 'deletedir' in params:
                self.deletedir = params['deletedir']
            if 'homedir' in params:
                self.homedir = params['homedir']
                self.inputdir = os.path.abspath(os.path.join(self.homedir,"./input_files"))
            if 'inputdir' in params:
                self.inputdir = params['inputdir']
            if 'ncores' in params:
                self.ncores = params['ncores']

        # load forward model operator
        tmp = loadmat('H.mat')
        self.H = tmp['H'] # forward operator for simul_obs = np.dot(H,s) 

        # note that outputdir is not used for simulation; force outputdir in ./simul/simul0000
        self.outputdir = None if 'outputdir' not in params else params['outputdir']
        self.parallel = False if 'parallel' not in params else params['parallel']
 
    def create_dir(self,idx=None):
        """
        create directory for each , so that it does not interfere possible file IO. 
        Not Used in this example!
        """
        mydirbase = "./simul/simul"
        if idx is None:
            idx = self.idx
        
        mydir = mydirbase + "{0:04d}".format(idx)
        mydir = os.path.abspath(os.path.join(self.homedir, mydir))
        
        if not os.path.exists(mydir):
            os.makedirs(mydir)
        
        for filename in os.listdir(self.inputdir):
            copy2(os.path.join(self.inputdir,filename),mydir)
        
        return mydir

    def cleanup(self,outputdir=None):
        """
        Removes outputdir if specified. Otherwise removes all output files
        in the current working directory.
        Not Used in this example!
        """
        import shutil
        import glob
        
        log = "dummy.log"
        if os.path.exists(log):
            os.remove(log)
        if outputdir is not None and outputdir != os.getcwd():
            if os.path.exists(outputdir):
                shutil.rmtree(outputdir)
        else:
            #filelist = glob.glob("*.out")
            #filelist += glob.glob("*.dat")
            
            #for file in filelist:
            #    os.remove(file)
            pass
        
    
    def run_model(self,s,idx=0):
        """
            run model y = Hs
            y.shape should be (m,) for parallelization!
        """
        # create directory
        #sim_dir = self.create_dir(idx)
        
        simul_obs = np.dot(self.H,s)
        simul_obs = simul_obs.reshape(-1)

        #if self.deletedir:
        # rmtree(sim_dir, ignore_errors=True)
        # self.cleanup(sim_dir)

        return simul_obs

    def run(self,s,par,ncores=None):
        if ncores is None:
            ncores = self.ncores

        method_args = range(s.shape[1])
        args_map = [(s[:, arg:arg + 1], arg) for arg in method_args]

        if par:
            pool = Pool(processes=ncores)
            simul_obs = pool.map(self, args_map)
        else:
            simul_obs =[]
            for item in args_map:
                simul_obs.append(self(item))

        return np.array(simul_obs).T


    def __call__(self,args):
        return self.run_model(args[0],args[1])


if __name__ == '__main__':
    import drawdown as dd
    import numpy as np
    from time import time

    s_true = np.zeros((10001,),'d')
    s_true[1001:3502] = 0.12 
    s_true[3502:4002] = 0.06

    s_true = s_true.reshape(-1, 1)
    #np.savetxt("true.txt",s_true)
    
    par = False # parallelization false

    params = {}

    mymodel = dd.Model(params)
    print('(1) single run')

    simul_obs = mymodel.run(s_true,par)
    obs = simul_obs + 0.01*np.random.randn(100,1)
    #np.savetxt('obs.txt',obs)

    ncores = 2
    nrelzs = 2
    
    # if your ipython does not work with parallel runs, please uncomment below
    #__spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"
    
    print('(2) parallel run with ncores = %d' % ncores)
    par = True # parallelization false
    srelz = np.zeros((np.size(s_true,0),nrelzs),'d')
    for i in range(nrelzs):
        srelz[:,i:i+1] = s_true + 0.1*np.random.randn(np.size(s_true,0),1)
    
    simul_obs_all = mymodel.run(srelz,par,ncores = ncores)

    print(simul_obs_all)

    # use all the physcal cores if not specify ncores
    #print('(3) parallel run with all the physical cores')
    simul_obs_all = mymodel.run(srelz,par)
    print(simul_obs_all)
