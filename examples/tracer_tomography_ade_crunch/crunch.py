import datetime as dt
import os
import sys
from multiprocessing import Pool
import numpy as np

from shutil import copy2, rmtree
import subprocess
#from subprocess import call
from time import time
from IPython.core.debugger import Tracer; debug_here = Tracer()

'''
three operations
1. write inputs
2. run simul
3. read input
'''

class Model:
    def __init__(self,params = None):
        self.idx = 0
        self.homedir = os.path.abspath('./')
        self.inputdir = os.path.abspath(os.path.join(self.homedir,"./input_files"))
        self.deletedir = True
        self.outputdir = None
        self.parallel = False
        self.record_cobs = False

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
            if 'outputdir' in params:
                # note that outputdir is not used for now; pyPCGA forces outputdir in ./simul/simul0000
                self.outputdir = params['outputdir']
            if 'parallel' in params:
                self.parallel = params['parallel']
            if 'nx' in params:
                self.nx = params['nx']
            else:
                raise NameError('nx is not defined')
            
            if 'ny' in params:
                self.ny = params['ny']
            else:
                raise NameError('ny is not defined')
            
            if 't' in params:
                self.t = params['t']
            else:
                raise NameError('t is not defined')

            if 'record_cobs' in params:
                self.record_cobs = True

    def create_dir(self,idx=None):
        
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
            filelist = glob.glob("*.out")
            filelist += glob.glob("*.sim")
            
            for file in filelist:
                os.remove(file)

    def run_model(self,s,idx=0):

        sim_dir = self.create_dir(idx)
        os.chdir(sim_dir)
        
        nx, ny = self.nx, self.ny
        m = nx*ny
        t = self.t
        nt = t.shape[0]

        
        perm2d = np.exp(s).reshape(ny,nx)

        # perm.x
        perm2dx = np.zeros((ny,nx+2),'d')
        perm2dx[:,1:-1] = perm2d
        perm2dx[:,0] = perm2dx[:,1]
        perm2dx[:,-1] = perm2dx[:,-2]
        
        np.savetxt("PermField.x",perm2dx.reshape(ny*(nx+2),),fmt='%10.4E')
        
        
        perm2dy = np.zeros((ny+2,nx),'d')
        perm2dy[1:-1,:] = perm2d
        perm2dx[0,:] = perm2dx[1,:]
        perm2dx[-1,:] = perm2dx[-2,:]
        np.savetxt("PermField.y",perm2dy.reshape((ny+2)*nx,),fmt='%10.4E')
        
        subprocess.call(["./CrunchTope","2DCr.in"], stdout=subprocess.PIPE)

        # read results
        simul_cobs = np.zeros((m,nt),'d')
        simul_obs = np.zeros((m,),'d')

        for it in range(nt):
            tmp = np.loadtxt('totcon%d.tec' % (it+1),skiprows=3)
            simul_cobs[:,it] = tmp[:,3]
        
        if self.record_cobs:
            self.simul_cobs = simul_cobs

        for it in range(m):        
            simul_obs[it] = np.trapz(t*simul_cobs[it,:],x=t)/np.trapz(simul_cobs[it,:],x=t)
        
        os.chdir(self.homedir)
        
        if self.deletedir:
            rmtree(sim_dir, ignore_errors=True)
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

        #pool.close()
        #pool.join()

    def __call__(self,args):
        return self.run_model(args[0],args[1])


if __name__ == '__main__':
    import crunch
    import numpy as np
    from time import time

    s = np.loadtxt("true.txt")
    s = s.reshape(-1, 1)
    nx = 50
    ny = 50
    m = nx*ny
    t = np.array([1.1574E-05, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7,\
     0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0])
    params = {'nx':nx,'ny':ny, 't': t, 'deletedir':False, 'record_cobs':True}

    #s = -30.*np.ones((nx*ny,1),'d')
    #s = s.reshape(-1, 1)
    par = False # parallelization false


    mymodel = crunch.Model(params)
    print('(1) single run')

    from time import time
    stime = time()
    simul_obs = mymodel.run(s,par)
    print('simulation run: %f sec' % (time() - stime))
    obs = simul_obs + 0.01*np.random.randn(m,1)
    obs[obs < 0] = 0
    np.savetxt('obs.txt',obs)
    np.savetxt('cobs.txt',mymodel.simul_cobs)
    #mymodel.simul_cobs = mymodel.simul_cobs
    #for it in range(nx*ny):        
    #    simul_obs[it] = np.trapz(t*mymodel.simul_cobs[it,:],x=t)/np.trapz(mymodel.simul_cobs[it,:],x=t)

    #savemat('simul.mat',{'simul_obs':simul_obs})    
    
    import sys
    sys.exit(0)

    ncores = 2
    nrelzs = 2
    
    print('(2) parallel run with ncores = %d' % ncores)
    par = True # parallelization false
    srelz = np.zeros((np.size(s,0),nrelzs),'d')
    for i in range(nrelzs):
        srelz[:,i:i+1] = s + 0.1*np.random.randn(np.size(s,0),1)
    
    simul_obs_all = mymodel.run(srelz,par,ncores = ncores)

    print(simul_obs_all)

    # use all the physcal cores if not specify ncores
    #print('(3) parallel run with all the physical cores')
    #simul_obs_all = mymodel.run(srelz,par)
    #print(simul_obs_all)
