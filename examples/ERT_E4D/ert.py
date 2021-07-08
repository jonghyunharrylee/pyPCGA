import datetime as dt
import os
import sys
from multiprocessing import Pool
import numpy as np

from scipy.interpolate import griddata
from shutil import copy2, rmtree
import subprocess
#from subprocess import call
from time import time
#import pdb; pdb.set_trace()
#os.environ["OMP_NUM_THREADS"] = "1"

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
            # if 'outputdir' in params:
            #     # note that outputdir is not used for now; pyPCGA forces outputdir in ./simul/simul0000
            #     self.outputdir = params['outputdir']
            if 'parallel' in params:
                self.parallel = params['parallel']

        #smalldomain..

            if 'xyz' in params:
                self.xyz = params['xyz']
            else:
                raise NameError('xyz is not defined')

            if 'xyzout' in params:
                self.xyzout = params['xyzout']
            else:
                raise NameError('xyzout is not defined')

            if 'xyzoutval' in params:
                self.xyzoutval = params['xyzoutval']
            else:
                raise NameError('xyzoutval is not defined')
                
            if 'elexyz' in params:
                self.elexyz = params['elexyz']
            else:
                raise NameError('coord is not defined')



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


    def interp3d(self,s):
        # interpolate

        source_coord = np.vstack([self.xyz,self.xyzout])
        s_all = np.vstack([s,self.xyzoutval*np.ones((self.xyzout.shape[0],1))])
        target_coord = self.elexyz

        s_interp = griddata(source_coord, s_all, target_coord, method = 'nearest')

        return s_interp

    def run_model(self,s,idx=0):

        sim_dir = self.create_dir(idx)
        os.chdir(sim_dir)
        s_interp = self.interp3d(s)
        with open("agu.sig","w") as f:
            f.write("%d    1\n" % (s_interp.shape[0]))
            for i in range(s_interp.shape[0]):
                f.write("%e\n" % (np.exp(s_interp[i])))

        subprocess.call(["./mpirun","-np","6","--bind-to","none","e4d"], stdout=open('/dev/null','w'))
        #subprocess.call(["./mpirun","-np","3","--bind-to","none","e4d"])
        output = np.loadtxt('agu.dpd',skiprows=1)
        simul_obs = output[:,-1]
        
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
    import ert
    import numpy as np
    from time import time

    s = np.loadtxt('true_all.txt')

    #s = np.loadtxt("true.txt")
    s = s.reshape(-1, 1)
    #m = 91*91*31
    xyz = np.load('xyz.npy')
    xyzout = np.load('xyzout.npy')
    xyzoutval = -4.6052
    s = xyzoutval*np.ones((m,1))
    
    elexyz = np.load('elexyz.npy') 
    params = {'deletedir':False, 'input_dir':'./input_files/', 'xyz': xyz, \
        'xyzout': xyzout, 'elexyz': elexyz, 'xyzoutval': xyzoutval}

    par = False # parallelization false

    mymodel = ert.Model(params)
    print('(1) single run')

    from time import time
    stime = time()
    simul_obs = mymodel.run(s,par)
    print('simulation run: %f sec' % (time() - stime))
    obs = simul_obs + 0.1*np.random.randn(simul_obs.shape[0],simul_obs.shape[1])
    #np.savetxt('obs.txt',obs)
    
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
