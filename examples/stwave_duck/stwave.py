import datetime as dt
import os
import sys
from stwave_utils.stwave_forward_problem import STWaveProblem
from stwave_utils.run_stwave import read_speed
from multiprocessing import Pool
import numpy as np

from shutil import copy2, rmtree
from subprocess import call, check_output
from time import time

#requires mpi4py 3.0
HAVE_MPIPOOL = True

try:
    from mpi4py.futures import MPIPoolExecutor
except:
    HAVE_MPIPOOL = False
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
            if 'use_mpi_pool' in params:
                self.use_mpi_pool = params['use_mpi_pool']
            else:
                self.use_mpi_pool = False
        self.nx = params['nx']
        self.ny = params['ny']
        self.Lx = params['Lx']
        self.Ly = params['Ly']
        self.x0 = 0. if 'x0' not in params else params['x0']
        self.y0 = 0. if 'y0' not in params else params['y0']
        self.t1 = params['t1']
        self.t2 = params['t2']
        # note that outputdir is not used for simulation; force outputdir in ./simul/simul0000
        self.outputdir = None if 'outputdir' not in params else params['outputdir']
        self.parallel = False if 'parallel' not in params else params['parallel']
        self.offline_dataloc = None if 'offline_dataloc' not in params else params['offline_dataloc']
        self.offline_dataname = None if 'offline_dataname' not in params else params['offline_dataname']
        self.true_elev_xyz_file = None if 'true_elev_xyz_file' not in params else params['true_elev_xyz_file']
        self.frfdataloc = None if 'frfdataloc' not in params else params['frfdataloc']
        self.chlDataLoc = None if 'chlDataLoc' not in params else params['chlDataLoc']
        if self.use_mpi_pool: assert HAVE_MPIPOOL
        
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

    def cleanup(outputdir=None):
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
            filelist += glob.glob("*.dep")
            filelist += glob.glob("*.0000")
            filelist += glob.glob("*.eng")
            filelist += glob.glob("*.dat")
            for file in filelist:
                os.remove(file)

    def run_model(self,bathy,idx=0):

        sim_dir = self.create_dir(idx)
        self.stwp = STWaveProblem(nx=self.nx, ny=self.ny, Lx=self.Lx, Ly=self.Ly,
                                  x0=self.x0, y0=self.y0, t1=self.t1, t2=self.t2,
                                  offline_dataloc=self.offline_dataloc,
                                  outputdir=sim_dir, parallel=self.parallel,
                                  testname='stwave_out')
        self.stwp.setup()
        self.stwp.bathy_true = bathy
        self.stwp.run()
        # stwp.write_grid_coordinates_to_file()

        simul_obs = read_speed(sim_dir, 'stwave_out.sim')

        if self.deletedir:
            rmtree(sim_dir, ignore_errors=True)
            # self.cleanup(sim_dir)

        return simul_obs

    def run(self,bathy,par,ncores=None):
        if ncores is None:
            ncores = self.ncores

        method_args = range(bathy.shape[1])
        args_map = [(bathy[:, arg:arg + 1], arg) for arg in method_args]

        if par and not self.use_mpi_pool:
            pool = Pool(processes=ncores)
            simul_obs = pool.map(self, args_map)
        elif par and self.use_mpi_pool:
            pool = MPIPoolExecutor(ncores)
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
    import stwave as st
    import numpy as np
    from time import time

    bathy = np.loadtxt("true_depth.txt")
    bathy = bathy.reshape(-1, 1)
    par = False # parallelization false

    nx = 110
    ny = 83
    Lx = 550
    Ly = 415
    x0, y0 = (62.0, 568.0)
    t1 = dt.datetime(2015, 10, 07, 20, 00)
    t2 = dt.datetime(2015, 10, 07, 21, 00)

    params = {'nx':nx,'ny':ny,'Lx':Lx,'Ly':Ly,'x0':x0,'y0':y0,'t1':t1,'t2':t2,
              'offline_dataloc':"./input_files/8m-array_2015100718_2015100722.nc"}

    mymodel = st.Model(params)
    print('(1) single run')

    simul_obs = mymodel.run(bathy,par)
    #savemat('simul.mat',{'simul_obs':simul_obs})    
    ncores = 2
    nrelzs = 2
    
    print('(2) parallel run with ncores = %d' % ncores)
    par = True # parallelization false
    bathyrelz = np.zeros((np.size(bathy,0),nrelzs),'d')
    for i in range(nrelzs):
        bathyrelz[:,i:i+1] = bathy + 0.1*np.random.randn(np.size(bathy,0),1)
    
    simul_obs_all = mymodel.run(bathyrelz,par,ncores = ncores)

    print(simul_obs_all)

    # use all the physcal cores if not specify ncores
    #print('(3) parallel run with all the physical cores')
    #simul_obs_all = mymodel.run(bathyrelz,par)
    #print(simul_obs_all)
