import flopy
import flopy.utils.binaryfile as bf
import numpy as np
import os
from shutil import copy2, rmtree
from multiprocessing import Pool

class Model:
    def __init__(self, params=None):
        self.idx = 0
        self.homedir = os.path.abspath('./')
        self.deletedir = True

        from psutil import cpu_count  # physcial cpu counts
        self.ncores = cpu_count(logical=False)

        if params is not None:
            if 'deletedir' in params:
                self.deletedir = params['deletedir']
            if 'homedir' in params:
                self.homedir = params['homedir']
            if 'ncores' in params:
                self.ncores = params['ncores']

            self.mf_exec = params['mf_exec']
            self.Lx = params['Lx']
            self.Ly = params['Ly']
            self.Q  = params['Q']
            self.Rch  = params['Rch']
            self.nlay = params['nlay']
            self.nrow = params['nrow']
            self.ncol = params['ncol']
            self.ztop = params['ztop']
            self.zbot = params['zbot']
            self.obs_locmat = params['obs_locmat']
            self.Q_locs = params['Q_locs']
            self.input_dir = params['input_dir']
            self.sim_dir = './simul' if 'sim_dir' not in params else params['sim_dir']
        else:
            raise ValueError("You have to provide relevant MODFLOW-FloPy parameters")


    def create_dir(self, idx=None):

        if idx is None:
            idx = self.idx

        mydir = os.path.join(self.sim_dir, "simul{0:04d}".format(idx))
        mydir = os.path.abspath(os.path.join(self.homedir, mydir))

        if not os.path.exists(mydir):
            os.makedirs(mydir)

        #copy2(os.path.abspath(os.path.join(self.homedir,self.input_dir,self.mf_exec)), mydir)

        return mydir

    def run_model_single(self, logHK, Q_loc, idx = 0):
        '''run modflow
        '''

        if not isinstance(Q_loc[0],(int,np.integer)):
            raise TypeError("Expected int for Q_loc[0], got %s" % (type(Q_loc[0]),))
        if not isinstance(Q_loc[1],(int,np.integer)):
            raise TypeError("Expected int for Q_loc[1], got %s" % (type(Q_loc[1]),))
        if not isinstance(Q_loc[2],(int,np.integer)):
            raise TypeError("Expected int for Q_loc[2], got %s" % (type(Q_loc[1]),))

        mydir = self.create_dir(idx)
        while not os.path.exists(mydir): # for windows..
            mydir = self.create_dir(idx)
        #self.create_dir(idx)

        Lx = self.Lx;Ly = self.Ly
        Q = self.Q; Rch = self.Rch
        nlay = self.nlay; nrow = self.nrow; ncol = self.ncol
        ztop = self.ztop; zbot = self.zbot
        HK = (np.exp(logHK)).reshape(nlay,nrow,ncol)

        obs_locmat = np.copy(self.obs_locmat)

        modelname = 'mf'

        exec_name = os.path.abspath(os.path.join(self.homedir, self.input_dir, self.mf_exec))
        mymf = flopy.modflow.Modflow(modelname=modelname, exe_name=exec_name, model_ws=mydir)

        delr = Lx / ncol
        delc = Ly / nrow
        delv = (ztop - zbot) / nlay
        botm = np.linspace(ztop, zbot, nlay + 1)

        # Create the discretization object
        dis = flopy.modflow.ModflowDis(mymf, nlay, nrow, ncol, delr=delr, delc=delc,top=ztop, botm=botm[1:])

        # Variables for the BAS package
        ibound = np.ones((nlay, nrow, ncol), dtype=np.int32)
        ibound[:, :, 0] = -1
        ibound[:, :, -1] = -1
        strt = np.ones((nlay, nrow, ncol), dtype=np.float32)
        strt[:, :, 0] = 120.
        strt[:, :, -1] = 110.
        bas = flopy.modflow.ModflowBas(mymf, ibound=ibound, strt=strt)

        # Add LPF package
        lpf = flopy.modflow.ModflowLpf(mymf, hk=HK, vka=HK, ipakcb=53)

        # Add RCH package
        flopy.modflow.mfrch.ModflowRch(mymf, nrchop=3, rech=0.001)
        # Add WEL package
        # Remember to use zero-based layer, row, column indices!
        wel_sp = [[Q_loc[0], Q_loc[1], Q_loc[2], Q]]  # lay, row, col index, pumping rate
        stress_period_data = {0: wel_sp}  # define well stress period {period, well info dictionary}
        wel = flopy.modflow.ModflowWel(mymf, stress_period_data=stress_period_data)

        # Add OC package
        spd = {(0, 0): ['save head']}
        oc = flopy.modflow.ModflowOc(mymf, stress_period_data=spd, compact=True)

        # Add PCG package to the MODFLOW model
        pcg = flopy.modflow.ModflowPcg(mymf)

        while not os.path.exists(mydir): # for windows..
            mydir = self.create_dir(idx)

        # Write the MODFLOW model input files
        mymf.write_input()

        # Run the MODFLOW model
        success, buff = mymf.run_model(silent=True)

        ##get the measurement locations
        hds = bf.HeadFile(os.path.join(mydir,modelname + '.hds'))
        times = hds.get_times()  # simulation time, steady state
        head = hds.get_data(totim=times[-1])

        obs_locmat[Q_loc] = False # don't count head at pumping well
        simul_obs = head[obs_locmat]
        simul_obs = simul_obs.reshape(-1) # 1d array
        hds.close()

        if self.deletedir:
            rmtree(mydir, ignore_errors=True)
            #rmtree(sim_dir)
        #return H_meas.dot(x_dummy)
        return simul_obs


    def run_model(self, HK, idx = 0):
        '''run adh
        '''
        Q_locs = self.Q_locs
        kk = 0
        for Q_loc in Q_locs:
            if kk == 0:
                simul_obs = self.run_model_single(HK, Q_loc, idx)
            else:
                simul_obs = np.hstack((simul_obs,self.run_model_single(HK, Q_loc, idx)))
            kk = kk + 1

        return simul_obs # 1d array


    def run(self, HK, par, ncores=None):
        if ncores is None:
            ncores = self.ncores

        method_args = range(HK.shape[1])
        args_map = [(HK[:, arg:arg + 1], arg) for arg in method_args]

        if par:
            pool = Pool(processes=ncores)
            simul_obs = pool.map(self, args_map)
        else:
            simul_obs = []
            for item in args_map:
                simul_obs.append(self(item))

        return np.array(simul_obs).T # make it 2D

        # pool.close()
        # pool.join()

    def __call__(self, args):
        return self.run_model(args[0], args[1])

        # return args[0](args[1], args[2])
    # return self.run_model(self,bathy,idx)
    # def run_in_parallel(self,args):
    #    return args[0].run_model(args[1], args[2])

if __name__ == '__main__':
    import numpy as np
    #from time import time
    import mf

    # parameters
    if os.name == 'nt':
        mf_exec = 'mf2005.exe'
    else:
        mf_exec = 'mf2005'

    # location of mf2005 executable
    input_dir = "./input_files"
    sim_dir = './simul'
    Lx = 1000.; Ly = 750.
    Q = 25.; Rch = 0.001
    nlay = 1; nrow = 75; ncol = 100
    ztop = 0.; zbot = -1.

    obs_locmat = np.zeros((nlay, nrow, ncol), np.bool)
    for i in range(5,71,16):
        for j in range(9,96,16):
            obs_locmat[0, i, j] =  1

    Q_locs_idx = np.where(obs_locmat == True)
    Q_locs = []
    #Q_locs.append((Q_locs_idx[0][0], Q_locs_idx[1][0], Q_locs_idx[2][0]))
    for Q_loc in zip(Q_locs_idx[0], Q_locs_idx[1], Q_locs_idx[2]):
        Q_locs.append(Q_loc)

    mf_params = {'mf_exec': mf_exec, 'input_dir': input_dir,
              'sim_dir': sim_dir,
              'Lx': Lx,'Ly': Ly,
              'Q': Q, 'Rch': Rch,
              'nlay': nlay, 'nrow': nrow, 'ncol': ncol,
              'zbot': zbot, 'ztop': ztop,
              'obs_locmat':obs_locmat, 'Q_locs':Q_locs}

    logHK = np.loadtxt('true_logK.txt')
    logHK = np.array(logHK).reshape(-1, 1) # make it m x 1 2D array

    par = False  # parallelization false

    mymodel = Model(mf_params)
    print('1) single run')
    #logHK = np.loadtxt('shat2.txt')
    #logHK = np.array(logHK).reshape(-1, 1)
    simul_obs = mymodel.run(logHK,par)
    # generate synthetic observations
    #obs = simul_obs + 0.5*np.random.randn(simul_obs.shape[0],simul_obs.shape[1])
    #np.savetxt('obs.txt', obs)

    #print('2) parallel run with ncores = %d' % ncores)
    par = True  # parallelization false
    ncores = 2
    nrelzs = 2

    logHKrelz = np.zeros((np.size(logHK, 0), nrelzs), 'd')

    for i in range(nrelzs):
        logHKrelz[:, i:i + 1] = logHK + 0.1 * np.random.randn(np.size(logHKrelz, 0), 1)

    #simul_obs_all = mymodel.run(logHKrelz, True, ncores)

    # print(simul_obs_all)

    # use all the physcal cores if not specify ncores
    # print('3) parallel run with all the physical cores')
    # simul_obs_all = mymodel.run(logHKrelz,par)
    # print(simul_obs_all)

    simul_obs = mymodel.run(logHK,par)
