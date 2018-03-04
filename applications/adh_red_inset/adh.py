import os
import numpy as np
from shutil import copy2, rmtree
from time import time
from multiprocessing import Pool
import setup_Red_Inset

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
        self.deletedir = True

        from psutil import cpu_count  # physcial cpu counts
        self.ncores = cpu_count(logical=False)
        self.ntsim = 1
        # inflow discharge and free surface elevation at the boundary
        self.z_f = 4.5 
        self.Q_b = 400
        if params is not None: 
            if 'deletedir' in params:
                self.deletedir = params['deletedir']
            if 'homedir' in params:
                self.homedir = params['homedir']
            if 'ncores' in params:
                self.ncores = params['ncores']

            self.adh_version = params['adh_version']
            self.adh_exec = params['adh_exec']
            self.pre_adh_exec = params['pre_adh_exec']
            self.adh_grid = params['adh_grid']
            self.adh_rect = params['adh_rect']
            self.adh_mesh = params['adh_mesh']
            self.adh_bc = params['adh_bc']

            if 'adh_ntsim' in params: self.ntsim = params['adh_ntsim']
            # inflow discharge and free surface elevation at the boundary
            # needed for writing initial condtions potentailly
            if 'z_f' in params: self.z_f = params['z_f']
            if 'Q_b' in params: self.Q_b = params['Q_b']
            
            self.velocity_obs_file = params['velocity_obs_file']
            self.elevation_obs_file = params['elevation_obs_file']
            self.true_soln_file_h5 = None if 'true_soln_file_h5' not in params else params[
                'true_soln_file_h5']
            self.true_soln_meshbase = None if 'true_soln_meshbase' not in params else params['true_soln_meshbase']
            self.sim_dir = './simul' if 'sim_dir' not in params else params['sim_dir']

    def create_dir(self,idx=None):
        
        if idx is None:
            idx = self.idx
        
        mydir = os.path.join(self.sim_dir,"simul{0:04d}".format(idx))
        mydir = os.path.abspath(os.path.join(self.homedir, mydir))
        
        if not os.path.exists(mydir):
            os.makedirs(mydir)

        if self.adh_version < 5:
            sim_prefix = os.path.abspath(mydir + "/Inset_sim_v46")
        else:
            sim_prefix = os.path.abspath(mydir + "/Inset_sim")

        copy2(self.adh_mesh, sim_prefix + '.3dm')
        copy2(self.adh_bc, sim_prefix + '.bc')
        return mydir, sim_prefix

    def run_model(self,bathy,idx=0):
        '''run adh
        '''

        sim_dir, sim_prefix = self.create_dir(idx)
        #print(sim_dir)

        forward_prob = setup_Red_Inset.RedRiver(grid_file=self.adh_grid,
                                                rect_file=self.adh_rect,
                                                mesh_file=self.adh_mesh,
                                                initial_free_surface_elevation=self.z_f)

        ##write out the base mesh, input file, and initial condition file
        forward_prob.writeMesh(sim_prefix)
        forward_prob.writeBCFile(sim_prefix)
        forward_prob.writeHotFile(sim_prefix)

        ##get the measurement locations
        velocity_obs_loc = np.loadtxt(self.velocity_obs_file)
        elev_obs_loc = np.loadtxt(self.elevation_obs_file)

        ##instantiate the inverse problem which controls the forward model simulation
        prm = setup_Red_Inset.RedRiverProblem(forward_prob.mesh,
                                              forward_prob,
                                              velocity_obs_loc,
                                              elev_obs_loc,
                                              sim_prefix=sim_prefix,
                                              debug_rigid_lid=False,
                                              ntsim=self.ntsim,
                                              AdH_version=self.adh_version,
                                              pre_adh_path=self.pre_adh_exec,
                                              adh_path=self.adh_exec,
                                              true_soln_file_h5=self.true_soln_file_h5,
                                              true_soln_meshbase=self.true_soln_meshbase,
                                              Q_b=self.Q_b,
                                              z_f=self.z_f)

        t0 = 0.
        x_true = prm.get_true_solution(t0)
        # measurment matrix
        H_meas = prm.get_measurement_matrix(t0)

        x_dummy = x_true.copy()
        #z_in = x_true[:prm.nn]
        bathy = bathy.reshape(-1)
        x_dummy[:prm.nn] = bathy
        x_dummy[prm.nn:] = prm.compute_velocity(bathy, t0)

        if self.deletedir:
            rmtree(sim_dir, ignore_errors=True)

        return H_meas.dot(x_dummy)


    def run(self,bathy,par,ncores=None):
        if ncores is None:
            ncores = self.ncores

        method_args = range(bathy.shape[1])
        args_map = [(bathy[:, arg:arg + 1], arg) for arg in method_args]

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

        #return args[0](args[1], args[2])
    #return self.run_model(self,bathy,idx)
    #def run_in_parallel(self,args):
    #    return args[0].run_model(args[1], args[2])



if __name__ == '__main__':
    import adh
    import numpy as np
    from time import time

    params = {'sim_dir':'./simul',
              'adh_exec':'./bin/v4/adh',
              'pre_adh_exec':'./bin/v4/pre_adh',
              'adh_version':4.5,
              'adh_grid':'./mesh_files/nx1001_ny51/grid_Inset_nx1001_ny51',
              'adh_rect':'./mesh_files/nx1001_ny51/rect_Inset_nx1001_ny51',
              'adh_mesh':'./mesh_files/nx1001_ny51/Inset_nx1001_ny51.3dm',
              'adh_bc':'./true_files/nx1001_ny51/Inset_true_v46.bc',
              'velocity_obs_file':'./observation_files/observation_loc_N250_M8_J1_I10.dat',
              'elevation_obs_file':'./observation_files/observation_loc_none.dat',
              'true_soln_file_h5':'./true_files/nx1001_ny51/Inset_true_v46_p0.h5',
              'true_soln_meshbase':'./true_files/nx1001_ny51/Inset_true_v46'
              }

    bathy = np.loadtxt("true.txt")
    bathy = np.array(bathy).reshape(-1, 1)

    par = False # parallelization false

    mymodel = adh.Model(params)
    print('1) single run')

    #simul_obs = mymodel.run(bathy,False)
    #simul_obs = mymodel.run_model(bathy)

    ncores = 2
    nrelzs = 2
    
    print('2) parallel run with ncores = %d' % ncores)
    par = True # parallelization false

    bathyrelz = np.zeros((np.size(bathy,0),nrelzs),'d')

    for i in range(nrelzs):
        bathyrelz[:,i:i+1] = bathy + 0.1*np.random.randn(np.size(bathy,0),1)

    simul_obs_all = mymodel.run(bathyrelz,True,ncores)

    #
    #simul_obs_all = pool.map(run_in_parallel, args_map)
    #pool.close()
    #pool.join()
    #simul_obs_all = mymodel.run(bathyrelz,par,ncores = ncores)
    #simul_obs = run_in_parallel(args_map[0])

    #print(simul_obs_all)

    # use all the physcal cores if not specify ncores
    #print('3) parallel run with all the physical cores')
    #simul_obs_all = mymodel.run(bathyrelz,par)
    #print(simul_obs_all)
