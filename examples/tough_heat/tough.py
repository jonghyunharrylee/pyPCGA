import matplotlib
#matplotlib.use('Agg')

import math
import os
import pandas as pd
import matplotlib.pyplot as plt
import sys
import pandas as pd

# PyTOUGH specific modules
from mulgrids import *
from t2thermo import *
from t2data import *
from t2incons import *
from os import system
import scipy.io as sio

import datetime as dt
import os
import sys
from multiprocessing import Pool
import numpy as np

from shutil import copy2, rmtree
import subprocess
#from subprocess import call
from time import time
from scipy.interpolate import interp1d
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
            
            required_params = ("nx","dx","xlocs","ylocs","zlocs","obs_type","t_obs_interval", \
            "max_timesteps","tstop","const_timestep","max_timestep","absolute_error","relative_error","print_interval", \
            "relative_error","print_interval","timestep_reduction","gravity",'default_incons', \
            "multi_params","solver_params","output_times_params")

            if not params.keys() != required_params:
                raise ValueError("You need to provide all the required tough parameters")
            
            self.__dict__.update((k, v) for k, v in params.items() if k in required_params)
            
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
        
        # Create TOUGH2 input data file:
        dat = t2data()
        dat.title = '3D synthetic Ex1'

        dx = self.dx # grid size
        nx = self.nx # number of grids in each dimension. Don't use large grid because of pytough naming convention
        geo = self.construct_grid(dx, nx)
        dat.grid = t2grid().fromgeo(geo)
        
        # simulation parameters:
        # Table 4.9 page 78 pytough tutorial and Appendix E of TOUGH2 tutorial
        # data.parameter is a dictionary
        # each parameter can be called as dat.parameter['parameter name']

        dat.parameter.update(
            {'max_timesteps': self.max_timesteps,           # maximum number of time steps
            'tstop': self.tstop,                            # stop time
            'const_timestep': self.const_timestep,          # time step length
            'max_timestep': self.max_timestep,              # maximum time step size
            'absolute_error': self.absolute_error,          # absolute conexgence tolerance
            'relative_error': self.relative_error,          # relative convergence tolerance
            'print_interval': self.print_interval,          # time step interval for printing
            'timestep_reduction': self.timestep_reduction,  # time step reduction factor
            'gravity': self.gravity,                        # gravitational acceleration
            'default_incons': self.default_incons})         # default initial conditions
            # Pressure in Pa, 100 m water = 10.e5 Pa water, 10 is the temperature in Celcius
        dat.start = True

        # Table 4.9 page 78 pytough tutorial and Appendix E of TOUGH2 tutorial
        # http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.396.8810&rep=rep1&type=pdf
        # Set MOPs: PARAM option in INFILE
        dat.parameter['option'][1] = 1
        dat.parameter['option'][5] = 3
        dat.parameter['option'][7] = 1
        dat.parameter['option'][11] = 2
        dat.parameter['option'][12] = 2
        dat.parameter['option'][15] = 1
        dat.parameter['option'][16] = 4
        dat.parameter['option'][20] = 0
        dat.parameter['option'][21] = 1

        # Set relative permeability (Corey) and capillarity functions:
        # Table 4.10, page 79 PYTOUGH
        dat.relative_permeability = {'type': 7, 'parameters': [0.95, 0.5, 1., 0.25, 0.]}
        dat.capillarity = {'type': 7, 'parameters': [0.8, 0.32, 1.e-4, 1.e7, 0.999]}

        # rocktype object, Table 3.3, page 67 PyTOUGH
        r1 = rocktype('dfalt', permeability = [0.e-13]*3,density = 2600, porosity = 0.25,conductivity =2.51 , specific_heat = 920,)
        r2 = rocktype('HOMO1', permeability = [9.e-13, 3.e-14, 3.e-14],density = 2600, porosity = 0.25,conductivity =2.51 , specific_heat = 920,)
        r3 = rocktype('OUTB1', permeability = [1.e-13]*3,density = 2600, porosity = 0.25,conductivity =2.51 , specific_heat = 20000,)
        
        dat.grid.add_rocktype(r2)
        dat.grid.add_rocktype(r3)
        
        dat.multi.update(self.multi_params)

        # SOLVR Table 4.11 page 79 PyTough
        dat.solver.update(self.solver_params)
        # TIMES table 4.8
        dat.output_times.update(self.output_times_params)

        # rocktypes:
        # Setting rocktype based on the block's 'z' coordinate
        z_bottom = -17.
        z_top = -2.

        # assign all non boundary elements to r2
        for blk in dat.grid.blocklist[1:]: 
            if z_bottom < blk.centre[2] < z_top:
                blk.rocktype=r2
            else: blk.rocktype=r3
        
        for blk, pmx in zip(dat.grid.blocklist[1:], np.exp(s)):
            blk.pmx = pmx
        
        # setting the blocks for FOFT
        # this section needs to be modified. FOFT blocks can be given by the coordinate of the block, or by the block name
        # currently, it accepts the cell number in each direction
        # for example the first cell here is located in grid (2,2) in 5th layer
        
        xlocs, ylocs, zlocs = self.xlocs, self.ylocs, self.zlocs
        
        nxlocs, nylocs, nzlocs = xlocs.shape[0],ylocs.shape[0],zlocs.shape[0]
        #nobs = nxlocs*nylocs*nzlocs

        x_obs = np.tile(xlocs,nylocs*nzlocs)
        y_obs = np.tile(np.repeat(ylocs,nxlocs),nylocs)
        z_obs = np.repeat(zlocs,nxlocs*nylocs) 
        
        # from function definition above 
        # set_measurement_blk(dat, x, y, z, nx):
        self.set_measurement_blk(dat, x_obs, y_obs, z_obs, nx)

        # adding boundary conditions
        center = [150, 150] # [x,y] position of the center
        L = 40     # length of one side of square
        qmax = 15000
        L_scale = 300
        self.heat_generator(dat, geo, center, L, qmax, L_scale, method = 'Square')
        
        # write data
        dat.write('INFILE')
        self.modify_infile()
        self.clear_FOFT(sim_dir)

        from time import time
        stime = time()
        # running the tough2 model
        subprocess.call(["mpirun","-n","6","--bind-to","none","tough2-mp-eos1.debug"], stdout=subprocess.PIPE)
        print('tough simulation run: %f sec' % (time() - stime))
        
        # read simulation results with obs_type = 'Gas Pressure' and/or 'Temperature'
        #from time import time
        measurements = []
        for str_obs_type in self.obs_type:
            #stime = time()
            measurements.extend(self.observation_model(sim_dir,str_obs_type))
            #print('measurement interpolation: %f sec' % (time() - stime))

        simul_obs = np.array(measurements).reshape(-1)
        
        os.chdir(self.homedir)
        
        if self.deletedir:
            rmtree(sim_dir, ignore_errors=True)
            # self.cleanup(sim_dir)

        return simul_obs

    def clear_FOFT(self,cur_dir):
        """To delete previous FOFT qnd MESH files in the directory.
        You can delete SAVE, INCON, ... files the same way if you need to do it before running"""
        for file in os.listdir(cur_dir):
            if file.startswith("FOFT") or file.startswith("MES"):
                try:
                    os.remove(file)
                except Exception.e:  # was a comma, changed to a period
                    print(e)         # was print e, changed to print(e)
                    
    def read_FOFT(self,cur_dir):
        """ Function to read all FOFT files in the directory.
        The function returns a pandas dataframe containing measurements"""
        
        FOFT_files = [filename for filename in os.listdir('.') if filename.startswith("FOFT")]
        columns_name = ['Element','Time', 'Gas Pressure', 'Gas Saturation','Temperature']
        frame = [] 
        for filename in FOFT_files:
            df = pd.read_fwf(filename,colspecs=[(12,16),(21,35),(37,52),(54,69),(72,86)],skiprows=[0,1],names=columns_name)
            frame.append(df)
        frame = pd.concat(frame)
            
        #     with open(filename, 'rb') as f_input:
        #         count =0
        #         for row in f_input:
        #             if count > 1:
        #                 # follow the file format!!!
        #                 rows.append([row[12:16],row[21:35],row[37:52],row[54:69],row[72:86]])
        #             count = count + 1
        # frame = pd.DataFrame(rows, columns=columns_name)

        frame[['Time','Gas Pressure','Gas Saturation','Temperature']] = frame[['Time','Gas Pressure','Gas Saturation','Temperature']].apply(pd.to_numeric)
        
        return frame

    def set_measurement_blk_list(self,dat, x_mid, dx, y_mid, dy, z_mid, dz):
        ''' set measurement blk list
        '''
        for blk in dat.grid.blocklist[1:]:
            if z_mid-dz/2 < blk.centre[2] < z_mid+dz/2:
                if x_mid-dx/2 < blk.centre[0] < x_mid+dx/2:
                    if y_mid-dy/2 < blk.centre[1] < y_mid+dy/2:
                        dat.history_block.append(blk.name)
        return 

    def set_measurement_blk(self,dat, x, y, z, nx):
        ''' set measurement blk
        '''    
        dat.history_block = []
        x_obs = np.array(x)
        y_obs = np.array(y)
        z_obs = np.array(z)
        block_list = nx[0]*nx[1]*(z_obs-1)+nx[0]*(y_obs-1)+x_obs
        for blk_number in block_list:
            dat.history_block.append(dat.grid.blocklist[blk_number].name)
        
        return 

    def heat_generator(self,dat, geo, center, L, qmax, L_scale, method = 'Square'):
        ''' heat generator
        '''
        cols =[]
        if method =='Square':
            for col in geo.columnlist:
                if center[0] - L/2 <= col.centre[0] <= center[0] + L/2:
                    if center[1] - L/2<= col.centre[1]<= center[1] + L/2:
                        cols.append(col)
            dxs=np.array([col.centre[0]-center[0] for col in cols])/L_scale
            dys=np.array([col.centre[1]-center[1] for col in cols])/L_scale
            #print(dxs)
            #print(dys)
            corlength = 0.5
            qcol=qmax*np.exp(-0.5*((dxs*dxs+dys*dys)/(corlength*corlength)))
            #print(qcol)
            
            #layer=geo.layerlist[-1] Changing to add the heat in a non-boundary layer
            layer=geo.layerlist[-5]
            # Page 87 from pyTOUGH #####################################
            # adding the generator to the problem
            dat.clear_generators()
            
            dat.add_generator(t2generator(name=' in1', block=' d125', type='COM1', gx=.6, ex=3.E+5))
            dat.add_generator(t2generator(name=' in2', block=' e125', type='COM1', gx=.6, ex=3.E+5))
            dat.add_generator(t2generator(name=' in3', block=' f125', type='COM1', gx=.6, ex=3.E+5))
            dat.add_generator(t2generator(name=' in4', block=' g125', type='COM1', gx=.6, ex=3.E+5))
            
            dat.add_generator(t2generator(name=' ex1', block=' d145', type='COM1', gx=-.6))
            dat.add_generator(t2generator(name=' ex2', block=' e145', type='COM1', gx=-.6))
            dat.add_generator(t2generator(name=' ex3', block=' f145', type='COM1', gx=-.6))
            dat.add_generator(t2generator(name=' ex4', block=' g145', type='COM1', gx=-.6))
            
            #print('Generators added')
            
            #for col,q in zip(cols,qcol):
            #    blkname=geo.block_name(layer.name,col.name)
            #    genname=' q'+col.name
            #    dat.add_generator(t2generator(name=genname, block=blkname, type='HEAT', gx=q*col.area))
            dat.short_output={}
        else:
            raise NotImplementedError

        return

    def construct_grid(self,dx_grid, nx):
        """Constructing a TOUGH2 input geometry file for a 3D rectangular
        model with the domain size of 300*100*20 m^3 and grid size 10*10*2 m^3."""

        # Here we set up a rectangular grid. For other option, refer to pytough examples and tutorials
        dx = [dx_grid[0]]*nx[0]
        dy = [dx_grid[1]]*nx[1]
        dz = [dx_grid[2]]*nx[2]
        # use convention =2 for naming convention. Otherwise it will generate an error when you want to write INFILE
        # for atmos_type, refer to page 124 pyTOUGH tutorial
        geo = mulgrid().rectangular(dx, dy, dz, atmos_type = 0, convention=2) 
        geo.write('geom.dat')      # write the geometry file

        return geo

    def read_SAVE(self):
        ''' read SAVE using pandas
        '''
        data_SAVE= pd.read_csv('SAVE',sep="\s+", header =None, skiprows=[0], error_bad_lines = False)
        data_SAVE = data_SAVE.drop([0,1, len(data_SAVE)-1])
        blk_1 = []
        blk_2 = []
        blk_3 = []
        for i in range(2,len(data_SAVE)+2):
            if i%2 ==0:
                if math.isnan(data_SAVE[2][i]):
                    blk_1.append(data_SAVE[1][i])
                else:
                    blk_1.append(data_SAVE[2][i])
            else:
                blk_2.append(float(data_SAVE[0][i]))
                blk_3.append(float(data_SAVE[1][i]))
        data_save = {'Saturation': blk_1,
                    'Pressure': blk_2,
                    'Temperature': blk_3}
        
        df = pd.DataFrame.from_dict(data_save)
        
        return df

    def plot_FoFT(self,data_FOFT, Elem, variable = 'Temperature', ylim = [0, 100], xlim = [0, 86400], figname=None):
        ''' plot FoFT
        NOTE THAT PLOTFOFT WILL NOT PLOT ANY OF THE LOCATIONS THAT HAVE A BLANK IN THEIR NAME
        '''
        x = np.array(data_FOFT[data_FOFT['Element']==Elem]['Time'])
        plt.plot(x,data_FOFT[data_FOFT['Element']==Elem][variable])
        #plt.axis([0, x[-1], ylim[0], ylim[1]])
        plt.axis([xlim[0], xlim[1], ylim[0], ylim[1]])
        #plt.yticks([])
        if figname is not None:
            plt.savefig(figname+'.png')
        plt.show()
        plt.close()
        return 

    def plot_SAVE(self, dat, df, nx, y=0, col = 'Temperature', clim = [0, 100], figname=None):
        ''' plot_SAVE
        '''
        elevation_save = np.zeros((3000,1))
        for idblock in range(1,3001):
            elevation_save[idblock-1] = dat.grid.blocklist[idblock].centre[2]
            
        z = np.reshape(elevation_save,(nx[2],nx[1],nx[0]))
        
        data_save=np.array(df[col])
        
        if col == 'Pressure':
            # Gas pressure converted to m H2O + elevation = hydraulic head in water
            a = np.reshape(data_save*0.00010197,(nx[2],nx[1],nx[0])).T + z.T
            #a = np.reshape(data_save,(nx[2],nx[1],nx[0])).T 
            plt.imshow(a[:,y,:].T)
            #plt.colorbar()
            plt.clim(clim[0],clim[1])
            #plt.title('Hydraulic head (m)')
            plt.title('Pressure in Pa')
            plt.xlabel('x(m)')
            plt.ylabel('z(m)')
            plt.colorbar(orientation='horizontal')
            
        elif col == 'Temperature':
            a = np.reshape(data_save,(nx[2],nx[1],nx[0])).T
            plt.imshow(a[:,y,:].T)
            #plt.colorbar()
            plt.clim(clim[0],clim[1])
            plt.title('Temperature in C')
            plt.xlabel('x(m)')
            plt.ylabel('z(m)')
            plt.colorbar(orientation='horizontal')            
        elif col == 'Gas Pressure':
            raise NotImplementedError
        else:
            raise ValueError('it support (Gas )Pressure or Temperature')
        
        if figname is not None:
            plt.savefig(figname+'.png')    
        
        plt.close()

        return

    def modify_infile(self):
        ''' modify_infile
        '''
        f = open("INFILE", "r")
        contents = f.readlines()
        f.close()
        contents.insert(5, 'SEED\n')
        f = open("INFILE", "w")
        contents = "".join(contents)
        f.write(contents)
        f.close()
        
        return

    def observation_model(self,cur_dir,obs_type):
        ''' observation_model
        '''
        if obs_type != 'Temperature' and  obs_type != 'Gas Pressure':
            raise ValueError('obs_type should be either Temperature of Gas Pressure')
        sim_results = self.read_FOFT(cur_dir)
        # order by Element and Time for easy plotting 
        #sim_obs.sort_values(by=['Element','Time']) 
        #obs = sim_obs[obs_type]

        elements = sim_results['Element'].unique()
        elements.sort()
        
        dt = self.t_obs_interval
        obs = []
        for elem in elements:
            t = np.array(sim_results[sim_results['Element']==elem]['Time'])
            strtidx= np.ceil(t.min()/dt).astype(np.int)
            endidx = np.floor(t.max()/dt).astype(np.int)

            t_obs = [dt*i for i in range(strtidx,endidx+1)]
            obs_all = np.array(sim_results[sim_results['Element']==elem][obs_type])
            f = interp1d(t, obs_all)
            obs_at_t_obs = f(t_obs)
            obs.extend(obs_at_t_obs)

        return np.array(obs)


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
    import tough
    import numpy as np
    from time import time


    s = np.loadtxt("true_30_10_10_gau.txt")
    s = s.reshape(-1, 1)
    nx = [30, 10, 10]
    dx = [10., 10., 2.]
    
    # monitoring indices 
    xlocs = np.arange(5,21,2)
    ylocs = np.array([1,3,5,7])
    zlocs = np.array([3,5,7,9])

    params = {'nx':nx,'dx':dx, 'deletedir':False, 'xlocs': xlocs, 'ylocs':ylocs, 'zlocs':zlocs, \
    'obs_type':['Gas Pressure','Temperature'],'t_obs_interval':86400.*5.,\
    'max_timesteps': 9000, 'tstop': 0.32342126E+08, 'const_timestep': 6, 'max_timestep':86400, \
    'absolute_error': 1, 'relative_error': 5.e-6, 'print_interval': 9000, 'timestep_reduction': 3., \
    'gravity': 9.81,'default_incons': [100.e4, 10], \
    'multi_params':{'num_components': 1, 'num_equations':2, 'num_phases':2, 'num_secondary_parameters':6},
    'solver_params':{'type': 5, 'z_precond':1,'o_precond': 0, 'relative_max_iterations':.8,'closure':1.e-7 },
    'output_times_params': {'num_times_specified':2, 'time': [0.8640E+04, 0.32342126E+08]}}

    par = False # parallelization false

    mymodel = tough.Model(params)
    print('(1) single run')

    from time import time
    stime = time()
    simul_obs = mymodel.run(s,par)
    print('simulation run: %f sec' % (time() - stime))

    obs_true = np.loadtxt('obs_true.txt')
    print(np.linalg.norm(obs_true.reshape(-1) - simul_obs.reshape(-1)))
    import sys
    sys.exit(0)

    #obs = np.copy(simul_obs)
    #nobs = obs.shape[0]
    #obs[:nobs/2] = simul_obs[:nobs/2] + 10000.*np.random.randn(nobs/2,1)
    #obs[nobs/2:] = simul_obs[nobs/2:] + 0.5*np.random.randn(nobs/2,1)
    
    #np.savetxt('obs.txt',obs)
    #np.savetxt('obs_pres.txt',obs[:nobs/2])
    #np.savetxt('obs_temp.txt',obs[nobs/2:])

    ncores = 6
    nrelzs = 12

    print('(2) parallel run with ncores = %d' % ncores)
    par = True # parallelization false
    srelz = np.zeros((np.size(s,0),nrelzs),'d')
    for i in range(nrelzs):
        srelz[:,i:i+1] = s + 0.1*np.random.randn(np.size(s,0),1)
    
    stime = time()
    
    simul_obs_all = mymodel.run(srelz,par,ncores = ncores)

    print('simulation run: %f sec' % (time() - stime))
    
    print(simul_obs_all)
