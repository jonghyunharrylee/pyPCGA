#! /usr/bin/env python

"""
Example for assimilation of a reach on the Savannah river

Uses logically rectangular gridgen mesh
"""
try:
    import proteus
    from proteus import Archiver
    from proteus.Archiver import *
except:
    from enkf_tools import Archiver_lite
    from enkf_tools.Archiver_lite import *
import numpy as np
from enkf_tools import Mesh_lite
import tables,os,subprocess
from enkf_tools import SolutionReader as sr
from enkf_tools import shallow_water_problems as swp
from enkf_tools import adh_drivers,buildProblemSimulations
    
class SavannahRiver(buildProblemSimulations.LogicallyRectangular_Simulation_Base):
    """
    Wrapper for Savannah River problem.

    Assumes have pre-generated bc, and hot files
    Assumes have gridgen mesh representation as well

    Needs a way to specify z in the mesh

    """
    def __init__(self,
                 grid_file='grid_savannah_river',
                 rect_file='rect_savannah_river',
                 initial_free_surface_elevation=97.4):
        self.grid_file=grid_file; self.rect_file=rect_file
        mesh = Mesh_lite.GridgenMesh(self.grid_file)
        self.nx=mesh.nx; self.ny=mesh.ny
        index_rect = np.loadtxt(rect_file)
        ##index space
        x0 = [index_rect[0,0],index_rect[1,0],0.]
        xL = [index_rect[0,1],index_rect[1,1],0.]
        geometry = buildProblemSimulations.LogicallyRectangularGeometry(Lx=xL[0]-x0[0],Ly=xL[1]-x0[1],x0=x0,v_drop=0.)
        buildProblemSimulations.LogicallyRectangular_Simulation_Base.__init__(self,geometry,mesh)
        self.z_f=initial_free_surface_elevation
    def writeHotFile(self,fileprefix):
        """
        Write a hotstart file compatible with this mesh
        """
        pf = open(fileprefix+'.hot','w')
        #write headers 
        pf.write('DATASET\n') 
        pf.write('OBJTYPE "mesh2d"\n') 
        pf.write('BEGSCL\n') 
        pf.write('ND %d\n' % (self.mesh.nNodes_global)) 
        pf.write('NC %d\n' % (self.mesh.nElements_global)) 
        pf.write('NAME IOH\n') 
        pf.write('TS 0 0.00000e+00\n')
        for n in range(0,self.mesh.nNodes_global):
            # use the slope to set the approximate initial depth
            #print "z_f zc v_drop", self.z_f, self.zc[n], self.v_drop
            pf.write('%21.16e\n' % (self.z_f-self.mesh.nodeArray[n,2]))
        pf.write('ENDDS\n')
        pf.close()
       
    #

class SavannahRiverProblem(swp.ShallowWater2D_SimulationBasedProblem_AdH):
    """
    Inversion problem for Savannah River

    """

    def __init__(self,mesh,forward_problem,
                 velocity_obs_loc,
                 elev_obs_loc,
                 nrens=50,
                 ovar=0.01,ivar=0.1,
                 assim_step=1,obs_step=1,
                 T=5.,ntsim=1,L_oc=[100.0,5],                 
                 dt=1.0,
                 initial_ensemble_flag=0,#0 from file, otherwise -- normal
                 ensemble_file='savannah_river_ensemble_lx50_ly10_nrens5.out',
                 true_soln_file_h5='savannah_gridgen_true_p0.h5',
                 true_soln_meshbase= 'savannah_gridgen_true',
                 read_mean_profile=False,
                 mean_profile_meshbase='savannah_gridgen_mean',
                 sim_prefix='savannah_sim',
                 pre_adh_path='../bin/pre_adh',
                 adh_path='../bin/adh',
                 sim_output='/dev/null',
                 debug_with_true_soln=False,
                 debug_rigid_lid=False,
                 Q_b=12.5,
                 z_f=10.0):

        swp.ShallowWater2D_SimulationBasedProblem_AdH.__init__(self,
                                                              mesh,
                                                              velocity_obs_loc,
                                                              elev_obs_loc,
                                                              nrens=nrens,
                                                              ovar=ovar,ivar=ivar,
                                                              assim_step=assim_step,
                                                              obs_step=obs_step,
                                                              T=T,
                                                              ntsim=ntsim,
                                                              L_oc=L_oc,                 
                                                              dt=dt,
                                                              ensemble_file=ensemble_file,
                                                              initial_ensemble_flag=initial_ensemble_flag,
                                                              true_soln_file_h5=true_soln_file_h5,
                                                              true_soln_meshbase=true_soln_meshbase,
                                                              sim_prefix=sim_prefix,
                                                              pre_adh_path=pre_adh_path,
                                                              adh_path=adh_path,
                                                              sim_output=sim_output,
                                                              debug_rigid_lid=debug_rigid_lid,
                                                              Q_b=Q_b,
                                                              z_f=z_f)
       

        ##make sure that bc file and hot start file exist
        for suffix in ['.3dm','.bc','.hot']:
            assert os.path.isfile(sim_prefix+suffix), sim_prefix+suffix+" not found"
        ##make sure that true solution files exist
        assert os.path.isfile(true_soln_file_h5)
        for suffix in ['.3dm','.xmf']:
            assert os.path.isfile(true_soln_meshbase+suffix), true_soln_meshbase+suffix+" not found"

        self.debug_with_true_soln = debug_with_true_soln
        self.read_mean_profile = read_mean_profile
        self.mean_profile_meshbase = mean_profile_meshbase
        self.forward_prob = forward_problem
    def generate_initial_ensemble(self):
        """
        Read the initial ensemble for z assuming generated from external source
        For now grabs true solution directly from hdf5
        assumes true solution stored in Ol_Head and Ol_Velocity at ntsim time step

        By default just return ensemble from file or white noise about true solution,
        but would also like to read in or also try inserting a non-trivial mean that's 
        estimated from sparse data or engineering guidance.
        --------------------------------
        Output

        :param: x_true -- true solution
        :param: E      -- ndof x nrens array with initial ensemble
        :param: t      -- initial time
        """
        t = 0.0
        x_true = self.get_true_solution(t)
        #h values from external program. These are perturbations around mean
        if self.initial_ensemble_flag==0: #readfromfile
            Ensembles = np.loadtxt(self.ensemble_file,skiprows=3)
            scale_perturbation = 1.
            Ensembles = Ensembles.reshape((self.nrens,self.nn))
            Ensembles = Ensembles.T
            Ensembles *= scale_perturbation
 
        else:
            Ensembles = np.zeros((self.nn,self.nrens),'d')
            for i in range(self.nrens):
                Ensembles[0:self.nn,i] = np.random.randn(self.nn)*np.sqrt(self.init_var)

        #
        E = np.zeros((self.ndof,self.nrens),'d')
        for i in range(self.nrens):
            E[:self.nn,i] = Ensembles[:,i]
        if self.read_mean_profile:
            import os
            from enkf_tools import Mesh_lite
            assert os.path.isfile(self.mean_profile_meshbase+'.3dm'),"couldn't find {f}".format(f=self.mean_profile_meshbase+'.3dm')
            base3dm=os.path.basename(self.mean_profile_meshbase)
            dir3dm =os.path.dirname(self.mean_profile_meshbase) 
            mesh_mean = Mesh_lite.Mesh2DM(os.path.join(dir3dm,base3dm.split('.')[0]),suffix='3dm')
            assert mesh_mean.nNodes_global == self.mesh.nNodes_global
            for i in range(self.nrens):
                E[:mesh_mean.nNodes_global,i] += mesh_mean.nodeArray[:,2]
            
        elif self.debug_with_true_soln:
            for i in range(E.shape[1]):
                E[:,i] += x_true + np.random.randn(E.shape[0])*np.sqrt(self.init_var)
        for i in range(self.nrens):
            #mwf debug
            print "Savannah River generate initial ensemble calling compute velocity for member {0}".format(i)
            E[self.nn:,i] = self.compute_velocity(E[:self.nn,i],t)
            #mwf hack
            #np.savetxt('init_E_{:d}.txt'.format(i+1),E[:,i])
        #
        #mwf debug
        #import pdb
        #pdb.set_trace()
        #mwf debug
        #print "Savannah River generate initial ensemble saving initial ensemble"
        #np.savetxt('savannah_river_initial_ensemble.tar.gz',E)
        return x_true,E,t
    def compute_velocity_adh(self,zin,t):
        """
        call adh to compute solution 
        allows forward problem to modify hotstart and bc file if needed
        Input 

        :param: zin -- assumed bathymetry
        :param: t   -- time
        
        Ouput

        :param: v -- velocity
        """

        fileprefix=self.sim_prefix
        self.mesh.nodeArray[:,2]=zin[:]
        self.mesh.writeMesh2DM(fileprefix)
        self.forward_prob.mesh = self.mesh
        self.forward_prob.writeBCFile(fileprefix)
        self.forward_prob.writeHotFile(fileprefix)
        with open(self.sim_output,'w') as fout:
            fail = subprocess.call([self.pre_adh_path,fileprefix],stdout=fout,stderr=fout)
            assert not fail
            fail = subprocess.call([self.adh_path,fileprefix],stdout=fout,stderr=fout)
            assert not fail
        hdf5_soln = tables.open_file(fileprefix+'_p0.h5',mode='r')
        h = sr.read_from_adh_hdf5(hdf5_soln,'Ol_Head',self.ntsim)
        v = sr.read_from_adh_hdf5(hdf5_soln,'Ol_Velocity',self.ntsim)
        hdf5_soln.close()

        return v[:,0:self.velocity_dim].flat[:]


if __name__ == "__main__":

    import os
    import argparse
    parser = argparse.ArgumentParser(description="Run assimimation on 2d regular domain using AdH simulations")
    parser.add_argument("--grid_gridgen",
                        help="name of xy gridgen file",
                        action="store",
                        default="./mesh_files/grid_savannah_river")
    parser.add_argument("--rect_gridgen",
                        help="name of rectd gridgen file",
                        action="store",
                        default="./mesh_files/rect_savannah_river")
    parser.add_argument("--sim_prefix",
                        help="basename of adh mesh and files for simulation",
                        action="store",
                        default="./sim_files/savannah_gridgen_new")
    parser.add_argument("--ovar",
                        help="observation error variance",
                        action="store",
                        type=float,
                        default=0.01)
    parser.add_argument("--seed",
                        help="random number seed",
                        action="store",
                        type=int,
                        default=12345)
    parser.add_argument("-R","--rigid_lid",
                        help="use rigid lid dynamics?",
                        action="store_const",
                        const=True,
                        default=False)
    parser.add_argument("--ensemble_file",
                        help="filename of ensembles in index space",
                        action="store",
                        default="./ensemble_files/savannah_river_ensemble_from_fake_snaps_nx301_ny11_nrens50.dat")
    parser.add_argument("--read_mean_profile",
                        help="use a specified mean profile for the bathymetry",
                        action="store_const",
                        const=True,
                        default=False)
    parser.add_argument("--mean_profile_meshbase",
                        help="base filename of 3dm for mean mesh profile",
                        action="store",
                        default="savannah_gridgen_mean")
    parser.add_argument("--nrens",
                        help="""
number of ensembles, must be consistent with ensemble_file
""",
                        action="store",
                        type=int,
                        default=50)
    parser.add_argument("--velocity_obs_file",
                        help="filename for velocity observation locations",
                        action="store",
                        default="./observation_files/observation_loc_vel.dat")
    parser.add_argument("--elevation_obs_file",
                        help="filename for water surface elevation observation locations",
                        action="store",
                        default="./observation_files/observation_loc_elev.dat")
    parser.add_argument("--observations_only",
                        help="just compute observation values and locations",
                        action="store_const",
                        const=True,
                        default=False)
    opts = parser.parse_args()

    ##use negative discharge to get negative velocities if rigid lid
    Q_b = 6873.5; z_f=97.14

    if opts.rigid_lid:
        Q_b = 1e2

    forward_prob = SavannahRiver(grid_file=opts.grid_gridgen,rect_file=opts.rect_gridgen,
                                 initial_free_surface_elevation=z_f)

    forward_prob.writeMesh(opts.sim_prefix)
    forward_prob.writeBCFile(opts.sim_prefix)
    forward_prob.writeHotFile(opts.sim_prefix)

    mesh =  forward_prob.mesh

    ##
    obs_dir = os.path.dirname(opts.velocity_obs_file)
    velocity_obs_loc = np.loadtxt(opts.velocity_obs_file)
    elev_obs_loc     = np.loadtxt(opts.elevation_obs_file)

    ##convert ensemble file to unstructured representation
    #base file
    ensfile=opts.ensemble_file
    #what it will be called after conversion
    import os,shutil
    infilename=os.path.basename(ensfile)
    if len(infilename.split('.')) > 1:
        outfilename="{0}_uns.{1}".format(infilename.split('.')[0],infilename.split('.')[1])
    else:
        outfilename="{0}_uns".format(infilename.split('.')[0],ny)
    ensfile_uns=os.path.join(os.path.dirname(ensfile),outfilename)

    #by default nersc sample files have y varying fastest
    ens_y_varies_fastest=False
    swp.extract_logically_rectangular_ensemble_to_unstructured(mesh,
                                                               ensfile,ensfile_uns,
                                                               opts.nrens,
                                                               y_varies_fastest=ens_y_varies_fastest)



    true_soln_file_h5='./true_files/savannah_gridgen_true_p0.h5'
    true_soln_meshbase= './true_files/savannah_gridgen_true'


    np.random.seed(opts.seed)
    prm = SavannahRiverProblem(mesh,forward_prob,
                               velocity_obs_loc,
                               elev_obs_loc,
                               nrens=opts.nrens,
                               ovar=opts.ovar,
                               T=1.,
                               ensemble_file=ensfile_uns,
                               initial_ensemble_flag=0, #0 file, 1 white noise
                               sim_prefix=opts.sim_prefix,
                               debug_rigid_lid=opts.rigid_lid,
                               debug_with_true_soln=False,
                               read_mean_profile=opts.read_mean_profile,
                               mean_profile_meshbase=opts.mean_profile_meshbase,
                               true_soln_file_h5=true_soln_file_h5,
                               true_soln_meshbase= true_soln_meshbase,
                               Q_b=Q_b,
                               z_f=z_f)

    ### setup necessary files for pdaf example 
    t0 = 0.
    #save measurement values projected to computational grid. Use mask for non-measured values
    fortran_base = 1
    x_true = prm.get_true_solution(t0)
    H = prm.get_measurement_matrix(t0)
    obs = prm.get_measurement(H,x_true,t0)

    obs_indices = H.indices.copy()
    obs_indices+= fortran_base

    pdaf_obs_file    = os.path.join(obs_dir,'pdaf_observations.dat')
    pdaf_obsind_file = os.path.join(obs_dir,'pdaf_observations_indices.dat')
    pdaf_obsloc_file = os.path.join(obs_dir,'pdaf_observations_coords.dat')
    pdaf_state_file = os.path.join(obs_dir,'pdaf_state_coords.dat')
    print "Saving observations and indices to {0} and {1}".format(pdaf_obs_file,pdaf_obsind_file)

    assert obs.shape[0] == prm.nrobs
    header='{0:d}'.format(prm.nrobs)
    np.savetxt(pdaf_obs_file,obs,header=header,comments='')
    assert obs_indices.shape[0] == prm.nrobs
    np.savetxt(pdaf_obsind_file,obs_indices,header=header,comments='',fmt='%d')

    #mwf hack, state is stored as
    #[z_0,z_1,...,z_{Nn-1},vx_0,vy_0,vx_1,vy_1,...]
    #mwf debug
    #import pdb
    #pdb.set_trace()
    velocity_offset = prm.nn    
    obs_nodeids = np.where(H.indices < prm.nn, H.indices, (H.indices-velocity_offset)/prm.velocity_dim)
    obs_coords = mesh.nodeArray[obs_nodeids,0:2]

    np.savetxt(pdaf_obsloc_file,obs_coords,header=header,comments='')

    #save the coordinates of all the state variables as well
    obs_state = np.zeros((prm.ndof,prm.velocity_dim),'d')
    obs_state[:velocity_offset,:]=mesh.nodeArray[:,0:prm.velocity_dim]
    obs_state[velocity_offset::prm.velocity_dim,:]=mesh.nodeArray[:,0:prm.velocity_dim]
    obs_state[velocity_offset+1::prm.velocity_dim,:]=mesh.nodeArray[:,0:prm.velocity_dim]

    header_state='{0:d}'.format(prm.ndof)

    np.savetxt(pdaf_state_file,obs_state,header=header_state,comments='')
    
    if not opts.observations_only:
        x_true,E,t = prm.generate_initial_ensemble()
        assert E.shape[1] == prm.nrens

        ens_dir = os.path.join('.','inputs_offline')

        for i in range(prm.nrens):
            ens_file = 'ens_{0:d}.txt'.format(i+1)
            np.savetxt(os.path.join(ens_dir,ens_file),E[:,i])



 
