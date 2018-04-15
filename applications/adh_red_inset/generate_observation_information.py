import numpy as np
import os
from shutil import copy2
import setup_Red_Inset

"""
Set the input values for running the problem
"""
##which version of AdH is being used and what is it's path
adh_version = 4
if adh_version >= 5:
    adh_exec = './bin/v5/adh'
    pre_adh_exec = None
else:
    adh_exec = './bin/v4/adh'
    pre_adh_exec = './bin/v4/pre_adh'

##describe the geometry of the problem
grid_gridgen = "./mesh_files/collect01/grid_Inset_02262018"  # name of xy gridgen file
rect_gridgen = "./mesh_files/collect01/rect_Inset_02262018"  # name of rectd gridgen file

##xms mesh representation of the gridgen domain for defining material parameters
mesh_gridgen = "./mesh_files/collect01/Inset_02262018_gridgen.3dm"
adh_bc = "./true_files/collect01/Inset_true_v46.bc"

##filenames running the forward problems
sim_prefix = "./sim_files/collect01/Inset_sim"  # basename of adh mesh and files for simulation
if adh_version < 5:
    #sim_prefix = "./sim_files/collect01/Inset_sim_v46"
    #sim_prefix = "./sim_files/collect01/Inset_sim_v46"
    sim_dir = "./sim_files/collect01_1/"
    sim_prefix = sim_dir + "Inset_sim_v46"
    if not os.path.exists(sim_dir):
        os.makedirs(sim_dir)
    copy2(mesh_gridgen, sim_prefix + '.3dm')
    copy2(adh_bc, sim_prefix + '.bc')

    # requires 3dm

##spatial location of observations
# filename for velocity observation locations
velocity_obs_file = "./observation_files/collect01/observation_loc_N250_M8_J1_I10.dat"  # drifter locations
# filename for elevation observation location
elevation_obs_file = "./observation_files/collect01/observation_loc_none.dat"  # empty

##solution and mesh information for the reference solution
true_soln_file_h5 = './true_files/collect01/Inset_true_p0.h5'
true_soln_meshbase = './true_files/collect01/Inset_true'
if adh_version < 5:
    true_soln_file_h5 = './true_files/collect01/Inset_true_v46_p0.h5'
    true_soln_meshbase = './true_files/collect01/Inset_true_v46'

##instantiate the class that describes the forward problem geometry, boundary conditions, initial conditions
# inflow discharge and free surface elevation at the boundary
Q_b = 965.
z_f = 4.764

forward_prob = setup_Red_Inset.RedRiver(grid_file=grid_gridgen, rect_file=rect_gridgen, mesh_file=mesh_gridgen, initial_free_surface_elevation=z_f)

##write out the base mesh, input file, and initial condition file
forward_prob.writeMesh(sim_prefix)
forward_prob.writeBCFile(sim_prefix)
forward_prob.writeHotFile(sim_prefix)

##get the measurement locations
velocity_obs_loc = np.loadtxt(velocity_obs_file)
elev_obs_loc = np.loadtxt(elevation_obs_file)

##which time step to use in the calculation
ntsim=1
##instantiate the inverse problem which controls the forward model simulation
prm = setup_Red_Inset.RedRiverProblem(forward_prob.mesh,
                                      forward_prob,
                                      velocity_obs_loc,
                                      elev_obs_loc,
                                      ntsim=ntsim,
                                      sim_prefix=sim_prefix,
                                      debug_rigid_lid=False,
                                      AdH_version=adh_version,
                                      pre_adh_path=pre_adh_exec,
                                      adh_path=adh_exec,
                                      true_soln_file_h5=true_soln_file_h5,
                                      true_soln_meshbase=true_soln_meshbase,
                                      Q_b=Q_b,
                                      z_f=z_f)

##write these out in the reference domain or not
write_coords_in_reference_domain = True
#where to write the observations 
obs_dir = os.path.join(os.getcwd(),'observation_files','collect01')

t0 = 0.
x_true = prm.get_true_solution(t0)
#measurment matrix
H = prm.get_measurement_matrix(t0)
x_dummy = x_true.copy()

mesh = forward_prob.mesh
##observations from the true solution
obs = H.dot(x_true)

fortran_base = 1
obs_indices = H.indices.copy()
obs_indices+= fortran_base

obs_file    = os.path.join(obs_dir,'observations.dat')
obsind_file = os.path.join(obs_dir,'observations_indices.dat')
obsloc_file = os.path.join(obs_dir,'observations_coords.dat')
state_file = os.path.join(obs_dir,'state_coords.dat')
print "Saving observations and indices to {0} and {1}".format(obs_file,obsind_file)

assert obs.shape[0] == prm.nrobs
header='{0:d}'.format(prm.nrobs)
np.savetxt(obs_file,obs,header=header,comments='')
assert obs_indices.shape[0] == prm.nrobs
np.savetxt(obsind_file,obs_indices,header=header,comments='',fmt='%d')

#mwf hack, state is stored as
#[z_0,z_1,...,z_{Nn-1},vx_0,vy_0,vx_1,vy_1,...]
#mwf debug
#import pdb
#pdb.set_trace()
velocity_offset = prm.nn    
obs_nodeids = np.where(H.indices < prm.nn, H.indices, (H.indices-velocity_offset)/prm.velocity_dim)
obs_coords = mesh.nodeArray[obs_nodeids,0:2]
if write_coords_in_reference_domain:
    #also compute the logical indices
    obs_I = np.mod(obs_nodeids,mesh.nx)
    obs_J = np.floor_divide(obs_nodeids,mesh.nx)
    dx_index = (forward_prob.geometry.Lx-forward_prob.geometry.x0[0])/float(mesh.nx)
    dy_index = (forward_prob.geometry.Ly-forward_prob.geometry.x0[1])/float(mesh.ny)
    obs_coords[:,0]=forward_prob.geometry.x0[0] + obs_I*dx_index
    obs_coords[:,1]=forward_prob.geometry.x0[1] + obs_J*dy_index


np.savetxt(obsloc_file,obs_coords,header=header,comments='')

#save the coordinates of all the state variables as well
obs_state = np.zeros((prm.ndof,prm.velocity_dim),'d')
obs_state[:velocity_offset,:]=mesh.nodeArray[:,0:prm.velocity_dim]
obs_state[velocity_offset::prm.velocity_dim,:]=mesh.nodeArray[:,0:prm.velocity_dim]
obs_state[velocity_offset+1::prm.velocity_dim,:]=mesh.nodeArray[:,0:prm.velocity_dim]
if write_coords_in_reference_domain:
    #also compute the logical indices
    state_nodeids=np.arange(prm.nn)
    obs_I = np.mod(state_nodeids,mesh.nx)
    obs_J = np.floor_divide(state_nodeids,mesh.nx)
    dx_index = (forward_prob.geometry.Lx-forward_prob.geometry.x0[0])/float(mesh.nx)
    dy_index = (forward_prob.geometry.Ly-forward_prob.geometry.x0[1])/float(mesh.ny)
    #z
    obs_state[:velocity_offset,0] = forward_prob.geometry.x0[0] + obs_I*dx_index
    obs_state[:velocity_offset,1] = forward_prob.geometry.x0[1] + obs_J*dy_index
    #vx
    obs_state[velocity_offset::prm.velocity_dim,0]=forward_prob.geometry.x0[0] + obs_I*dx_index
    obs_state[velocity_offset::prm.velocity_dim,0]=forward_prob.geometry.x0[1] + obs_J*dy_index
    #vy
    obs_state[velocity_offset+1::prm.velocity_dim,0]=forward_prob.geometry.x0[0] + obs_I*dx_index
    obs_state[velocity_offset+1::prm.velocity_dim,1]=forward_prob.geometry.x0[1] + obs_J*dy_index


header_state='{0:d}'.format(prm.ndof)

np.savetxt(state_file,obs_state,header=header_state,comments='')
