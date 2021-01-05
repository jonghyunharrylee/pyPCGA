#! /usr/bin/env python

"""
Wrap Savannah River test problem for PCGA

"""
import numpy as np
import setup_savannah

"""
Set the input values for running the problem
"""
##describe the geometry of the problem
grid_gridgen= "./mesh_files/grid_savannah_river_nx501_ny41" #name of xy gridgen file
rect_gridgen= "./mesh_files/rect_savannah_river_nx501_ny41" #name of rectd gridgen file

##filenames running the forward problems
sim_prefix= "./sim_files/savannah_gridgen_new_nx501_ny41"   #basename of adh mesh and files for simulation

##spatial location of observations
#filename for velocity observation locations
velocity_obs_file= "./observation_files/observation_loc_drogue12345_50ft.dat" #drifter locations
#filename for elevation observation location
elevation_obs_file= "./observation_files/observation_loc_none.dat" #empty 

##solution and mesh information for the reference solution
true_soln_file_h5='./true_files/savannah_gridgen_true_nx501_ny41_p0.h5'
true_soln_meshbase= './true_files/savannah_gridgen_true_nx501_ny41'

##instantiate the class that describes the forward problem geometry, boundary conditions, initial conditions
#inflow discharge and free surface elevation at the boundary
Q_b = 6873.5; z_f=97.14

forward_prob = setup_savannah.SavannahRiver(grid_file=grid_gridgen,rect_file=rect_gridgen,
                                            initial_free_surface_elevation=z_f)

##write out the base mesh, input file, and initial condition file
forward_prob.writeMesh(sim_prefix)
forward_prob.writeBCFile(sim_prefix)
forward_prob.writeHotFile(sim_prefix)

##get the measurement locations
velocity_obs_loc = np.loadtxt(velocity_obs_file)
elev_obs_loc     = np.loadtxt(elevation_obs_file)

##instantiate the inverse problem which controls the forward model simulation
prm = setup_savannah.SavannahRiverProblem(forward_prob.mesh,
                                          forward_prob,
                                          velocity_obs_loc,
                                          elev_obs_loc,
                                          sim_prefix=sim_prefix,
                                          debug_rigid_lid=False,
                                          pre_adh_path='../bin/pre_adh',
                                          adh_path='../bin/adh',
                                          true_soln_file_h5=true_soln_file_h5,
                                          true_soln_meshbase= true_soln_meshbase,
                                          Q_b=Q_b,
                                          z_f=z_f)
##go ahead and evaluate once
t0 = 0.
#true solution
x_true = prm.get_true_solution(t0)
#measurment matrix
H_meas = prm.get_measurement_matrix(t0)
x_dummy = x_true.copy()

def get_measurements():
    """
    Returns actual measurements for the problem and the degree of freedom indices 
    Here we are using synthetic measurements
    """
    obs = prm.get_measurement(H_meas,x_true,t0)
    return obs

def run_forward_model(z_in):
    """
    Run forward model and return approximate measured values
    """
    x_dummy[:prm.nn]=z_in
    x_dummy[prm.nn:]=prm.compute_velocity(z_in,t0)

    x_meas = H_meas.dot(x_dummy)

    return x_meas

if __name__ == "__main__":

    import os
    ##write out indices so that we can compare with setup_savannah
    fortran_base= 1
    obs_dir = os.path.dirname(velocity_obs_file)
    
    pdaf_obs_file    = os.path.join(obs_dir,'pcga_pdaf_observations.dat')
    pdaf_obsind_file = os.path.join(obs_dir,'pcga_pdaf_observations_indices.dat')

    obs = get_measurements()
    obs_indices = H_meas.indices.copy()
    obs_indices += fortran_base
 
    assert obs.shape[0] == prm.nrobs
    header='{0:d}'.format(prm.nrobs)
    np.savetxt(pdaf_obs_file,obs,header=header,comments='')
    assert obs_indices.shape[0] == prm.nrobs
    np.savetxt(pdaf_obsind_file,obs_indices,header=header,comments='',fmt='%d')

    ##now run the forward model with the true bathymetry, calculate the measurements and compare the difference
    x_meas = run_forward_model(x_true[:prm.nn])

    diff = x_meas - obs

    print "max difference between true and calcuated observations = {0:5.5e}".format(np.max(np.absolute(diff)))
    
