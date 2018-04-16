#! /usr/bin/env python

import numpy as np
import pandas
import glob

from enkf_tools import Mesh_lite

def append_drifter_runs(files,add_z=None):
    dfs = []
    for f in files:
        df = pandas.read_csv(f)
        df = df[np.isfinite(df['u'])]
        df = df[np.isfinite(df['depth'])]
        dfs.append(df)
    dfall = pandas.concat(dfs)
    if add_z is None:
        z_col = np.zeros(dfall['north'].shape,'d')
        dfall = dfall.assign(elev=z_col)
    else:
        assert add_z in dfall
        dfall = dfall.assign(elev=dfall[add_z])
    return dfall

def write_xyzuv(df,filename,header=False):
    df.to_csv(filename,index=False,columns=['east','north','elev','u','v'],header=header,sep=' ')
    return df

def write_observation_locations(df,filename,header=False):
    df.to_csv(filename,index=False,columns=['east','north','elev'],header=header,sep=' ')
    return df

def write_velocity_observations(df,filename,add_header=False):
    space_dim=2
    obs_interleave=np.zeros(space_dim*df['u'].shape[0])
    obs_interleave[0::space_dim]=df['u']
    obs_interleave[1::space_dim]=df['v']
    if add_header:
        np.savetxt(filename,obs_interleave,header='{0:d}'.format(len(obs_interleave)))
    else:
        np.savetxt(filename,obs_interleave)

def select_unique_data_on_mesh(mesh,obs):
    """
    Given a data set of x,y,z d0,d1, ...
    and a mesh 
    makes sure there is only one data set per node in the mesh
    """
    from enkf_tools.buildProblemSimulations import findNearestNode
    
    nrobs = obs.shape[0]
    ndata  = obs.shape[1]-3
    assert ndata > 0, "assumes format x,y,z d1 ... ndata={0}".format(ndata)
    I = []
    for ii in range(nrobs):
        x_obs = obs[ii,0]; y_obs=obs[ii,1]; z_obs=obs[ii,2]
            
        nearestNode = findNearestNode(mesh,np.array([x_obs,y_obs,z_obs]))
        I.append(nearestNode)

    iobs = np.zeros((nrobs,1+ndata))
    iobs[:,0]=I
    iobs[:,1:]=obs[:,3:]

    #now sort the data determine the unique rows
    ordered = np.lexsort(iobs[:,(0,)].T)
    iobs_sorted=iobs[ordered]
    diff=np.diff(iobs_sorted,axis=0)
    ui  =np.ones(len(iobs_sorted),'bool')
    ui[1:]=diff[:,0] !=0.
    ui[0] = False
    
    unique_obs = obs[ui]
    
    return unique_obs,ui#,iobs,iobs_sorted

def select_unique_data_from_dataframe(mesh,df):
    obs=df.as_matrix(columns=['east','north','elev','u','v'])    
    unique_obs,ui= select_unique_data_on_mesh(mesh,obs)

    df_unique=pandas.DataFrame(unique_obs,columns=['east','north','elev','u','v'])

    return df_unique

def extract_unique_observations(files,meshfile,run_label):

    ##append all the data together
    dfall = append_drifter_runs(files,add_z='wse')
    ##write the relevant data to a single file
    xyzuv_file = run_label+'_xyzuv'+'.csv'
    dfall = write_xyzuv(dfall,xyzuv_file,header=True)

    ##read in the mesh to be used
    meshbase = meshfile.split('.')[0]
    meshsuff = meshfile.split('.')[1]
    mesh = Mesh_lite.Mesh2DM(meshbase,suffix=meshsuff)
    
    ##extract the unique data on the mesh.    
    ##should pick the first observation in lexicagraphic order for the coords
    df_unique = select_unique_data_from_dataframe(mesh,dfall)

    ##write out the observation file
    locsfile=run_label+'_locs_unique'+'.csv'
    df_unique = write_observation_locations(df_unique,locsfile)
    
    ##write out the velocity observations
    velfile=run_label+'_velocity_unique'+'.csv'
    df_unique = write_velocity_observations(df_unique,velfile)

    
    return df_unique,dfall,mesh

if __name__ == "__main__":

    meshfile = "Inset_true.3dm"

    runlabel = "collect01_run1"
    files = glob.glob('run1*.csv')

    df_unique,dfall,mesh=pp.extract_unique_observations(files,meshfile,runlable)
    
