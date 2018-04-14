import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, FormatStrFormatter

import numpy as np
from pyPCGA import PCGA
from multiprocessing import Pool
import math
from scipy.io import savemat, loadmat
import adh

adh_params = {'sim_dir': './simul',
              'adh_exec': './bin/v4/adh',
              'pre_adh_exec': './bin/v4/pre_adh',
              'adh_version': 4.5,
              'adh_grid': './mesh_files/nx1001_ny51/grid_Inset_nx1001_ny51',
              'adh_rect': './mesh_files/nx1001_ny51/rect_Inset_nx1001_ny51',
              'adh_mesh': './mesh_files/nx1001_ny51/Inset_nx1001_ny51.3dm',
              'adh_bc': './true_files/nx1001_ny51/Inset_true_v46.bc',
              'adh_ntsim': 2,
              'z_f': 4.8,'Q_f': 700., #free-surface elevation and default inflow
              'velocity_obs_file': './observation_files/observation_loc_N250_M8_J1_I10.dat',
              'elevation_obs_file': './observation_files/observation_loc_none.dat',
              'true_soln_file_h5': './true_files/nx1001_ny51/Inset_true_v46_p0.h5',
              'true_soln_meshbase': './true_files/nx1001_ny51/Inset_true_v46'
              }

adh_params_collect01 = {'sim_dir': './simul',
                        'adh_exec': './bin/v4/adh',
                        'pre_adh_exec': './bin/v4/pre_adh',
                        'adh_version': 4.5,
                        'adh_grid': './mesh_files/collect01/grid_Inset_02262018',
                        'adh_rect': './mesh_files/collect01/rect_Inset_02262018',
                        'adh_mesh': './mesh_files/collect01/Inset_02262018_gridgen.3dm',
                        'adh_bc': './true_files/collect01/Inset_true_v46.bc',
                        'adh_ntsim': 4,
                        'z_f': 4.764,'Q_f': 965., #free-surface elevation and default inflow
                        'velocity_obs_file': './observation_files/collect01/observation_loc_N250_M8_J1_I10.dat',
                        'elevation_obs_file': './observation_files/collect01/observation_loc_none.dat',
                        'true_soln_file_h5': './true_files/collect01/Inset_true_v46_p0.h5',
                        'true_soln_meshbase': './true_files/collect01/Inset_true_v46'
                        }

#which set of simulation parameters to use
sim_params = adh_params_collect01 #adh_params
#where the 'true' solution is
true_file = './mesh_files/collect01/z_Inset_02262018' #true.txt
obs_file  = './observation_files/collect01/observations.dat' #obs.txt
#describe the mesh for visualization
elements_file='./true_files/collect01/triangles.txt'#'triangles.txt'
nodes_file   ='./true_files/collect01/meshnode.txt'#'meshnode.txt'

nx = 1001
ny = 51

N = np.array([nx, ny])
m = np.prod(N)
dx = np.array([1., 1.])

x = np.linspace(0., 1001., N[0])
y = np.linspace(0., 51., N[1])

xmin = np.array([x[0], y[0]])
xmax = np.array([x[-1], y[-1]])

# covairance kernel and scale parameters
prior_std = 5.0
prior_cov_scale = np.array([40., 5.])

def kernel(r): return (prior_std ** 2) * np.exp(-r ** 2)
#def kernel(r): return (prior_std ** 2) * np.exp(-r)


XX, YY = np.meshgrid(x, y)
pts = np.hstack((XX.ravel()[:, np.newaxis], YY.ravel()[:, np.newaxis]))

s_true = np.loadtxt(true_file)
s_true = s_true.reshape(-1, 1)

obs = np.loadtxt(obs_file)
obs = obs.reshape(-1, 1)

s_init = np.mean(s_true) * np.ones((m, 1))


# s_init = np.copy(s_true) # you can try with s_true!

# prepare interface to run as a function
def forward_model(s, parallelization, ncores=None):
    mymodel = adh.Model(sim_params)

    if parallelization:
        if ncores is None:
            from psutil import cpu_count  # physcial cpu counts
            ncores = cpu_count(logical=False)
        simul_obs = mymodel.run(s, parallelization, ncores)
    else:
        simul_obs = mymodel.run(s, parallelization)

    if simul_obs.ndim == 1:
        simul_obs = simul_obs.reshape(-1, 1)

    return simul_obs

params = {'R': (0.05) ** 2, 'n_pc': 100,
          'maxiter': 6, 'restol': 5e-2, #mwf drop from 10 for checking
          'matvec': 'FFT', 'xmin': xmin,
          'xmax': xmax, 'N': N,
          'prior_std': prior_std, 'prior_cov_scale': prior_cov_scale,
          'kernel': kernel, 'post_cov': True, 'precond': True,
          'parallel': True, 'LM': True,
          'linesearch': True,
          'forward_params': sim_params,
          'forward_model_verbose': False, 'verbose': False,
          'iter_save': True
          }

# params['objeval'] = False, if true, it will compute accurate objective function
# params['ncores'] = 36, with parallell True, it will determine maximum physcial core unless specified

# initialize
prob = PCGA(forward_model, s_init, pts, params, s_true, obs)
# prob = PCGA(forward_model, s_init, pts, params, s_true, obs, X = X) #if you want to add your own drift X
# run inversion
s_hat, simul_obs, post_diagv, iter_best = prob.Run()

savemat('results.mat', {'s_hat': s_hat, 'simul_obs': simul_obs,
                        'iter_best': iter_best,
                        'objvals': prob.objvals, 'R': prob.R,
                        'n_pc': prob.n_pc,'matvec':prob.matvec,
                        'prior_std': params['prior_std'], 'prior_cov_scale': params['prior_cov_scale'],
                        'LM': prob.LM, 'linesearch': prob.linesearch,
                        'Q2': prob.Q2_best, 'cR': prob.cR_best,
                        'maxiter': prob.maxiter, 'i_best': prob.i_best,
                        'restol': prob.restol, 'diagv': post_diagv})

#go ahead and save the last solution to a file
np.savetxt('shat.txt',s_hat)

s_hat2d = s_hat.reshape(N[1], N[0])
s_true2d = s_true.reshape(N[1], N[0])
minv = s_true.min()
maxv = s_true.max()

triangles = np.loadtxt(elements_file)
meshnode = np.loadtxt(nodes_file)
velocity_obs_loc = np.loadtxt(sim_params['velocity_obs_file'])

matplotlib.rcParams.update({'font.size': 16})

fig = plt.figure(figsize=(10., 10.), dpi=300)
ax = plt.gca()
im = plt.tripcolor(meshnode[:, 0], meshnode[:, 1], triangles, meshnode[:, 2], cmap=plt.get_cmap('jet'),
                   label='_nolegend_')
ax.set_xlabel("Easting [m]")
ax.set_ylabel("Northing [m]")
plt.gca().set_aspect('equal', adjustable='box')
cbar = plt.colorbar(im, fraction=0.025, pad=0.05)
cbar.set_label('Elevation [m]')
plt.scatter(velocity_obs_loc[:, 0], velocity_obs_loc[:, 1], c='k', s=0.5, alpha=0.7, label='obs.')
plt.rcParams['axes.axisbelow'] = True
plt.rc('axes', axisbelow=True)
plt.grid()
ax.set_axisbelow(True)
plt.tight_layout()
ax.set_title('True Red Inset Bathymetry')
plt.savefig('./bathymetry_true.png', dpi=300, bbox_inches='tight', pad_inches=0.0)
# plt.show()
plt.close(fig)

fig = plt.figure(figsize=(10., 10.), dpi=300)
ax = plt.gca()
im = plt.tripcolor(meshnode[:, 0], meshnode[:, 1], triangles, s_hat.reshape(-1), cmap=plt.get_cmap('jet'),
                   label='_nolegend_')
ax.set_xlabel("Easting [m]")
ax.set_ylabel("Northing [m]")
plt.gca().set_aspect('equal', adjustable='box')
cbar = plt.colorbar(im, fraction=0.025, pad=0.05)
cbar.set_label('Elevation [m]')
plt.rcParams['axes.axisbelow'] = True
plt.rc('axes', axisbelow=True)
plt.grid()
ax.set_axisbelow(True)
plt.tight_layout()
ax.set_title('Estimated Red Inset Bathymetry')
plt.savefig('./bathymetry_estimate.png', dpi=300, bbox_inches='tight', pad_inches=0.0)
# plt.show()
plt.close(fig)

fig = plt.figure(figsize=(10., 10.), dpi=300)
ax = plt.gca()
im = plt.tripcolor(meshnode[:, 0], meshnode[:, 1], triangles, post_diagv.reshape(-1), cmap=plt.get_cmap('jet'),
                   label='_nolegend_')
ax.set_xlabel("Easting [m]")
ax.set_ylabel("Northing [m]")
plt.gca().set_aspect('equal', adjustable='box')
cbar = plt.colorbar(im, fraction=0.025, pad=0.05)
cbar.set_label('Elevation [m]')
plt.rcParams['axes.axisbelow'] = True
plt.rc('axes', axisbelow=True)
plt.grid()
ax.set_axisbelow(True)
plt.tight_layout()
ax.set_title('Uncertainty Map (std): Red Inset Bathymetry')
plt.savefig('./bathymetry_postv.png', dpi=300, bbox_inches='tight', pad_inches=0.0)
# plt.show()
plt.close(fig)

nobs = prob.obs.shape[0]
fig = plt.figure()
plt.title('obs. vs simul.')
plt.plot(prob.obs, simul_obs, '.')
plt.xlabel('observation')
plt.ylabel('simulation')
minobs = np.vstack((prob.obs, simul_obs)).min(0)
maxobs = np.vstack((prob.obs, simul_obs)).max(0)
plt.plot(np.linspace(minobs, maxobs, 20), np.linspace(minobs, maxobs, 20), 'k-')
plt.axis('equal')
axes = plt.gca()
axes.set_xlim([math.floor(minobs), math.ceil(maxobs)])
axes.set_ylim([math.floor(minobs), math.ceil(maxobs)])
fig.savefig('obs.png', dpi=fig.dpi)
# plt.show()
plt.close(fig)

fig = plt.figure()
plt.semilogy(range(len(prob.objvals.flat[:])), prob.objvals.flat[:], 'r-')
plt.title('obj values over iterations')
plt.axis('tight')
fig.savefig('obj.png', dpi=fig.dpi)
plt.close(fig)
