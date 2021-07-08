import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from pyPCGA import PCGA
import ert
import math
import datetime as dt
import os
import sys
from scipy.io import loadmat

# model domain and discretization
#pts = np.load("pts.npy")
#m = pts.shape[0]

nx = ny = 91
nz = 31
m = nx*ny*nz
N = np.array([nx,ny,nz])
xmin = np.array([0,0,0])
xmax = np.array([90.0,90.0,30.0])
pts = None # for regular grids, you don't need to specify pts. 

input_dir = "./input_files"

# covairance kernel and scale parameters
prior_std = 0.5
prior_cov_scale = np.array([30., 30., 15.])

# observation error prior
obs_std = 15.0
def kernel(r): return (prior_std ** 2) * np.exp(-r)

#n_pc = priord.shape[0]
n_pc = 96

xyz = np.load('xyz.npy')
xyzout = np.load('xyzout.npy')
xyzoutval = -4.6052
elexyz = np.load('elexyz.npy') 

# load true value for comparison purpose
#s_true = np.load('true.txt')
s_true = None
obs = np.loadtxt('obs.txt')

forward_params = {'deletedir':True, 'input_dir':'./input_files/', 'xyz': xyz, \
    'xyzout': xyzout, 'elexyz': elexyz, 'xyzoutval': xyzoutval}

# prepare interface to run as a function
def forward_model(s, parallelization, ncores=None):
    model = ert.Model(forward_params)

    if parallelization:
        simul_obs = model.run(s, parallelization, ncores)
    else:
        simul_obs = model.run(s, parallelization)
    return simul_obs

params = {'R': (obs_std) ** 2, 'n_pc': n_pc,
          'maxiter': 7, 'restol': 0.01,
          'matvec': 'FFT', 'xmin': xmin, 'xmax': xmax, 'N': N, 'kernel': kernel,
          'prior_std': prior_std, 'prior_cov_scale': prior_cov_scale,
          'post_cov': "diag", 'precision': 1.e-8,
          'precond': True, 'LM': True, #'LM_smin' : 1.0, 'LM_smax' : 4.0,
          'parallel': True, 'linesearch': True,
          'forward_model_verbose': False, 'verbose': True,
          'iter_save': True}
          #,'simul_obs_init': simul_obs_init}

# params['objeval'] = False, if true, it will compute accurate objective function
# params['ncores'] = 36, with parallell True, it will determine maximum physcial core unless specified
params['ncores'] = 8 # for 48 cores, only 16 PC evalution with 3 cores on each run

s_init =  np.log(0.015)*np.ones((m,1))
#s_init = np.copy(s_true) # you can try with s_true!



# initialize
prob = PCGA(forward_model, s_init, pts, params, s_true, obs)
# prob = PCGA(forward_model, s_init, pts, params, s_true, obs, X = X) #if you want to add your own drift X

# run inversion
s_hat, simul_obs, post_diagv, iter_best = prob.Run()

# observation mismatch
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

# objective values
fig = plt.figure()
plt.semilogy(range(len(prob.objvals)), prob.objvals, 'r-')
plt.title('obj values over iterations')
plt.axis('tight')
fig.savefig('obj.png', dpi=fig.dpi)
plt.close(fig)
