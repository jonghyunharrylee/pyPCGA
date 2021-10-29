'''
adapted from Harry
'''
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pyPCGA import PCGA
# import mf
import math
import datetime as dt
import os
import sys
from poro import Model

#print(np.__version__)


# domain parameters
nx = 128
ny = 128

N = np.array([nx, ny])
m = np.prod(N)

x = np.linspace(0., 1., N[0])
y = np.linspace(0., 1., N[1])

xmin = np.array([x[0], y[0]])
xmax = np.array([x[-1], y[-1]])

# forward problem parameters
pts_fem = np.loadtxt('dof_perm_dg0.csv', delimiter=',')
ptx = np.linspace(0,1,nx)
pty = np.linspace(0,1,ny)
logk_idx = np.loadtxt('logk_idx.txt').astype(int)

forward_params = {'ptx': ptx, 'pty': pty, 'pts_fem': pts_fem, 'logk_idx': logk_idx}

# Load files for s_true and obs 
s_true = np.loadtxt('s_true.txt').reshape(-1,1)
obs = np.loadtxt('obs.txt').reshape(-1,1) # gerenated noisy obs from poro.py


# covairance kernel and scale parameters
prior_std = 2.0
prior_cov_scale = np.array([0.1, 0.1])

def kernel(r): return (prior_std ** 2) * np.exp(-r)

XX, YY = np.meshgrid(x, y)
pts = None # for uniform grids, you don't need pts of s 

# prepare interface to run as a function
def forward_model(s, parallelization, ncores=None):
    model = Model(forward_params)
    if parallelization:
        simul_obs = model.run(s, parallelization, ncores)
    else:
        simul_obs = model.run(s, parallelization)

    return simul_obs

params = {'R': (50.0) ** 2, 'n_pc': 96,
          'maxiter': 10, 'restol': 0.1,
          'matvec': 'FFT', 'xmin': xmin, 'xmax': xmax, 'N': N,
          'prior_std': prior_std, 'prior_cov_scale': prior_cov_scale,
          'kernel': kernel, 'post_cov': 'diag',
          'precond': False, 'LM': True,  # 'LM_smin' : -30.0, 'LM_smax' : 5.0, # 'alphamax_LM' : 1.E+5, 
          'parallel': True, 'linesearch': True, #'precision': 1.e-4,
          'forward_model_verbose': True, 'verbose': True,
          'iter_save': True}

#s_init = np.mean(s_true) * np.ones((m, 1))
s_init = -20. * np.ones((m, 1))
# initialize
prob = PCGA(forward_model, s_init, pts, params, s_true, obs)
# prob = PCGA(forward_model, s_init, pts, params, s_true, obs, X = X) #if you want to add your own drift X
# run inversion
s_hat, simul_obs, post_diagv, iter_best = prob.Run()
