'''
adapted from Section 3.1 in Lee and Kitanidis WRR 2014
Note: this is not a reproduction of the paper!!!
'''
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pyPCGA import PCGA
import mf
import math
import datetime as dt
import os

# model domain and discretization
Lx = 1000.; Ly = 750.; Lz = 1; nlay = 1; nrow = 75; ncol = 100
Q = 25.; Rch = 0.001
ztop = 0.; zbot = -1.

# seems confusing considering flopy notation, remember python array ordering of col, row and lay
N = np.array([ncol, nrow, nlay])
m = np.prod(N)
dx = np.array([10., 10., 1.])
xmin = np.array([0. + dx[0] / 2., 0. + dx[1] / 2., 0. + dx[2] / 2.])
xmax = np.array([Lx - dx[0] / 2., Ly - dx[1] / 2., Lz - dx[2] / 2.])

# parameters
if os.name == 'nt':
    mf_exec = 'mf2005.exe'
else:
    mf_exec = 'mf2005'

input_dir = "./input_files"
sim_dir = './simul'

# location of observations
obs_locmat = np.zeros((nlay, nrow, ncol), np.bool)
for i in range(5, 71, 16):
    for j in range(9, 96, 16):
        obs_locmat[0, i, j] = 1

# Hydraulic tomography - crosswell pumping test setting
Q_locs_idx = np.where(obs_locmat == True)
Q_locs = []
for Q_loc in zip(Q_locs_idx[0], Q_locs_idx[1], Q_locs_idx[2]):
    Q_locs.append(Q_loc)

# covairance kernel and scale parameters
prior_std = 1.0
prior_cov_scale = np.array([200., 200., 1.])

def kernel(r): return (prior_std ** 2) * np.exp(-r)

# for plotting
x = np.linspace(0. + dx[0] / 2., Lx - dx[0] / 2., N[0])
y = np.linspace(0. + dx[1] / 2., Ly - dx[1] / 2., N[1])
XX, YY = np.meshgrid(x, y)
pts = np.hstack((XX.ravel()[:, np.newaxis], YY.ravel()[:, np.newaxis]))

# load true value for comparison purpose
s_true = np.loadtxt('true_logK.txt')
s_true = np.array(s_true).reshape(-1, 1)  # make it 2D array

obs = np.loadtxt('obs.txt')

mf_params = {'mf_exec': mf_exec, 'input_dir': input_dir,
          'sim_dir': sim_dir,
          'Lx': Lx, 'Ly': Ly,
          'Q': Q, 'Rch': Rch,
          'nlay': nlay, 'nrow': nrow, 'ncol': ncol,
          'zbot': zbot, 'ztop': ztop,
          'obs_locmat': obs_locmat, 'Q_locs': Q_locs}

# prepare interface to run as a function
def forward_model(s, parallelization, ncores=None):
    model = mf.Model(mf_params)

    if parallelization:
        simul_obs = model.run(s, parallelization, ncores)
    else:
        simul_obs = model.run(s, parallelization)
    return simul_obs


params = {'R': (0.5) ** 2, 'n_pc': 50,
          'maxiter': 10, 'restol': 0.01,
          'matvec': 'FFT', 'xmin': xmin, 'xmax': xmax, 'N': N,
          'prior_std': prior_std, 'prior_cov_scale': prior_cov_scale,
          'kernel': kernel, 'post_cov': "diag",
          'precond': True, 'LM': True, #'LM_smin' : 1.0, 'LM_smax' : 4.0,
          'parallel': True, 'linesearch': True,
          'forward_model_verbose': False, 'verbose': False,
          'iter_save': True}

# params['objeval'] = False, if true, it will compute accurate objective function
# params['ncores'] = 36, with parallell True, it will determine maximum physcial core unless specified

s_init = np.ones((m, 1))
# s_init = np.copy(s_true) # you can try with s_true!

# initialize
prob = PCGA(forward_model, s_init, pts, params, s_true, obs)
# prob = PCGA(forward_model, s_init, pts, params, s_true, obs, X = X) #if you want to add your own drift X

# run inversion
s_hat, simul_obs, post_diagv, iter_best = prob.Run()

# plotting results
s_hat3d = s_hat.reshape(nlay, nrow, ncol)
s_hat2d = s_hat3d[0,:,:]
s_true3d = s_true.reshape(nlay, nrow, ncol)
s_true2d = s_true3d[0,:,:]

post_diagv[post_diagv < 0.] = 0.  # just in case
post_std = np.sqrt(post_diagv)
post_std3d = post_std.reshape(nlay, nrow, ncol)
post_std2d = post_std3d[0,:,:]

minv = s_true.min()
maxv = s_true.max()

# best result
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
plt.suptitle('prior var.: (%g)^2, n_pc : %d' % (prior_std, params['n_pc']))
im = axes[0].pcolormesh(XX,YY,s_true2d, vmin=minv, vmax=maxv, cmap=plt.get_cmap('jet'))
axes[0].set_title('(a) True', loc='left')
axes[0].set_aspect('equal')
axes[0].set_xlabel('x (m)')
axes[0].set_ylabel('y (m)')
axes[0].axis([XX.min(), XX.max(), YY.min(), YY.max()])
axes[1].pcolormesh(XX, YY, s_hat2d, vmin=minv, vmax=maxv, cmap=plt.get_cmap('jet'))
axes[1].set_title('(b) Estimate', loc='left')
axes[1].set_xlabel('x (m)')
axes[1].set_aspect('equal')
axes[1].axis([XX.min(), XX.max(), YY.min(), YY.max()])

fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
fig.colorbar(im, cax=cbar_ax)
fig.savefig('best.png')
plt.close(fig)

# uncertainty
fig = plt.figure()
im = plt.pcolormesh(XX,YY,post_std2d, cmap=plt.get_cmap('jet'))
plt.axis([XX.min(), XX.max(), YY.min(), YY.max()])
plt.title('Uncertainty (std)', loc='left')
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.gca().set_aspect('equal', adjustable='box')
fig.colorbar(im)
fig.savefig('std.png')
plt.close(fig)

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

fig, axes = plt.subplots(4, 4, sharex=True, sharey=True)
fig.suptitle('n_pc : %d' % params['n_pc'])
for i in range(4):
    for j in range(4):
        tmp3d = prob.priorU[:, (i * 4 + j) * 2].reshape(nlay,nrow,ncol)
        axes[i, j].pcolormesh(XX,YY,tmp3d[0,:,:])
        axes[i, j].set_title('%d-th eigv' % ((i * 4 + j) * 2))
        axes[i, j].axis([XX.min(), XX.max(), YY.min(), YY.max()])

fig.savefig('eigv.png', dpi=fig.dpi)
plt.close(fig)

fig = plt.figure()
plt.semilogy(prob.priord, 'o')
fig.savefig('eig.png', dpi=fig.dpi)
# plt.show()
plt.close(fig)
