import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import mare2dem
from pyPCGA import PCGA
import math

if __name__ == '__main__':  # for windows application

    # model domain and discretization
    nx = ny = 100
    
    m = nx*ny
    N = np.array([nx,ny])
    xmin = np.array([0,0])
    xmax = np.array([2000.0,2000.0])
    pts = None # for regular grids, you don't need to specify pts. 

# covairance kernel and scale parameters
prior_std = 1.0
prior_cov_scale = np.array([500.0,200.0])

def kernel(r): return (prior_std ** 2) * np.exp(-r**2)

# forward model wrapper for pyPCGA
s_true = np.loadtxt('true_100x100.txt')
obs = np.loadtxt('obs.txt')

# prepare interface to run as a function
def forward_model(s, parallelization, ncores=None):
    params = {'nx':nx,'ny':ny}
    model = mare2dem.Model(params)

    if parallelization:
        simul_obs = model.run(s, parallelization, ncores)
    else:
        simul_obs = model.run(s, parallelization)
    return simul_obs

params = {'R': (0.1) ** 2, 'n_pc': 50,
        'maxiter': 4, 'restol': 0.1,
        'matvec': 'FFT', 'xmin': xmin, 'xmax': xmax, 'N': N,
        'prior_std': prior_std, 'prior_cov_scale': prior_cov_scale,
        'kernel': kernel, 'post_cov': "diag",
        'precond': True, 'LM': True,
        'parallel': True, 'linesearch': True,
        'forward_model_verbose': False, 'verbose': True,
        'iter_save': True, 'precision':1.E-4}

# params['objeval'] = False, if true, it will compute accurate objective function
# params['ncores'] = 36, with parallell True, it will determine maximum physcial core unless specified
params['ncores'] = 6

s_init = np.mean(s_true) * np.ones((m, 1))
# s_init = np.copy(s_true) # you can try with s_true!

# initialize

prob = PCGA(forward_model, s_init, pts, params, s_true, obs)
# prob = PCGA(forward_model, s_init, pts, params, s_true, obs, X = X) #if you want to add your own drift X

# run inversion
s_hat, simul_obs, post_diagv, iter_best = prob.Run()

post_std = np.sqrt(post_diagv)

smin = s_true.min()
smax = s_true.max()

fig, ax = plt.subplots(nrows=1,ncols=2,sharey=True)
im0 = ax[0].imshow(s_true.reshape(ny,nx),vmin=smin,vmax=smax, cmap=plt.get_cmap('jet'),extent=(0,2000,2000,0))
ax[0].set_title(r'true $\ln \rho$')
ax[0].set_aspect('equal','box-forced')
ax[1].imshow(s_hat.reshape(ny,nx),vmin=smin,vmax=smax, cmap=plt.get_cmap('jet'),extent=(0,2000,2000,0))
ax[1].set_title(r'estimated $\ln \rho$')
ax[1].set_aspect('equal','box-forced')
fig.colorbar(im0, ax=ax.ravel().tolist(),shrink=0.4)
fig.savefig('est.png')
plt.show()
plt.close(fig)

fig = plt.figure()
plt.title(r'Uncertainty (posterior std) in $\ln \sigma$ estimate)')
im0 = plt.imshow(post_std.reshape(ny,nx), cmap=plt.get_cmap('jet'),extent=(0,2000,2000,0))
plt.gca().set_aspect('equal','box-forced')
fig.colorbar(im0)
fig.savefig('std.png')
plt.show()
plt.close(fig)

nobs = prob.obs.shape[0]
fig = plt.figure()
plt.title('obs. vs simul.')
plt.plot(prob.obs, simul_obs, '.')
plt.xlabel('observed')
plt.ylabel('simulated')
minobs = np.vstack((prob.obs, simul_obs)).min(0)
maxobs = np.vstack((prob.obs, simul_obs)).max(0)
plt.plot(np.linspace(minobs, maxobs, 20), np.linspace(minobs, maxobs, 20), 'k-')
plt.axis('equal')
axes = plt.gca()
axes.set_xlim([math.floor(minobs), math.ceil(maxobs)])
axes.set_ylim([math.floor(minobs), math.ceil(maxobs)])
fig.savefig('obs.png')
plt.close(fig)

fig = plt.figure()
plt.semilogy(np.linspace(1,len(prob.objvals),len(prob.objvals)), prob.objvals, 'r-')
plt.xticks(np.linspace(1,len(prob.objvals),len(prob.objvals)))
plt.title('obj values over iterations')
plt.axis('tight')
fig.savefig('obj.png')
plt.close(fig)
