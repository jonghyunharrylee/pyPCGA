import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from scipy.io import savemat, loadmat
import numpy as np
import drawdown as dd
from pyPCGA import PCGA
import math

# model domain and discretization
m = 10001
N = np.array([m])
dx = np.array([0.1])
xmin = np.array([0])
xmax = np.array([1000])

# covairance kernel and scale parameters

prior_std = 0.04
prior_cov_scale = np.array([200.0])

def kernel(r): return (prior_std ** 2) * np.exp(-r)

# for plotting
x = np.linspace(xmin, xmax, m)
pts = np.copy(x)

s_true = np.loadtxt('true.txt')
obs = np.loadtxt('obs.txt')

# prepare interface to run as a function
def forward_model(s, parallelization, ncores=None):
    params = {}
    model = dd.Model(params)

    if parallelization:
        simul_obs = model.run(s, parallelization, ncores)
    else:
        simul_obs = model.run(s, parallelization)
    return simul_obs

params = {'R': (0.04) ** 2, 'n_pc': 50,
          'maxiter': 10, 'restol': 0.01,
          'matvec': 'FFT', 'xmin': xmin, 'xmax': xmax, 'N': N,
          'prior_std': prior_std, 'prior_cov_scale': prior_cov_scale,
          'kernel': kernel, 'post_cov': "diag",
          'precond': True, 'LM': True,
          'parallel': True, 'linesearch': True,
          'forward_model_verbose': False, 'verbose': False,
          'iter_save': True}

# params['objeval'] = False, if true, it will compute accurate objective function
# params['ncores'] = 36, with parallell True, it will determine maximum physcial core unless specified

s_init = np.mean(s_true) * np.ones((m, 1))
# s_init = np.copy(s_true) # you can try with s_true!

# initialize
prob = PCGA(forward_model, s_init, pts, params, s_true, obs)
# prob = PCGA(forward_model, s_init, pts, params, s_true, obs, X = X) #if you want to add your own drift X

# run inversion
s_hat, simul_obs, post_diagv, iter_best = prob.Run()

post_diagv[post_diagv < 0.] = 0.  # just in case
post_std = np.sqrt(post_diagv)

fig = plt.figure()
plt.plot(x,s_hat,'k-',label='estimated')
plt.plot(x,s_hat + 2.*post_std,'k--',label='95%')
plt.plot(x,s_hat - 2.*post_std,'k--',label='')
plt.plot(x,s_true,'r-',label='true')
plt.title('pumping history')
plt.xlabel('time (min)')
plt.ylabel(r's ($m^3$/min)')
plt.legend()
fig.savefig('best.png')
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
fig.savefig('obs.png')
# plt.show()
plt.close(fig)

fig = plt.figure()
plt.semilogy(np.linspace(1,len(prob.objvals),len(prob.objvals)), prob.objvals, 'r-')
plt.xticks(np.linspace(1,len(prob.objvals),len(prob.objvals)))
plt.title('obj values over iterations')
plt.axis('tight')
fig.savefig('obj.png')
plt.close(fig)

