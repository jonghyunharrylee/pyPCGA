import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import crunch
from pyPCGA import PCGA
import math

if __name__ == '__main__':  # for windows application

    # model domain and discretization
    nx = ny = 50
    t = np.array([1.1574E-05, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7,\
     0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0])

    m = nx*ny
    N = np.array([nx,ny])
    xmin = np.array([0,0])
    xmax = np.array([5,5])
    pts = None # for regular grids, you don't need to specify pts. 

    # covairance kernel and scale parameters
    prior_std = 0.02
    prior_cov_scale = np.array([2.0,2.0])

    def kernel(r): return (prior_std ** 2) * np.exp(-r)

    # forward model wrapper for pyPCGA
    s_true = np.loadtxt('true.txt')
    obs = np.loadtxt('obs.txt')

    # prepare interface to run as a function
    def forward_model(s, parallelization, ncores=None):
        params = {'nx':nx,'ny':ny,'t':t}
        model = crunch.Model(params)

        if parallelization:
            simul_obs = model.run(s, parallelization, ncores)
        else:
            simul_obs = model.run(s, parallelization)
        return simul_obs

    params = {'R': (0.01) ** 2, 'n_pc': 50,
            'maxiter': 10, 'restol': 0.01,
            'matvec': 'FFT', 'xmin': xmin, 'xmax': xmax, 'N': N,
            'prior_std': prior_std, 'prior_cov_scale': prior_cov_scale,
            'kernel': kernel, 'post_cov': "diag",
            'precond': True, 'LM': True,
            'parallel': True, 'linesearch': True,
            'forward_model_verbose': False, 'verbose': True,
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

    post_std = np.sqrt(post_diagv)

    fig, ax = plt.subplots(nrows=1,ncols=2,sharey=True)
    ax[0].pcolor(s_true.reshape(ny,nx),vmin=-33.5,vmax=-28.0)
    ax[0].set_title('true lnK')
    ax[0].set_aspect('equal')
    ax[1].pcolor(s_hat.reshape(ny,nx),vmin=-33.5,vmax=-28.0)
    ax[1].set_title('estimated lnK')
    ax[1].set_aspect('equal')
    fig.savefig('est.png')
    plt.close(fig)

    fig = plt.figure()
    plt.pcolor(post_std.reshape(ny,nx))
    plt.title('Uncertainty (posterior std) in lnK estimate')
    plt.colorbar()
    fig.savefig('std.png')
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
    plt.close(fig)

    fig = plt.figure()
    plt.semilogy(np.linspace(1,len(prob.objvals),len(prob.objvals)), prob.objvals, 'r-')
    plt.xticks(np.linspace(1,len(prob.objvals),len(prob.objvals)))
    plt.title('obj values over iterations')
    plt.axis('tight')
    fig.savefig('obj.png')
    plt.close(fig)

