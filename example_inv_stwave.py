import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt 
from scipy.io import savemat, loadmat
import numpy as np
import stwave as st
from pcga import PCGA

#Testing Linear Inversion using interpolation
N = np.array([110,83])
m = np.prod(N) 
dx = np.array([5.,5.])
xmin = np.array([0. + dx[0]/2., 0. + dx[1]/2.])
xmax = np.array([110.*5. - dx[0]/2., 83.*5. - dx[1]/2.])
# covairance scale parameter
theta = np.array([110.*3, 83.*3])
x = np.linspace(0. + dx[0]/2., 110*5 - dx[0]/2., N[0])
y = np.linspace(0. + dx[1]/2., 83*5 - dx[0]/2., N[1])
X, Y = np.meshgrid(x, y)
pts = np.hstack((X.ravel()[:,np.newaxis], Y.ravel()[:,np.newaxis]))
    
bathyfile = loadmat('true_depth.mat')
bathy = np.float64(bathyfile['true'])
#bathy = bathy.ravel()[:,np.newaxis]
    
# prepare interface to run as a function
def forward_model(s,dir=None):
    model = st.Model()
    if dir is None:
        dir = 0
    simul_obs = model.run(s,dir)
    return simul_obs
    
def forward_model_parallel(s,ncores):
    model = st.Model()
    simul_obs = model.parallel_run(s,ncores)
    return simul_obs 

def kernel(r): return ((0.1)**2)*np.exp(-r)
    
params = {'R':1.e-2, 'n_pc':50, 'maxiter':8, 'restol':1e-4, 'covariance_matvec':'FFT','xmin':xmin, 'xmax':xmax, 'N':N, 'theta':theta, 'kernel':kernel, 'parallel':True, 'obj':True}

#params['ncores'] = 36, with parallell True, it will determine maximum physcial core unless specified

s_init = np.mean(bathy)*np.ones((m,1))
    
# initialize 

prob = PCGA(forward_model, forward_model_parallel, s_init, pts, params, s_true = bathy)
# run inversion
s_hat, beta, simul_obs, iter_final = prob.Run()

s_hat2d = s_hat.reshape(N[1],N[0])
bathy2d = bathy.reshape(N[1],N[0])

fig, axes = plt.subplots(1,2, sharey = True)
im = axes[0].imshow(bathy2d, extent=[0, 110, 0, 83], vmin=-2, vmax=8)
axes[0].set_title('(a) True', loc='left')

axes[1].imshow(s_hat2d, extent=[0, 110, 0, 83], vmin=-2, vmax=8)
axes[1].set_title('(b) Estimate', loc='left')

fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
fig.colorbar(im, cax=cbar_ax)
fig.savefig('best.png', dpi=fig.dpi)
#plt.show()
plt.close(fig)

fig = plt.figure()
plt.plot(prob.obs,simul_obs,'.')
minv = np.vstack((a,b)).min(0)
maxv = np.vstack((a,b)).max(0)
plt.plot(np.linspace(minv,maxv,20),'k-')
plt.axis('equal')
axes = plt.gca()
axes.set_xlim([-2.,8.])
axes.set_ylim([-2.,8.])
fig.savefig('obs.png', dpi=fig.dpi)
#plt.show()
plt.close(fig)
    
fig = plt.figure()
plt.imshow(prob.priorU[:,0].reshape(N[1],N[0]), extent=[0, 110, 0, 83])
cbar = plt.colorbar()
fig.savefig('eigv.png', dpi=fig.dpi)
#plt.show()
plt.close(fig)
    
fig = plt.figure()
plt.plot(prob.priord,'o')
fig.savefig('eig.png', dpi=fig.dpi)
#plt.show()
plt.close(fig) 
