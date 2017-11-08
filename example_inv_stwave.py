import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt 
from scipy.io import savemat, loadmat
import numpy as np
import stwave as st
from pcga import PCGA
import math
#Testing Linear Inversion using interpolation
N = np.array([110,83])
m = np.prod(N) 
dx = np.array([5.,5.])
xmin = np.array([0. + dx[0]/2., 0. + dx[1]/2.])
xmax = np.array([110.*5. - dx[0]/2., 83.*5. - dx[1]/2.])

# covairance kernel and scale parameters 
# following Hojat's paper but introduces isotropy
def kernel(r): return np.exp(-r**2)
#theta = np.array([9.*5., 18.*5.])
theta = np.array([18.*5., 54.*5.])

x = np.linspace(0. + dx[0]/2., 110*5 - dx[0]/2., N[0])
y = np.linspace(0. + dx[1]/2., 83*5 - dx[0]/2., N[1])
X, Y = np.meshgrid(x, y)
pts = np.hstack((X.ravel()[:,np.newaxis], Y.ravel()[:,np.newaxis]))
    
bathyfile = loadmat('true_depth.mat')
s_true = np.float64(bathyfile['true'])
obsfile = loadmat('obs.mat')
obs = np.float64(obsfile['obs'])

# prepare interface to run as a function
def forward_model(s,parallelization,ncores = None):
    model = st.Model()
    
    if parallelization:
        simul_obs = model.run(s,parallelization,ncores)
    else:
        simul_obs = model.run(s,parallelization)
    return simul_obs

params = {'R':1.e-2, 'n_pc':50, 'maxiter':8, 'restol':1e-4, 'covariance_matvec':'FFT','xmin':xmin, 'xmax':xmax, 'N':N, 'theta':theta, 'kernel':kernel, 'parallel':True, 'LM': True}

#params['objeval'] = False, if true, it will compute accurate objective function
#params['ncores'] = 36, with parallell True, it will determine maximum physcial core unless specified

s_init = np.mean(s_true)*np.ones((m,1))
#s_init = np.copy(s_true)
# initialize
prob = PCGA(forward_model, s_init, pts, params, s_true, obs)
# run inversion
s_hat, simul_obs, iter_best, iter_final = prob.Run()

s_hat2d = s_hat.reshape(N[1],N[0])
s_true2d = s_true.reshape(N[1],N[0])
minv = s_true.min()
maxv = s_true.max()

fig, axes = plt.subplots(1,2)
im = axes[0].imshow(s_true2d, extent=[0, 110, 0, 83], vmin=math.floor(minv), vmax=math.ceil(maxv), cmap=plt.get_cmap('jet'))
axes[0].set_title('(a) True', loc='left')
axes[0].set_aspect('equal')
axes[1].imshow(s_hat2d, extent=[0, 110, 0, 83], vmin=math.floor(minv), vmax=math.ceil(maxv), cmap=plt.get_cmap('jet'))
axes[1].set_title('(b) Estimate', loc='left')
axes[1].set_aspect('equal')
fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
fig.colorbar(im, cax=cbar_ax)
fig.savefig('best_.png')
plt.close(fig)


fig, axes = plt.subplots(1,2)
im = axes[0].imshow(np.flipud(np.fliplr(-s_true2d)), extent=[0, 110, 0, 83], vmin=-7., vmax=0., cmap=plt.get_cmap('jet'))
axes[0].set_title('(a) True', loc='left')
axes[0].set_aspect('equal')
axes[1].imshow(np.flipud(np.fliplr(-s_hat2d)), extent=[0, 110, 0, 83], vmin=-7., vmax=0., cmap=plt.get_cmap('jet'))
axes[1].set_title('(b) Estimate', loc='left')
axes[1].set_aspect('equal')
fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
fig.colorbar(im, cax=cbar_ax)
fig.savefig('best.png')
plt.close(fig)


fig, axes = plt.subplots(1,2)
line1_true = s_true2d[83-25+1,:]
line1 = s_hat2d[83-25+1,:]
line2_true = s_true2d[83-45+1,:]
line2 = s_hat2d[83-45+1,:]
linex = np.arange(1,111)
axes[0].plot(linex, np.flipud(-line1_true),'r-')
axes[0].plot(linex, np.flipud(-line1),'k-')
axes[0].set_title('(a) 125 m', loc='left')
axes[1].plot(linex, np.flipud(-line2_true),'r-')
axes[1].plot(linex, np.flipud(-line2),'k-')
axes[1].set_title('(b) 225 m', loc='left')
fig.savefig('transect.png')
plt.close(fig)


fig = plt.figure()
plt.plot(prob.obs,simul_obs,'.')
minobs = np.vstack((prob.obs,simul_obs)).min(0)
maxobs = np.vstack((prob.obs,simul_obs)).max(0)
plt.plot(np.linspace(minobs,maxobs,20),np.linspace(minobs,maxobs,20),'k-')
plt.axis('equal')
axes = plt.gca()
axes.set_xlim([math.floor(minobs),math.ceil(maxobs)])
axes.set_ylim([math.floor(minobs),math.ceil(maxobs)])
fig.savefig('obs.png', dpi=fig.dpi)
#plt.show()
plt.close(fig)

fig, axes = plt.subplots(4,4, sharex = True, sharey = True)
for i in range(4):
    for j in range(4):
        axes[i,j].imshow(prob.priorU[:,(i*4+j)*2].reshape(N[1],N[0]), extent=[0, 110, 0, 83])
        axes[i,j].set_title('%d-th eigv' %((i*4+j)*2))
fig.savefig('eigv.png', dpi=fig.dpi)
plt.close(fig)
    
fig = plt.figure()
plt.semilogy(prob.priord,'o')
fig.savefig('eig.png', dpi=fig.dpi)
#plt.show()
plt.close(fig) 
