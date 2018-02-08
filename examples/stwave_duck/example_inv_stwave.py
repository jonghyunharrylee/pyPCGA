import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt 
from scipy.io import savemat, loadmat
import numpy as np
import stwave as st
from pyPCGA import PCGA
import math

N = np.array([110,83])
m = np.prod(N) 
dx = np.array([5.,5.])
xmin = np.array([0. + dx[0]/2., 0. + dx[1]/2.])
xmax = np.array([110.*5. - dx[0]/2., 83.*5. - dx[1]/2.])
theta1 = 0.1

# covairance kernel and scale parameters 
# following Hojat's paper
#def kernel(r): return np.exp(-r**2)
def kernel(r): return (theta1**2)*np.exp(-r**2)

theta2 = np.array([9.*5., 18.*5.])
#theta2 = np.array([9.*5., 9.*5.])

x = np.linspace(0. + dx[0]/2., 110*5 - dx[0]/2., N[0])
y = np.linspace(0. + dx[1]/2., 83*5 - dx[0]/2., N[1])
XX, YY = np.meshgrid(x, y)
pts = np.hstack((XX.ravel()[:,np.newaxis], YY.ravel()[:,np.newaxis]))
    
bathyfile = loadmat('true_depth.mat')
s_true = np.float64(bathyfile['true'])
obsfile = loadmat('obs.mat')
obs = np.float64(obsfile['obs'])

# 1st-order polynomial (linear trend)
#X = np.zeros((m,2),'d')
#X[:,0] = 1/np.sqrt(m)
#X[:,1] = pts[:,0]/np.linalg.norm(pts[:,0])

# 2nd-order polynomial 
#X = np.zeros((m,3),'d')
#X[:,0] = 1/np.sqrt(m)
#X[:,1] = pts[:,0]/np.linalg.norm(pts[:,0])
#X[:,2] = pts[:,0]**2/np.linalg.norm(pts[:,0]**2)

# sqrt(x) + c 
#X = np.zeros((m,2),'d')
#X[:,0] = 1/np.sqrt(m)
#X[:,1] = np.sqrt(110.*5. - pts[:,0])/np.linalg.norm(np.sqrt(110.*5. - pts[:,0]))

#
# prepare interface to run as a function
def forward_model(s,parallelization,ncores = None):
    model = st.Model()
    
    if parallelization:
        simul_obs = model.run(s,parallelization,ncores)
    else:
        simul_obs = model.run(s,parallelization)
    return simul_obs

#params = {'R':1.e-2, 'n_pc':70, 'maxiter':10, 'restol':1e-2, 'matvec':'FFT','xmin':xmin, 'xmax':xmax, 'N':N, 'prior_std':theta1,'prior_cov_scale':theta2, 'kernel':kernel, 'postcov':"diag",'parallel':True, 'LM': True, 'linesearch' : True}

params = {'R':1.e-2, 'n_pc':70, 'maxiter':10, 'restol':1e-2, 'matvec':'FFT','xmin':xmin, 'xmax':xmax, 'N':N, 'prior_std':theta1,'prior_cov_scale':theta2, 'kernel':kernel, 'postcov':False,'parallel':True, 'LM': True, 'linesearch' : True}
#params['objeval'] = False, if true, it will compute accurate objective function
#params['ncores'] = 36, with parallell True, it will determine maximum physcial core unless specified

s_init = np.mean(s_true)*np.ones((m,1))
#s_init = np.copy(s_true) # you can try with s_true! 

# initialize
prob = PCGA(forward_model, s_init, pts, params, s_true, obs)
#prob = PCGA(forward_model, s_init, pts, params, s_true, obs, X = X) #if you want to add your own drift X 
# run inversion
s_hat, simul_obs, iter_best, iter_final = prob.Run()

s_hat2d = s_hat.reshape(N[1],N[0])
s_true2d = s_true.reshape(N[1],N[0])
minv = s_true.min()
maxv = s_true.max()

fig, axes = plt.subplots(1,2)
fig.suptitle('prior var. : (%g)^2, n_pc : %d' % (theta1,params['n_pc']))
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
plt.suptitle('prior var.: (%g)^2, n_pc : %d' % (theta1,params['n_pc']))
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

# estimated deterministic trend
Xbeta = np.dot(prob.X,prob.beta_best)
Xbeta2d = Xbeta.reshape(N[1],N[0])

fig, axes = plt.subplots(1,2)
fig.suptitle('transect with prior var.: (%g)^2, n_pc : %d, lx = %f px, ly = %f px' % (theta1, params['n_pc'],theta2[0]/5., theta2[1]/5.))

linex = np.arange(1,111)

line1_true = s_true2d[83-25+1,:]
line1 = s_hat2d[83-25+1,:]
line1_X = Xbeta2d[83-25+1,:]

line2_true = s_true2d[83-45+1,:]
line2 = s_hat2d[83-45+1,:]
line2_X = Xbeta2d[83-45+1,:]

axes[0].plot(linex, np.flipud(-line1_true),'r-', label='True')
axes[0].plot(linex, np.flipud(-line1),'k-', label='Estimated')
axes[0].plot(linex, np.flipud(-line1_X),'b--', label='Drift/Trend')
#axes[0].set_title('(a) 125 m', loc='left')
axes[0].set_title('(a) 25 px', loc='left')
handles, labels = axes[0].get_legend_handles_labels()
axes[0].legend(handles, labels)

axes[1].plot(linex, np.flipud(-line2_true),'r-', label='True')
axes[1].plot(linex, np.flipud(-line2),'k-', label='Estimated')
axes[1].plot(linex, np.flipud(-line2_X),'b--', label='Drift/Trend')
#axes[1].set_title('(b) 225 m', loc='left')
axes[1].set_title('(b) 45 px', loc='left')
handles, labels = axes[1].get_legend_handles_labels()
axes[1].legend(handles, labels)
fig.savefig('transect.png')
plt.close(fig)

nobs = prob.obs.shape[0]
fig = plt.figure()
plt.title('obs. vs simul.')
plt.plot(prob.obs,simul_obs,'.')
plt.xlabel('observation')
plt.ylabel('simulation')
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

fig = plt.figure()
plt.semilogy(range(len(prob.objvals)),prob.objvals,'r-')
plt.title('obj values over iterations')
plt.axis('tight')
fig.savefig('obj.png', dpi=fig.dpi)
plt.close(fig)

fig, axes = plt.subplots(4,4, sharex = True, sharey = True)
fig.suptitle('n_pc : %d' % params['n_pc'])
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