import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt 
import numpy as np
import stwave as st
from pyPCGA import PCGA
import math,os
import datetime as dt

# model domain and discretization
Lx = 500.0
Ly = 300.0
nx = 100
ny = 60
x0, y0 = (80., 80.)
z0 = 0.
dx0 = (Lx-x0)/float(nx)

N = np.array([nx,ny])
m = np.prod(N) 
dx = np.array([dx0,dx0])
xmin = np.array([0. + dx[0]/2., 0. + dx[1]/2.])
xmax = np.array([Lx - dx[0]/2., Ly - dx[1]/2.])

t1 = dt.datetime(2016, 07, 22, 19, 30)
t2 = dt.datetime(2016, 07, 22, 20, 30)

# covairance kernel and scale parameters
# following Hojat's paper
prior_std = 2.0
prior_cov_scale = np.array([50., 50.])
def kernel(r): return (prior_std**2)*np.exp(-r**2)

# for plotting
x = np.linspace(dx[0]/2., Lx - dx[0]/2. , N[0])
y = np.linspace(dx[1]/2., Ly - dx[1]/2. , N[1])
XX, YY = np.meshgrid(x, y)
pts = np.hstack((XX.ravel()[:,np.newaxis], YY.ravel()[:,np.newaxis]))

#true bathyemtry interpolated to the stwave grid
uas_dir =  os.path.join('uas','LENKF_uas_c_and_topo_20160722_2030_f_v20180130')
elev_file = os.path.join(uas_dir,'observation_files','true_bathymetry.txt')
s_true = np.loadtxt(elev_file)
s_true = z0-s_true
#observations extracted onto stwave grid
wave_obs_file  = os.path.join(uas_dir,'observation_files','wave_speed_measurements.txt')
topo_obs_file  = os.path.join(uas_dir,'observation_files','topo_measurements.txt')
#observation indices
wave_obs_inds_file  = os.path.join(uas_dir,'observation_files','wave_speed_measurement_cells.txt')
topo_obs_inds_file  = os.path.join(uas_dir,'observation_files','topo_measurement_cells.txt')

include_topo_obs = True

wave_obs = np.loadtxt(wave_obs_file)
wave_obs_inds  = np.loadtxt(wave_obs_inds_file,dtype='i')
obs = wave_obs.copy()
topo_obs_inds = None
if include_topo_obs:
    topo_obs = np.loadtxt(topo_obs_file)
    topo_obs = z0-topo_obs #convert to bathymetry
    obs = np.append(obs,topo_obs)
    topo_obs_inds = np.loadtxt(topo_obs_inds_file,dtype='i')

# if std_waves == std_topo, then use scalar R for now - Harry 20180702
std_waves = 0.3
std_obs   = std_waves
std_topo = 0.05

std_obs = std_waves
if include_topo_obs:# and np.absolute(std_topo-std_waves) > 1.0e-6:
    std_obs = np.ones_like(obs)
    std_obs[:len(wave_obs)]=std_waves
    std_obs[len(wave_obs):]=std_topo

# 1st-order polynomial (linear trend)
X = np.zeros((m,2),'d')
X[:,0] = 1/np.sqrt(m)
X[:,1] = pts[:,0]/np.linalg.norm(pts[:,0])

# 2nd-order polynomial 
#X = np.zeros((m,3),'d')
#X[:,0] = 1/np.sqrt(m)
#X[:,1] = pts[:,0]/np.linalg.norm(pts[:,0])
#X[:,2] = pts[:,0]**2/np.linalg.norm(pts[:,0]**2)

# sqrt(x) + c 
#X = np.zeros((m,2),'d')
#X[:,0] = 1/np.sqrt(m)
#X[:,1] = np.sqrt(nx.*5. - pts[:,0])/np.linalg.norm(np.sqrt(nx.*5. - pts[:,0]))


stwave_params = {'nx': nx, 'ny': ny, 'Lx': Lx, 'Ly': Ly, 'x0': x0, 'y0': y0, 't1': t1, 't2': t2,
                 'wave_speed_obs_indices':wave_obs_inds,
                 'topo_obs_indices':topo_obs_inds,
                 'offline_dataloc':"./input_files/FRF-ocean_waves_8m-array_201607.nc"}
    
# prepare interface to run as a function
def forward_model(s,parallelization,ncores = None):
    model = st.Model(stwave_params)
    
    if parallelization:
        simul_obs = model.run(s,parallelization,ncores)
    else:
        simul_obs = model.run(s,parallelization)
    return simul_obs

params = {'R':std_obs**2, 'n_pc':50,
          'maxiter':10, 'restol':0.01,
          'matvec':'FFT','xmin':xmin, 'xmax':xmax, 'N':N,
          'prior_std':prior_std,'prior_cov_scale':prior_cov_scale,
          'kernel':kernel, 'post_cov':"diag",
          'precond':True, 'LM': True,
          'parallel':True, 'linesearch' : True,
          'forward_model_verbose': False, 'verbose': True,
          'iter_save': True}

#params['objeval'] = False, if true, it will compute accurate objective function
#params['ncores'] = 36, with parallell True, it will determine maximum physcial core unless specified

s_init = np.mean(s_true)*np.ones((m,1))
#s_init = np.copy(s_true); s_init = s_init[:,np.newaxis] # you can try with s_true! 

# initialize
#prob = PCGA(forward_model, s_init, pts, params, s_true, obs)
prob = PCGA(forward_model, s_init, pts, params, s_true, obs, X = X) #if you want to add your own drift X 

# run inversion
s_hat, simul_obs, post_diagv, iter_best = prob.Run()

s_hat2d = s_hat.reshape(N[1],N[0])
s_true2d = s_true.reshape(N[1],N[0])
post_diagv[post_diagv <0.] = 0. # just in case
post_std = np.sqrt(post_diagv)
post_std2d = post_std.reshape(N[1],N[0])

minv = s_true.min()
maxv = s_true.max()

fig, axes = plt.subplots(1,2, figsize=(15,5))
plt.suptitle('prior var.: (%g)^2, n_pc : %d' % (prior_std, params['n_pc']))
im = axes[0].imshow(np.flipud(np.fliplr(-s_true2d)), extent=[0, nx, 0, ny], vmin=-7., vmax=0.,
                    cmap=plt.get_cmap('jet'))
axes[0].set_title('(a) True', loc='left')
axes[0].set_aspect('equal')
axes[0].set_xlabel('Offshore distance (px)')
axes[0].set_ylabel('Alongshore distance (px)')
axes[1].imshow(np.flipud(np.fliplr(-s_hat2d)), extent=[0, nx, 0, ny], vmin=-7., vmax=0., cmap=plt.get_cmap('jet'))
axes[1].set_title('(b) Estimate', loc='left')
axes[1].set_xlabel('Offshore distance (px)')
axes[1].set_aspect('equal')
fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
fig.colorbar(im, cax=cbar_ax)
fig.savefig('best.png')
plt.close(fig)

fig, axes = plt.subplots(1,2, figsize=(15,5))
plt.suptitle('prior var.: (%g)^2, n_pc : %d' % (prior_std, params['n_pc']))
im = axes[0].imshow(np.flipud(np.fliplr(-s_true2d)), extent=[0, nx, 0, ny], vmin=minv, vmax=maxv)
axes[0].set_title('(a) True', loc='left')
axes[0].set_aspect('equal')
axes[0].set_xlabel('Offshore distance (px)')
axes[0].set_ylabel('Alongshore distance (px)')
axes[1].imshow(np.flipud(np.fliplr(-s_hat2d)), extent=[0, nx, 0, ny], vmin=minv, vmax=maxv)
axes[1].set_title('(b) Estimate', loc='left')
axes[1].set_xlabel('Offshore distance (px)')
axes[1].set_aspect('equal')
fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
fig.colorbar(im, cax=cbar_ax)
fig.savefig('best2.png')
plt.close(fig)


fig = plt.figure()
im = plt.imshow(np.flipud(np.fliplr(post_std2d)), extent=[0, nx, 0, ny], cmap=plt.get_cmap('jet'))
plt.title('Uncertainty (std)', loc='left')
plt.xlabel('Offshore distance (px)')
plt.ylabel('Alongshore distance (px)')
plt.gca().set_aspect('equal', adjustable='box')
fig.colorbar(im)
fig.savefig('std.png')
plt.close(fig)

# estimated deterministic trend
# Xbeta = np.dot(prob.X,prob.beta_best)
# Xbeta2d = Xbeta.reshape(N[1],N[0])

fig, axes = plt.subplots(1, 2)
fig.suptitle('transect with prior var.: (%g)^2, n_pc : %d, lx = %f m, ly = %f m' % (
prior_std, params['n_pc'], prior_cov_scale[0], prior_cov_scale[1]))

linex = np.arange(1, nx+1) * dx[0]
line1_true = s_true2d[ny - 25 + 1, :]
line1 = s_hat2d[ny - 25 + 1, :]
line1_u = s_hat2d[ny - 25 + 1, :] + 1.96 * post_std2d[ny - 25 + 1, :]
line1_l = s_hat2d[ny - 25 + 1, :] - 1.96 * post_std2d[ny - 25 + 1, :]
# line1_X = Xbeta2d[ny-25+1,:]

line2_true = s_true2d[ny - 45 + 1, :]
line2 = s_hat2d[ny - 45 + 1, :]
line2_u = s_hat2d[ny - 45 + 1, :] + 1.96 * post_std2d[ny - 45 + 1, :]
line2_l = s_hat2d[ny - 45 + 1, :] - 1.96 * post_std2d[ny - 45 + 1, :]
# line2_X = Xbeta2d[ny-45+1,:]

axes[0].plot(linex, np.flipud(-line1_true), 'r-', label='True')
axes[0].plot(linex, np.flipud(-line1), 'k-', label='Estimated')
axes[0].plot(linex, np.flipud(-line1_u), 'k--', label='95% credible interval')
axes[0].plot(linex, np.flipud(-line1_l), 'k--')
# axes[0].plot(linex, np.flipud(-line1_X),'b--', label='Drift/Trend')
axes[0].set_title('(a) 125 m', loc='left')
# axes[0].set_title('(a) 25 px', loc='left')
handles, labels = axes[0].get_legend_handles_labels()
axes[0].legend(handles, labels)

axes[1].plot(linex, np.flipud(-line2_true), 'r-', label='True')
axes[1].plot(linex, np.flipud(-line2), 'k-', label='Estimated')
axes[1].plot(linex, np.flipud(-line2_u), 'k--', label='95% credible interval')
axes[1].plot(linex, np.flipud(-line2_l), 'k--')
# axes[1].plot(linex, np.flipud(-line2_X),'b--', label='Drift/Trend')
axes[1].set_title('(b) 225 m', loc='left')
# axes[1].set_title('(b) 45 px', loc='left')
handles, labels = axes[1].get_legend_handles_labels()
axes[1].legend(handles, labels)
fig.savefig('transect.png')
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
plt.semilogy(range(len(prob.objvals)), prob.objvals, 'r-')
plt.title('obj values over iterations')
plt.axis('tight')
fig.savefig('obj.png', dpi=fig.dpi)
plt.close(fig)

fig, axes = plt.subplots(4, 4, sharex=True, sharey=True)
fig.suptitle('n_pc : %d' % params['n_pc'])
for i in range(4):
    for j in range(4):
        axes[i, j].imshow(prob.priorU[:, (i * 4 + j) * 2].reshape(N[1], N[0]), extent=[0, nx, 0, ny])
        axes[i, j].set_title('%d-th eigv' % ((i * 4 + j) * 2))
fig.savefig('eigv.png', dpi=fig.dpi)
plt.close(fig)

fig = plt.figure()
plt.semilogy(prob.priord, 'o')
fig.savefig('eig.png', dpi=fig.dpi)
# plt.show()
plt.close(fig)

obs_loc_file  = os.path.join(uas_dir,'observation_files','wave_speed_measurement_locations.txt')
obs_loc = np.loadtxt(obs_loc_file)
topo_obs_loc_file = os.path.join(uas_dir,'observation_files','topo_measurement_locations.txt')
topo_obs_loc = np.loadtxt(topo_obs_loc_file)

fig = plt.figure()
plt.pcolormesh(XX,YY,np.fliplr(-s_true2d), vmin=-7., vmax=0., cmap=plt.get_cmap('jet'))
plt.colorbar()
plt.scatter(Lx-obs_loc[:,0], obs_loc[:,1])
plt.scatter(Lx-topo_obs_loc[:,0], topo_obs_loc[:,1], c='r')
plt.axis([0, Lx, 0, Ly])
plt.gca().set_aspect('equal', adjustable='box')
fig.savefig('obs_loc.png')
plt.close(fig)

