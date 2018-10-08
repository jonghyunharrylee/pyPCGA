import numpy as np
import math
import matplotlib.pyplot as plt 
from mpl_toolkits.axes_grid1 import make_axes_locatable

s_true = np.loadtxt('true_30_10_10_gau.txt')
s_hat = np.loadtxt('shat5.txt')
obs = np.loadtxt('obs_pres.txt')
simul_obs = np.loadtxt('simulobs5.txt')
post_diagv = np.loadtxt('postv.txt')


nx = np.array([30,10,10])

post_std = np.sqrt(post_diagv)

s_true3d = s_true.reshape([nx[2],nx[1],nx[0]])
s_hat3d = s_hat.reshape([nx[2],nx[1],nx[0]])
post_std = post_std.reshape([nx[2],nx[1],nx[0]])

# plz add p_ref so that it looks real log-permeability
for i in range(0,10,2):
    fig, ax = plt.subplots(nrows=2,ncols=1,sharex=True)
    ax[0].pcolor(s_true3d[i,:,:],vmin=-2.0,vmax=1.5, cmap=plt.get_cmap('jet'))
    ax[0].set_title('true ln(pmx) in layer %0d' %(i))
    
    ax[1].pcolor(s_hat3d[i,:,:],vmin=-2.0,vmax= 1.5, cmap=plt.get_cmap('jet'))
    ax[1].set_title('estimated ln(pmx) in layer %0d' %(i))
    #fig.savefig('est_lay%0d.png' % (i))
    plt.show()
    plt.close(fig)

i = 4
fig = plt.figure()
ax = plt.gca()
im = plt.pcolor(post_std[i,:,:], cmap=plt.get_cmap('jet'))
plt.title('Uncertainty (posterior std) in lnK estimate, layer %d' % (i))
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(im, cax=cax)
ax.set_aspect('equal', 'box')
plt.show()
#fig.savefig('std.png')
plt.close(fig)

# change to head?
interval = 1000000

nobs = obs.shape[0]
fig = plt.figure()
plt.title('obs. vs simul.')
plt.plot(obs, simul_obs, '.')
plt.xlabel('observation')
plt.ylabel('simulation')
minobs = np.vstack((obs, simul_obs)).reshape(-1).min()
maxobs = np.vstack((obs, simul_obs)).reshape(-1).max()

xmin, xmax = math.floor(minobs/interval)*interval, math.ceil(maxobs/interval)*interval

plt.plot(np.linspace(xmin, xmax, 20), np.linspace(xmin, xmax, 20), 'k-')
axes = plt.gca()
axes.set_xlim([xmin,xmax])
axes.set_ylim([xmin,xmax])
axes.xaxis.set_ticks(np.linspace(xmin,xmax,int(xmax/interval)+1))
axes.yaxis.set_ticks(np.linspace(xmin,xmax,int(xmax/interval)+1))
axes.set_aspect('equal', 'box')
#fig.savefig('obs.png')
plt.show()
plt.close(fig)


obs = np.loadtxt('obs.txt')
simul_obs = np.loadtxt('simulobs3.txt')

interval_pres = 1000000
interval_temp = 10
nobs = obs.shape[0]
fig, ax = plt.subplots(nrows=1,ncols=2)

obs_pres = obs[:7400]
simul_obs_pres = simul_obs[:7400]
obs_temp = obs[7400:]
simul_obs_temp = simul_obs[7400:]

ax[0].plot(obs_pres,simul_obs_pres,'.')
ax[0].set_xlabel('observation')
ax[0].set_ylabel('simulation')
ax[0].set_title('pressure')
minobs = np.vstack((obs_pres.reshape(-1), simul_obs_pres.reshape(-1))).reshape(-1).min()
maxobs = np.vstack((obs_pres.reshape(-1), simul_obs_pres.reshape(-1))).reshape(-1).max()
xmin, xmax = math.floor(minobs/interval_pres)*interval_pres, math.ceil(maxobs/interval_pres)*interval_pres
ax[0].plot(np.linspace(xmin, xmax, 20), np.linspace(xmin, xmax, 20), 'k-')
ax[0].set_xlim([xmin,xmax])
ax[0].set_ylim([xmin,xmax])
ax[0].xaxis.set_ticks(np.linspace(xmin,xmax,int(xmax/interval_pres)+1))
ax[0].yaxis.set_ticks(np.linspace(xmin,xmax,int(xmax/interval_pres)+1))
ax[0].set(adjustable='box-forced', aspect='equal')

ax[1].plot(obs_temp,simul_obs_temp,'.')
ax[1].set_xlabel('observation')
ax[1].set_ylabel('simulation')
ax[1].set_title('temperature')
minobs = np.vstack((obs_temp.reshape(-1), simul_obs_temp.reshape(-1))).reshape(-1).min()
maxobs = np.vstack((obs_temp.reshape(-1), simul_obs_temp.reshape(-1))).reshape(-1).max()
xmin, xmax = math.floor(minobs/interval_temp)*interval_temp, math.ceil(maxobs/interval_temp)*interval_temp
ax[1].plot(np.linspace(xmin, xmax, 20), np.linspace(xmin, xmax, 20), 'k-')
ax[1].set_xlim([xmin,xmax])
ax[1].set_ylim([xmin,xmax])
ax[1].xaxis.set_ticks(np.linspace(xmin,xmax,int(xmax/interval_temp)+1))
ax[1].yaxis.set_ticks(np.linspace(xmin,xmax,int(xmax/interval_temp)+1))
ax[1].set(adjustable='box-forced', aspect='equal')

fig.savefig('obs_joint.png')
plt.show()
plt.close(fig)
