import matplotlib
#matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
import os
import math
from matplotlib.ticker import ScalarFormatter, FormatStrFormatter
from scipy.io import loadmat
triangles = np.loadtxt("triangles.txt")
meshnode = np.loadtxt("meshnode.txt")
velocity_obs_loc = np.loadtxt("./observation_loc_N250_M8_J1_I10.dat")
nx = 1001
ny = 51

maxs = math.ceil(meshnode[:,2].max())
mins = math.floor(meshnode[:,2].min())

#plt.figure()
#plt.pcolor(meshnode[:,2].reshape(51,1001))
#plt.show()

matplotlib.rcParams.update({'font.size': 16})
plt.figure(figsize=(10.,10.),dpi=300)
ax = plt.gca()

im = plt.tripcolor(meshnode[:,0],meshnode[:,1],triangles,meshnode[:,2], vmax= maxs,vmin = mins,cmap=plt.get_cmap('jet'), label='_nolegend_')
ax.set_xlabel("Easting [m]")
ax.set_ylabel("Northing [m]")
plt.gca().set_aspect('equal', adjustable='box')
cbar = plt.colorbar(im,fraction = 0.025, pad=0.05 )
cbar.set_label('Elevation [m]')
plt.scatter(velocity_obs_loc[:,0],velocity_obs_loc[:,1],c='k', s=0.5,alpha=0.7, label='obs.')
plt.rcParams['axes.axisbelow'] = True
plt.rc('axes', axisbelow=True)
plt.grid()
ax.set_axisbelow(True)
plt.tight_layout()
ax.set_title('True Red Inset Bathymetry')
plt.savefig('./bathymetry_true.png', dpi = 300, bbox_inches='tight', pad_inches=0.0)
#plt.show()
plt.close('all')

obs = np.loadtxt("obs.txt")
nobs = obs.shape[0]

#mydir = "./results/100_20_100_0.05/"
#mydir = "./results/5_80_8_100_0.05/"
mydir = "./"
onlyfiles = [f for f in os.listdir(mydir) if os.path.isfile(os.path.join(mydir, f)) and f[:4] == "shat"]
i = 0

for fname in onlyfiles:
    i = i + 1
    s_hat = np.loadtxt(mydir + 'shat%d.txt' % (i))
    simul_obs = np.loadtxt(mydir + 'simulobs%d.txt' % i)

    if i < 15:
        matplotlib.rcParams.update({'font.size': 16})
        plt.figure(figsize=(10., 10.), dpi=300)
        ax = plt.gca()
        #im = plt.tripcolor(meshnode[:, 0], meshnode[:, 1], triangles, s_hat.reshape(-1), cmap=plt.get_cmap('jet'), vmax= maxs,vmin = mins,label='_nolegend_')
        im = plt.tripcolor(meshnode[:, 0], meshnode[:, 1], triangles, s_hat.reshape(-1), cmap=plt.get_cmap('jet'), label='_nolegend_')
        ax.set_xlabel("Easting [m]")
        ax.set_ylabel("Northing [m]")
        plt.gca().set_aspect('equal', adjustable='box')
        cbar = plt.colorbar(im, fraction=0.025, pad=0.05)
        cbar.set_label('Elevation [m]')
        #plt.scatter(velocity_obs_loc[:, 0], velocity_obs_loc[:, 1], c='k', s=0.5, alpha=0.7, label='obs.')
        plt.rcParams['axes.axisbelow'] = True
        plt.rc('axes', axisbelow=True)
        plt.grid()
        ax.set_axisbelow(True)
        plt.tight_layout()
        ax.set_title(('Estimated Red Inset Bathymetry iter %d, RMSE: %f' % (i,(np.linalg.norm(obs - simul_obs))/math.sqrt(nobs))))
        #ax.set_title(('Estimated Red Inset Bathymetry, RMSE: %f' % ((np.linalg.norm(obs - simul_obs)) / math.sqrt(nobs))))

        #if i == 9:
        plt.savefig(('./bathymetry_%d.png' %i), dpi=300, bbox_inches='tight', pad_inches=0.0)

        #plt.show()
        plt.close('all')

post_std = np.sqrt(np.loadtxt("postv.txt"))

matplotlib.rcParams.update({'font.size': 16})
fig, axes = plt.subplots(1, 2, sharey=True, figsize=(18, 8), dpi=100)
im = axes[0].tripcolor(meshnode[:,0],meshnode[:,1],triangles,meshnode[:,2], vmax= maxs,vmin = mins,cmap=plt.get_cmap('jet'), label='_nolegend_')
axes[0].set_xlabel("Easting [m]")
axes[0].set_ylabel("Northing [m]")
#axes[0].set_aspect('equal', adjustable='box')
#cbar = plt.colorbar(im,fraction = 0.025, pad=0.05 )
axes[0].set_label('Elevation [m]')
plt.rcParams['axes.axisbelow'] = True
plt.rc('axes', axisbelow=True)
axes[0].grid()
axes[0].set_axisbelow(True)
axes[0].set_title('True Red Inset Bathymetry')

im = axes[1].tripcolor(meshnode[:,0],meshnode[:,1],triangles,s_hat.reshape(-1), vmax= maxs,vmin = mins,cmap=plt.get_cmap('jet'), label='_nolegend_')
axes[1].set_xlabel("Easting [m]")
#axes[1].set_ylabel("Northing [m]")
#axes[1].set_aspect('equal', adjustable='box')
cbar = plt.colorbar(im,fraction = 0.025, pad=0.05 )
axes[1].set_label('Elevation [m]')
plt.rcParams['axes.axisbelow'] = True
plt.rc('axes', axisbelow=True)
axes[1].grid()
axes[1].set_axisbelow(True)
axes[1].set_title('Esimtated Red Inset Bathymetry')

plt.tight_layout()
plt.savefig('./bathymetry_comparison.png', dpi = 300, bbox_inches='tight', pad_inches=0.0)
#plt.show()
plt.close('all')

fig = plt.figure(figsize=(10., 10.), dpi=300)
ax = plt.gca()
im = plt.tripcolor(meshnode[:, 0], meshnode[:, 1], triangles, post_std.reshape(-1), cmap=plt.get_cmap('jet'),
                   label='_nolegend_')
ax.set_xlabel("Easting [m]")
ax.set_ylabel("Northing [m]")
plt.gca().set_aspect('equal', adjustable='box')
cbar = plt.colorbar(im, fraction=0.025, pad=0.05)
cbar.set_label('Elevation [m]')
plt.rcParams['axes.axisbelow'] = True
plt.rc('axes', axisbelow=True)
plt.grid()
ax.set_axisbelow(True)
plt.tight_layout()
ax.set_title('Uncertainty Map (std): Red Inset Bathymetry')
plt.savefig('./bathymetry_postv.png', dpi=300, bbox_inches='tight', pad_inches=0.0)
# plt.show()
plt.close(fig)


fig = plt.figure()
plt.title('measured vs simulated velocity, RMSE : %g' % (np.linalg.norm(obs - simul_obs)/np.sqrt(nobs)))
#plt.plot(obs,simul_obs,'r.')
plt.plot(obs[0::2],simul_obs[0::2],'bo')
plt.plot(obs[1::2],simul_obs[1::2],'ro')
plt.legend(('v_x','v_y'))
plt.xlabel('observed vel.')
plt.ylabel('simulated vel.')
minobs = min(obs.min(),simul_obs.min())
maxobs = max(obs.max(),simul_obs.max())
plt.plot(np.linspace(minobs,maxobs,20),np.linspace(minobs,maxobs,20),'k-')
plt.axis('equal')
axes = plt.gca()
axes.set_xlim([math.floor(minobs*100.0)/100.0,math.ceil(maxobs*100.0)/100.0])
axes.set_ylim([math.floor(minobs*100.0)/100.0,math.ceil(maxobs*100.0)/100.0])
plt.gca().set_adjustable("box")
fig.savefig('obs.png', dpi=fig.dpi)
#plt.show()
plt.close(fig)
