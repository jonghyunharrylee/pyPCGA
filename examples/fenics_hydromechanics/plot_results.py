import matplotlib.pyplot as plt
import numpy as np

s_true = np.loadtxt('s_true.txt')

x = np.linspace(0,1,128)
y = np.linspace(0,1,128)
xx, yy = np.meshgrid(x,y)

obs = np.loadtxt('obs.txt')
obs_true = np.loadtxt('obs_true.txt')
nobs = obs.shape[0]


simulobs0 = np.loadtxt('simulobs0.txt')
shat0 = np.loadtxt('shat0.txt')

# for i in range():

i = 7 # change it to the last iter

shat = np.loadtxt('shat%d.txt' % (i))
simulobs = np.loadtxt('simulobs%d.txt' % (i))

fig, ax = plt.subplots(1,2)
pcm = ax[0].pcolormesh(xx,yy,s_true.reshape(128,128),cmap=plt.cm.get_cmap("jet"),vmin=-14, vmax=-7.5)
#fig.colorbar(pcm, ax=ax[0])
ax[0].set_aspect('equal')
ax[0].set_title('True')

# note that we subtract mean(logk_true)
pcm = ax[1].pcolormesh(xx,yy,shat.reshape(128,128)-12.,cmap=plt.cm.get_cmap("jet"),vmin=-14, vmax=-7.5)
#pcm.set_clim(vmin,vmax)

ax[1].set_aspect('equal')
ax[1].set_title('MAP with Gaussian prior')
fig.colorbar(pcm, ax=ax[:], location='bottom')
fig.savefig('best_estimate.png')
plt.show()
#plt.close(fig)

fig = plt.figure()
plt.title('pressure - obs. vs simul., RMSE = %f' % (np.sqrt( ((obs- simulobs)**2).sum()/nobs)))
plt.plot(obs, simulobs, 'b.', label='final iter.')
plt.plot(obs, simulobs0, 'k.', label='initial sol.')
plt.xlabel('observed')
plt.ylabel('simulated')
minobs = np.vstack((obs, simulobs)).min()
maxobs = np.vstack((obs, simulobs)).max()
plt.plot(np.linspace(minobs, maxobs, 20), np.linspace(minobs, maxobs, 20), 'k-')
plt.axis('equal')
plt.legend()

fig.savefig('p_fitting.png')
plt.show()
#plt.close(fig)

fig = plt.figure()
plt.title('pressure - obs. vs simul.')
plt.plot(obs,'rx',label='obs')
plt.plot(simulobs,'bo',label='simul', mfc='none')
plt.xlabel('i-th obs')
plt.ylabel('pressure')
plt.legend()
#fig.savefig('p_fitting_over_idx.png')
plt.show()
#plt.close(fig)

print('RMSE obs0 : %f' % np.sqrt( ((obs- simulobs0)**2).sum()/nobs))
print('RMSE final: %f' % np.sqrt( ((obs- simulobs)**2).sum()/nobs))
