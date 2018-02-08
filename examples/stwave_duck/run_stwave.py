import stwave as st
import numpy as np
from time import time
from scipy.io import savemat, loadmat
bathyfile = loadmat('true_depth.mat')
bathy = bathyfile['true']
par = False # parallelization false

mymodel = st.Model()

#simul_obs = mymodel.run(bathy,par)
#savemat('simul.mat',{'simul_obs':simul_obs})    
par = True # parallelization false
nrelzs =  100
bathyrelz = np.zeros((np.size(bathy,0),nrelzs),'d')
for i in range(nrelzs):
    bathyrelz[:,i:i+1] = bathy + 0.1*np.random.randn(np.size(bathy,0),1)

# use all the physcal cores if not specify ncores
print('- parallel run with all the available physical cores on your computer')
simul_obs_all = mymodel.run(bathyrelz,par)

print(simul_obs_all)

#savemat('simul.mat',{'simul_obs':simul_obs,'bathyrelz':bathyrelz})

