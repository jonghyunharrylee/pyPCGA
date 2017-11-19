import os
import numpy as np
from shutil import copy2, rmtree
from subprocess import call, check_output
from time import time

'''
three operations
1. write inputs
2. run simul
3. read input
'''

class Model:
    def __init__(self,params = None):
        self.idx = 0
        self.homedir = os.path.abspath(os.getcwd())
        self.inputdir = os.path.abspath(os.path.join(self.homedir,"input_files"))
        self.stwave_exec= os.path.join(self.homedir,'stwave')
        self.deletedir = True
        self.ncores = 1

        if params is not None: 
            if 'deletedir' in params:
                self.deletedir = params['deletedir']
            if 'homedir' in params:
                self.homedir = params['homedir']
                self.inputdir = os.path.abspath(os.path.join(self.homedir,"input_files"))
            if 'inputdir' in params:
                self.inputdir = params['inputdir']
            if 'ncores' in params:
                self.ncores = params['ncores']
            if 'stwave' in params:
                self.stwave_exec = params['stwave']
        #some sanity checks
        assert os.path.isdir(self.inputdir)
        assert os.path.isdir(self.homedir)
        
    def create_dir(self,idx=None):
        
        mydirbase = os.path.join(os.path.curdir,"simul","simul")
        if idx is None:
            idx = self.idx
        
        mydir = mydirbase + "{0:04d}".format(idx)
        mydir = os.path.abspath(os.path.join(self.homedir, mydir))
        
        if not os.path.exists(mydir):
            os.makedirs(mydir)
        
        for filename in os.listdir(self.inputdir):
            copy2(os.path.join(self.inputdir,filename),mydir)
        
        return mydir

    def write_input(self,bathy,idx=None):
        '''
            create directory, copy relevant files and write input with bathymetry
        '''
        if idx is None:
            mydir = self.create_dir()
        else:
            mydir = self.create_dir(idx)

        file = open(os.path.join(mydir,"stwave_out.dep"),"wb")
        file.write(b"#STWAVE_SPATIAL_DATASET\n")
        # DATA DIMENSION
        file.write(b"&DataDims\n")
        file.write(b"   NI = 110,\n")
        file.write(b"   NJ = 83,\n");
        file.write(b"   DataType = 0,\n");
        file.write(b"   NumRecs = 1,\n");
        file.write(b"   DX = 5.0,\n");
        file.write(b"   DY = 5.0,\n");
        file.write(b"   GridName = \"stwave_out\",\n")
        file.write(b"   NumFlds = 1,\n");
        file.write(b"/\n");
        # Dataset information
        file.write(b"&Dataset\n")
        file.write(b"   FldUnits(1) = ,\n")
        file.write(b"   FldName(1) = \"Depth\",\n")
        file.write(b"/\n")
        # IDD
        file.write(b"IDD 1\n")
        np.savetxt(file, bathy, fmt='%.5f', newline=os.linesep)
        file.close()
        return mydir

    def read_output(self, mydir):
        
        # read outputs from mydir
        assert os.path.isdir(mydir)
        try: 
        # read significant wave height, mean period, mean direction from "stwave_out.wave.out" 
            with open(os.path.join(mydir,'stwave_out.wave.out'),'r') as f:# wave height, mean wave period, mean wave direction
                # DataDims
                for line in f:
                    if line.rstrip() == "/":
                        break

                # DataSets
                for line in f:
                    if line.rstrip() == "/":
                        break

                # read IDD
                f.next()
        
                #IDD
                waveinfo = [[float(x) for x in line.split()] for line in f]
                f.close()
        
                waveinfo = np.array(waveinfo,'d')

                #Hm0, T, alpha
        except IOError:
            print("Could not read file {0}".format(os.path.join(mydir,'stwave_out.wave.out')))

        # read peak wave period from "stwave_out.TP.out" 
        try:
            with open(os.path.join(mydir,'stwave_out.Tp.out'),'r') as f: # wave
                # DataDims
                for line in f:
                    if line.rstrip() == "/":
                        break

                # DataSets
                for line in f:
                    if line.rstrip() == "/":
                        break

                # read IDD
                f.next()
        
                # peak wave preiod
                Tp = [[float(x) for x in line.split()] for line in f]
        
                f.close()
                Tp = np.array(Tp,'d')
                # wave speed
                w = 2.*np.pi/Tp
        
        except IOError:
            print("Could not read file {0}".format(os.path.join(mydir,'stwave_out.Tp.out'))) 


        # Read wave number from c2shore.out
        try:
            with open(os.path.join(mydir,'c2shore.out'),'r') as f: # wave number
                # DataDims
                for line in f:
                    if line.rstrip() == "/":
                        break

                # DataSets
                for line in f:
                    if line.rstrip() == "/":
                        break
        
                # read IDD
                f.next()
        
                # peak wave preiod
                k = [[float(x) for x in line.split()] for line in f]
                f.close()
                k = np.array(k,'d')
                with np.errstate(divide='ignore'):
                    wlen = np.divide(1.,k)
                wlen[k == 0.] = 0.
                c = wlen*w # wave speed = freq* length

                # clean generated input/output files
                if self.deletedir == True:
                    rmtree(mydir)
                return c

        except IOError:
            print("Could not read file {0}".format(os.path.join(mydir,'c2shore.out'))) 

    def run_model(self,simul_dir):
        '''
            run stwave by changing directory to simul_dir, execute, then come back to home directory
        '''
        os.chdir(simul_dir)
        call([self.stwave_exec,"stwave_out.sim"])
        #call(["./stwave","stwave_out.sim"])
        os.chdir(self.homedir)        
        
        # or you can run in shell environment
        #call(["cd " + simul_dir + ";./stwave stwave_out.sim; cd " + self.homedir], shell=True)
        
    def run(self,bathy,parallelization,ncores=None):
        '''
            write inputs, run stwave, read outputs and remove directory if needed
        '''
        nruns = np.size(bathy,1)

        if parallelization:
            if ncores is None:
                from psutil import cpu_count # physcial cpu counts
                ncores = cpu_count(logical=False)

            from joblib import Parallel, delayed  

            # serial reading/writing (may need parallelize this part, but may be ok for now)
            start = time()
            myexcutables = []
            mydirs = []
            for idx in range(nruns):
                mydir = self.write_input(bathy[:,idx:idx+1],idx)
                mydirs.append(mydir)
                myexcutables.append(os.path.join(mydir,"stwave"))
        
            print('-- time for writing stwave input files (sequential) is %g sec' % round(time() - start))

            start = time()
        
            # I don't think running in shell mode is recommended but it works for now.
            Parallel(n_jobs = ncores)(delayed(call)(["cd " + mydirs[idx] + ";{0} stwave_out.sim".format(self.stwave_exec)],shell=True) for idx in range(nruns))
        
            print('-- time for running %d stwave simulations (parallel) on %d cores is %g sec' % (nruns, ncores,round(time() - start)))
        
            start = time()
        
            for idx in range(nruns):
                if idx == 0:
                    simul_obs = self.read_output(mydirs[idx])
                else:            
                    simul_obs = np.concatenate((simul_obs, self.read_output(mydirs[idx])), axis=1)
        
            print('-- time for reading stwave output files (sequentially) is %g sec' % round(time() - start))
        
            assert(np.size(simul_obs,1) == nruns) # should satisfy this

        else:
            mydir = self.write_input(bathy,0)
            self.run_model(mydir)
            simul_obs = self.read_output(mydir)
        
        return simul_obs


if __name__ == '__main__':
    import stwave as st
    import numpy as np
    from time import time
    from scipy.io import savemat, loadmat
    bathyfile = loadmat('true_depth.mat')
    bathy = bathyfile['true']
    par = False # parallelization false

    mymodel = st.Model()
    print('1) single run')

    simul_obs = mymodel.run(bathy,par)
    #savemat('simul.mat',{'simul_obs':simul_obs})    
    ncores = 12
    nrelzs = 36
    
    print('2) parallel run with ncores = %d' % ncores)
    par = True # parallelization false
    bathyrelz = np.zeros((np.size(bathy,0),nrelzs),'d')
    for i in range(nrelzs):
        bathyrelz[:,i:i+1] = bathy + 0.1*np.random.randn(np.size(bathy,0),1)
    
    simul_obs_all = mymodel.run(bathyrelz,par,ncores = ncores)

    print(simul_obs_all)

    # use all the physcal cores if not specify ncores
    print('3) parallel run with all the physical cores')
    simul_obs_all = mymodel.run(bathyrelz,par)

    print(simul_obs_all)

