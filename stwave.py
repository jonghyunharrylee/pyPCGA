import os
import numpy as np

'''
three operations
1. write inputs
2. run simul
3. read input
'''

#__all__ =['Parameters','Run']

#class Parameters:
#    def __init__(self):
#        #DATADIMS
#        self.NI = None
#        self.NJ = None
#        self.DX = None
#        self.DY = None
#        self.GridName = None
#        self.NumFlds = None
#        self.NumRecs = None
#        self.DataType = None
#        #IDD
#        self.bathy = None

#        #self.DataDims.NI = None
#        #self.DataDims.NJ = None
#        #self.DataDims.DX = None
#        #self.DataDims.DY = None
#        #self.DataDims.GridName = None
#        #self.DataDims.NumFlds = None
#        #self.DataDims.NumRecs = None
#        #self.DataDims.DataType = None
#        #self.IDD.bathy = None

class Model:
    def __init__(self, bathy, idx = None):
        self.bathy = bathy
        if idx == None:
            self.idx = 0
        else:
            self.idx = idx
        self.homedir = os.path.abspath('./')
        self.inputdir = os.path.join(self.homedir,"./input_files")
        self.deletedir = True

    def create_dir(self):
        
        mydirbase = "./simul/simul"
        mydir = mydirbase + "{0:04d}".format(self.idx)
        mydir = os.path.join(self.homedir, mydir)
        if not os.path.exists(mydir):
            os.makedirs(mydir)
        self.mydir = os.path.abspath(mydir)
        return mydir

    def write_input(self):
        mydir = self.create_dir()
        os.chdir(mydir)
        file = open("./stwave_out.dep","wb")
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
        file.write(b"IDD 1\n")#file.write("%f\n" % self.bathy);#file.write("{}\n".format(self.bathy))
        np.savetxt(file, self.bathy, fmt='%.5f', newline=os.linesep)
        #np.savetxt(file, array1 , delimiter = ',')
        file.close()
        return


    def read_output(self):
        # read outputs 
        # read significant wave height, mean period, mean direction from "stwave_out.wave.out" 
        f = open(os.path.join(self.homedir,'./stwave_out.wave.out'),'r') # wave height, mean wave period, mean wave direction
        
        # DataDims
        for line in f:
            if line == "/":
                break

        # DataSets
        for line in f:
            if line == "/":
                break
        #IDD
        waveinfo = [[float(x) for x in line.strip().split(' ')] for line in f]
        #Hm0, T, alpha
        #%% read peak wave period
        #% read peak wave period from "stwave_out.TP.out" 
        f.close()

        f = open(os.path.join(self.homedir,'./stwave_out.Tp.out'),'r') # wave
        # DataDims
        for line in f:
            if line == "/":
                break

        # DataSets
        for line in f:
            if line == "/":
                break
        # peak wave preiod
        Tp = [[float(x) for x in line.strip().split(' ')] for line in f]
        
        f.close()

        # Read wave number from c2shore.out
        f = open(os.path.join(self.homedir,'./c2shore.out'),'r') # wave
        # DataDims
        for line in f:
            if line == "/":
                break

        # DataSets
        for line in f:
            if line == "/":
                break
        # peak wave preiod
        k = [[float(x) for x in line.strip().split(' ')] for line in f]
        
        f.close()

        # wave speed
        w = 2.*np.pi/Tp
        wlen = np.divide(1.,k)
        wlen[k == 0.] = 0.
        c = wlen*w # wave speed = freq* length

        #%% clean generated input/output files
        #if fm.iclean ~= 0
        #    delete('./*.out');
        #    delete('./stwave_out.dep');
        #end
        return c

    def run(self):
        self.write_input()
        
        import shutil
        for filename in os.listdir(self.inputdir):
            shutil.copy2(os.path.join(self.inputdir,filename),self.mydir)
        
        import subprocess
        subprocess.call(["./stwave","stwave_out.sim"])
        simul_obs = self.read_input()
        # move to the current directory and remove the files 
        os.chdir(self.homedir)
        #if self.deletedir == True:
        #    shutil.rmtree(self.mydir)
        return simul_obs
#if __name__ == '__main__':
    #Testing 
    #params = Parameters()
    #params.bathy = np.zeros((10,10),dtype='d')
    #Model(params)
#import stwave as st
#import numpy as np
#bathy = np.array([3.,2.,1.])
#a = st.Model(bathy)
#a.run()