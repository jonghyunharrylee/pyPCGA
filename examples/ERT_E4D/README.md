#### pyPCGA E4D example

This is code example from AGU 2014 presentation by Peter Kitanidis. The code is rewritten in python for ICEG2021.

The code needs to be cleaned and updated with notebook example for reproducible results. For now, you need to adjust the number of cores used in E4D and pyPCGA. I will update this soon. 

To run the code, you need to do followings:

1. install pyPCGA
2. install E4D (https://github.com/pnnl/E4D). Installation guide (https://github.com/pnnl/E4D/blob/master/Installation.txt)
3. copy E4D's executables (e4d and mpirun) in the 'input_files' folder 
4. change inv_ert.py params['ncores'] = the number of your cores/the number of coresd used in E4D (set to 6) for the best performance. 
5. run! 
 
