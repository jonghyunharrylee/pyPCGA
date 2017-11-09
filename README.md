# pyPCGA

Python library for Principal Component Geostatistical Approach
version 0.1

For now, it uses a simple STWAVE interface that write bathymetry input for the Hojat's example. 

If you want to test, please build STWAVE on your system and copy it to "input_files" directory. Then run example_inv_stwave.py. Manual will be provided soon. 

todo 
- clean up the code with better interface
- link with ERDC-stwave script (currently using a script for bathymetry update) 
- uncertainty
- cross-validation Q2/cR (extend to automatic cross-validation) 
- Hmatrix and FMM
