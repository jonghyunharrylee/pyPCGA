# pyPCGA

Python library for Principal Component Geostatistical Approach
version 0.2

updates
- Exact preconditioner construction (inverse of cokriging/saddle-point matrix) using generalized eigendecomposition [Lee et al., WRR 2016, Saibaba et al, NLAA 2016]
- Fast predictive model validation using cR/Q2 criteria [Kitanidis, Math Geol 1991] ([Lee et al., 2018 in preparation]) 
- Fast posterior variance/std computation using exact preconditioner

to-do 
- automatic covariance model parameter calibration
- clean up the code with better interface
- link with ERDC-stwave script (currently using a script for bathymetry update) 
- link with FMM (pyFMM3D) and HMatrix (pyHmatrix)
