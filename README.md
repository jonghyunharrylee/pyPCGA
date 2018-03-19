# pyPCGA

Python library for Principal Component Geostatistical Approach

version 0.1

updates
- Exact preconditioner construction (inverse of cokriging/saddle-point matrix) using generalized eigendecomposition [Lee et al., WRR 2016, Saibaba et al, NLAA 2016]
- Fast predictive model validation using cR/Q2 criteria [Kitanidis, Math Geol 1991] ([Lee et al., 2018 in preparation]) 
- Fast posterior variance/std computation using exact preconditioner

version 0.2 will include
- automatic covariance model parameter calibration
- link with FMM (pyFMM3D) and HMatrix (pyHmatrix)

# Credits

pyPCGA is based on Lee et al. [2016] and currently used for Stanford-USACE ERDC project and NSF EPSCoR `Ike Wai project. 

Code contributors include:

* Jonghyun Harry Lee 
* Matthew Farthing

FFT-based matvec code are adapted from Arvind Saibaba's work 

# References

- J Lee, H Yoon, PK Kitanidis, CJ Werth, AJ Valocchi, "Scalable subsurface inverse modeling of huge data sets with an application to tracer concentration breakthrough data from magnetic resonance imaging", Water Resources Research 52 (7), 5213-5231

- AK Saibaba, J Lee, PK Kitanidis, Randomized algorithms for generalized Hermitian eigenvalue problems with application to computing Karhunen–Loève expansion, Numerical Linear Algebra with Applications 23 (2), 314-339

- J Lee, H Ghorbanidehno, MW Farthing, TJ Hesser, EF Darve, PK Kitanidis, Riverine bathymetry imaging with indirect observations, in review

- J Lee, PK Kitanidis, "Large‐scale hydraulic tomography and joint inversion of head and tracer data using the Principal Component Geostatistical Approach (PCGA)", WRR 50 (7), 5410-5427

- PK Kitanidis, J Lee, Principal Component Geostatistical Approach for large‐dimensional inverse problems, WRR 50 (7), 5428-5443
