# pyPCGA

Python library for Principal Component Geostatistical Approach

version 0.1

updates
- Exact preconditioner construction (inverse of cokriging/saddle-point matrix) using generalized eigendecomposition [Lee et al., WRR 2016, Saibaba et al, NLAA 2016]
- Fast hyperparameter tuning and predictive model validation using cR/Q2 criteria [Kitanidis, Math Geol 1991] ([Lee et al., 2021 in preparation]) 
- Fast posterior variance/std computation using exact preconditioner

version 0.2 will include
- automatic covariance model parameter calibration with nearshore application example
- link with [FMM](https://github.com/ruoxi-wang/PBBFMM3D) and HMatrix to support unstructured grids 

# Installation 

```
python -m pip install git+https://github.com/jonghyunharrylee/pyPCGA.git
```

# Courses

* [UH CEE696 course on data assimilation](https://www2.hawaii.edu/~jonghyun/classes/S21/CEE696/)

# Example Notebooks

1D linear inversion example below will be helpful to understand how pyPCGA can be implemented. Please check Google Colab examples. 

* [1D linear inversion example](https://github.com/jonghyunharrylee/pyPCGA/blob/master/examples/pumping_history_identification/linear_inverse_problem_pumping_history_identification.ipynb) (from Stanford 362G course) [Google Colab example](https://colab.research.google.com/drive/13lpxTYgNxOc1gYm2bMvIaTpGFejTVX0r?usp=sharing)

* [1D nonlinear inversion example](https://github.com/jonghyunharrylee/pyPCGA/blob/master/examples/pumping_history_identification/nonlinear_inverse_problem_pumping_history_identification.ipynb) (from Stanford 362G course) [Google Colab example](https://colab.research.google.com/drive/1NPX-q_os5_kVAyFBDOhX_BJJMWUti0u_?usp=sharing)

* [Hydraulic conductivity estimation example using USGS-FloPy (MODFLOW)](https://github.com/jonghyunharrylee/pyPCGA/blob/master/examples/modflow_flopy/inversion_modflow.ipynb) [Lee and Kitanidis, 2014] [Google Colab example](https://colab.research.google.com/drive/1djVDZNjh390czXlzBbu7FvRQne9mf8SP?usp=sharing)

* [Tracer tomography example using Crunch](https://github.com/jonghyunharrylee/pyPCGA/blob/master/examples/tracer_tomography_ade_crunch/inversion_example_advection_diffusion_crunchtope.ipynb) (with Mahta Ansari from UIUC Druhan Lab)

* [Bathymetry estimation example using STWAVE](https://github.com/jonghyunharrylee/pyPCGA/blob/master/examples/stwave_duck/inversion_stwave.ipynb) (with USACE-ERDC-CHL)

* [Permeability estimation example using TOUGH2](https://github.com/jonghyunharrylee/pyPCGA/blob/master/examples/tough_heat/joint_inversion_example_tough.ipynb) (with Amalia Kokkianki, USFCA)

* [Electrical conductivity estimation example using magnetotelluric (MT) survey with MARE2DEM](https://github.com/jonghyunharrylee/pyPCGA/blob/master/examples/mare2dem_MT/inversion_mare2dem.ipynb) (with Niels Grobbe, UHM) 

* [DNAPL plume estimation using hydraulic head, self-potential (SP) and partitioning tracer data](https://github.com/XueyuanK/DNAPL_Inv) (with Xueyuan Kang et al.)

* [ERT example using E4D](https://github.com/jonghyunharrylee/pyPCGA/tree/master/examples/ERT_E4D) will be completed soon.

* MODFLOW-USG/SEAWAT/MODFLOW6 examples coming soon! 

# Credits

pyPCGA is based on Lee et al. [2016] and was used for Stanford-USACE ERDC project led by EF Darve and PK Kitanidis and NSF EPSCoR `Ike Wai project. 

Code contributors include:

* Jonghyun Harry Lee 
* Matthew Farthing
* Ty Hesser (STWAVE example)
* Antoine COLLET
 
FFT-based matvec code is adapted from Arvind Saibaba's work (https://github.com/arvindks/kle). 

FMM-based code (https://arxiv.org/abs/1903.02153) will be incorporated in version 0.2 

# References

- J Lee, H Yoon, PK Kitanidis, CJ Werth, AJ Valocchi, "Scalable subsurface inverse modeling of huge data sets with an application to tracer concentration breakthrough data from magnetic resonance imaging", Water Resources Research 52 (7), 5213-5231

- AK Saibaba, J Lee, PK Kitanidis, Randomized algorithms for generalized Hermitian eigenvalue problems with application to computing Karhunen–Loève expansion, Numerical Linear Algebra with Applications 23 (2), 314-339

- J Lee, PK Kitanidis, "Large‐scale hydraulic tomography and joint inversion of head and tracer data using the Principal Component Geostatistical Approach (PCGA)", WRR 50 (7), 5410-5427

- PK Kitanidis, J Lee, Principal Component Geostatistical Approach for large‐dimensional inverse problems, WRR 50 (7), 5428-5443

# Applications

- T. Kadeethum, D. O'Malley, JN Fuhg, Y. Choi, J. Lee, HS Viswanathan and N. Bouklas, A framework for data-driven solution and parameter estimation of PDEs using conditional generative adversarial networks, Nature Computational Science, 819–829, 2021

- J Lee, H Ghorbanidehno, M Farthing, T. Hesser, EF Darve, and PK Kitanidis, Riverine bathymetry imaging with indirect observations, Water Resources Research, 54(5): 3704-3727, 2018

- J Lee, A Kokkinaki, PK Kitanidis, Fast large-scale joint inversion for deep aquifer characterization using pressure and heat tracer measurements, Transport in Porous Media, 123(3): 533-543, 2018

- PK Kang, J Lee, X Fu, S Lee, PK Kitanidis, J Ruben, Improved Characterization of Heterogeneous Permeability in Saline Aquifers from Transient Pressure Data during Freshwater Injection, Water Resources Research, 53(5): 4444-458, 2017

- S. Fakhreddine, J Lee, PK Kitanidis, S Fendorf, M Rolle, Imaging Geochemical Heterogeneities Using Inverse Reactive Transport Modeling: an Example Relevant for Characterizing Arsenic Mobilization and Distribution, Advances in Water Resources, 88: 186-197, 2016
