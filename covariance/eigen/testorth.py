import os as _os
import sys as _sys

parent = _os.path.abspath('../')
if parent not in _sys.path:
        _sys.path.append(parent)
del _sys, _os


import numpy as np
from matplotlib import pyplot as plt
from covariance import CovarianceMatrix, Matern
from time import time
from scipy.linalg import eigh, cholesky, inv, svd
import math
from scipy.sparse.linalg import LinearOperator


import matplotlib
matplotlib.rcParams['xtick.labelsize'] = 12.
matplotlib.rcParams['ytick.labelsize'] = 12.

class Matrix:
	def __init__(self,mat):
		self.mat   = mat
		self.shape = mat.shape
		self.dtype = mat.dtype

	def matvec(self,x):
		return np.dot(self.mat,x)

	def solve(self, x):
		return np.linalg.solve(self.mat,x)

def kle1d(kernel, a = 1.):
	from dolfin import UnitInterval, FunctionSpace
	from kle import KLE
	

	#Shift them to domain [-a, a]	
	mesh = UnitInterval(200)
	mesh.coordinates()[:] = 2.*a*mesh.coordinates()[:] - a

	pts  = mesh.coordinates()
        V    = FunctionSpace(mesh, "Lagrange", 1)

	Q = CovarianceMatrix('Dense',pts,kernel)

        kle = KLE(pts, Q, V)
        kle.BuildMassMatrix()


	return Q, kle.M.todense() 



def test_orthogonalization(k):
	from eigen import mgs, mgs_stable, precholqr


	kernel = Matern(p = 0, l = 0.4)

	Q, M =  kle1d(kernel)
	Mop = Matrix(M)
	Omega = np.random.randn(M.shape[0],k)
	

	QM = np.dot(Q.mat,M)
	Y = np.dot(QM, Omega)
	_,_,_ = precholqr(Mop, Y, verbose = True)
	_,_,_ = mgs_stable(Mop, Y,verbose = True)
	_,_,_ = mgs(Mop, Y, verbose = True)


	kernel = Matern(p = 1, l = 0.4)
	Q, M =  kle1d(kernel)
	QM = np.dot(Q.mat,M)
	Y = np.dot(QM, Omega)
	_,_,_ = precholqr(Mop, Y, verbose = True)
	_,_,_ = mgs_stable(Mop, Y,verbose = True)
	_,_,_ = mgs(Mop, Y, verbose = True)
	

	kernel = Matern(p = 2, l = 0.4)
	Q, M =  kle1d(kernel)
	QM = np.dot(Q.mat,M)
	Y = np.dot(QM, Omega)
	_,_,_ = precholqr(Mop, Y, verbose = True)
	_,_,_ = mgs_stable(Mop, Y,verbose = True)
	_,_,_ = mgs(Mop, Y, verbose = True)
	
	return


if __name__ == '__main__':

	
	test_orthogonalization(100)

