import numpy as np
from scipy.sparse.linalg import LinearOperator, aslinearoperator

from lowrank import LowRankGHEP, LowRankAddition 
from orth import *

def construct_mat(n = 100):

	x = np.linspace(0.,1.,n)
	X,Y = np.meshgrid(x,x)
	Q = np.exp(-np.abs(X-Y))

	return Q

class Matrix:
	def __init__(self, Q):
		self.Q = Q
		self.dtype = Q.dtype 
		self.shape = Q.shape
	def matvec(self,x):
		return np.dot(self.Q,x)

	def solve(self,x):
		return np.linalg.solve(self.Q,x)




class Inverse:
	def __init__(self, Q):
		self.Q = Q
		self.dtype = Q.dtype 
		self.shape = Q.shape
	def matvec(self,x):
		return np.linalg.solve(self.Q,x)

	def solve(self,x):
		return np.dot(self.Q,x)


def testlowrankaddition(Q):

	Qop = Inverse(Q)
	u = np.random.randn(Q.shape[0],10)
	u, _, _ = mgs_stable(Qop, u)
	v = np.random.randn(Q.shape[0],8)
	v, _, _ = mgs_stable(Qop, v)

	d1 = np.ones((10,), dtype = 'd')
	d2 = np.ones((8,), dtype = 'd')

	
	d, w = LowRankGHEP(u,d1,v,d2,Qop,inverse=False)


	print np.linalg.norm(np.dot(w.T,np.linalg.solve(Q,w))-np.eye(w.shape[1]),2)
	w = np.linalg.solve(Q,w)
	print np.linalg.norm(np.dot(u,u.T)+np.dot(v,v.T) - np.dot(w,np.dot(np.diag(d),w.T)),2)
	return	

def testlowrankaddition2(Q):

	Qop = Inverse(Q)
	u = np.random.randn(Q.shape[0],10)
	u, _, _ = mgs_stable(Qop, u)
	v = np.random.randn(Q.shape[0],8)
	v, Bv, _ = mgs_stable(Qop, v)

	d1 = np.ones((10,), dtype = 'd')
	d2 = np.ones((8,), dtype = 'd')

	d, w = LowRankAddition(u,d1,v,d2,Qop, BV = Bv, inverse = True)

	print np.linalg.norm(np.dot(u,u.T)+np.dot(v,v.T) - np.dot(w,np.dot(np.diag(d),w.T)),2)
	print np.linalg.norm(np.dot(w.T,np.linalg.solve(Q,w))-np.eye(w.shape[1]),2)

	return	


if __name__ == '__main__':
	Q = construct_mat()
	testlowrankaddition(Q)
	testlowrankaddition2(Q)

