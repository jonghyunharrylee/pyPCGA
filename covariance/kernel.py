import numpy as np
from exceptions import NotImplementedError
from scipy.special import gamma
from scipy.misc import factorial

__all__ = ['Matern']

def _maternpolykernel(R, p, l):
	nu = p + 0.5	
	r = np.sqrt(2.*nu)*R/l
	poly = np.zeros_like(r)
	factor = gamma(p+1)/gamma(2*p+1)
	for i in np.arange(p+1):
		poly +=  factorial(p+i)*np.power(2.*r,p-i)/(factorial(i)*factorial(p-i))

	return factor*np.exp(-r)*poly
	
def _gaussian(R,l):	
	r = R/l
	return np.exp(-r*r/2.)
	

class Matern:
	"""	
	Class for Mat\'{e}rn covariance kernel for nu = p + 1/2 and p is an integer.

	Parameters:
	-----------
		
	p:	{int, inf}, optional
		Integer controlling the order of the Mat\'{e}rn covariance kernel	
		Default is 0 corresponding to exponential covariance kernel exp(-r/l)


	l:	real, optional
		Controls the length-scale of the Mat\'{e}rn covariance kernel
		Default is 1.

	Methods:
	--------
	__call__ 

	
	Examples:
	---------

	#Plots the first three covariance kernels corresponding to p=0,1,2 against the hard-coded versions

	import matplotlib.pyplot as plt
	l = 2.
	plt.close('all')

	#Verify maternn
	r = np.linspace(0,1,100)

	k1 = Matern(p = 0, l = l)	#exp(-r)
	plt.plot(r, np.exp(-r/l), label = r'True $nu = 1/2$')
	plt.plot(r, k1(r), label = r'Matern $nu = 1/2$')

		
	k2 = Matern(p = 1, l = l)	#(1+sqrt(3)r)*exp(-sqrt(3)*r)
	plt.plot(r, (1.+np.sqrt(3)*r/l)*np.exp(-np.sqrt(3)*r/l), label = r'True $nu = 3/2$')
	plt.plot(r, k2(r), label = r'Matern $nu = 3/2$')
		
	
	k3 = Matern(p = 2, l = l)	#(1+sqrt(5)*r+5r^2/3)*exp(-sqrt(5)r)
	plt.plot(r,  (1.+np.sqrt(5)*r/l + 5.*r**2./(3.*l**2.))*np.exp(-np.sqrt(5)*r/l), label = r'True $nu = 5/2$')
	plt.plot(r, k3(r), label = r'Matern $nu = 5/2$')


	plt.legend()
	plt.show()


	"""

	def __init__(self, p = 0,l = 1.):
		self.p, self.l = p, l
		if isinstance(p,int): 	
			self.kernel = lambda r:  _maternpolykernel(r, self.p, self.l)
		elif p == np.inf:
			self.kernel = _gaussian(r, self.l)
		else:
			raise NotImplementedError

	def __call__(self, r):
		return self.kernel(r)
						
if __name__ == '__main__':


	import matplotlib.pyplot as plt
	l = 2.
	plt.close('all')

	#Verify maternn
	r = np.linspace(0,1,100)

	k1 = Matern(p = 0, l = l)
	plt.plot(r, np.exp(-r/l), label = r'True $\nu = 1/2$')
	plt.plot(r, k1(r), label = r'Matern $\nu = 1/2$')

	
	k2 = Matern(p = 1, l = l)
	plt.plot(r, (1.+np.sqrt(3)*r/l)*np.exp(-np.sqrt(3)*r/l), label = r'True $\nu = 3/2$')
	plt.plot(r, k2(r), label = r'Matern $\nu = 3/2$')
		
	
	k3 = Matern(p = 2, l = l)
	plt.plot(r,  (1.+np.sqrt(5)*r/l + 5.*r**2./(3.*l**2.))*np.exp(-np.sqrt(5)*r/l), label = r'True $\nu = 5/2$')
	plt.plot(r, k3(r), label = r'Matern $\nu = 5/2$')


	plt.legend()
	plt.show()

