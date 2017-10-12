from numpy import *
from scitools.numpyutils import ndgrid
import scipy.io as sio

__all__ =['CreateRow','ToeplitzProduct','Realizations']

# Haven't really tested dim = 3
def DistanceVector(x,y,theta):
	dim = x.ndim 

	DM = zeros(x.shape[0])

	if dim == 1:
		DM = (x[:] - y)**2./theta**2.

	else:	
		for i in arange(dim):
			DM += (x[:,i]-y[i])**2./theta[i]**2.
		
	
	DM = sqrt(DM)
	return DM


def CreateRow(xmin,xmax,N,kernel,theta):
	dim = N.size
	
	if dim == 1:
		x = linspace(xmin[0],xmax[0],N[0])
		R = DistanceVector(x,x[0],theta)

	elif dim == 2:
		x1 = linspace(xmin[0],xmax[0],N[0])
		x2 = linspace(xmin[1],xmax[1],N[1])

		xx, yy = ndgrid(x1,x2)
	
		x = vstack((ravel(xx, order = 'F'),ravel(yy, order = 'F'))).transpose()
		R = DistanceVector(x,x[0,:].transpose(),theta)

	elif dim == 3:
		x1 = linspace(xmin[0],xmax[0],N[0])
		x2 = linspace(xmin[1],xmax[1],N[1])
		x3 = linspace(xmin[2],xmax[2],N[2])

		xx, yy, zz = ndgrid(x1,x2,x3)
	
		x = vstack((ravel(xx, order = 'F'),ravel(yy, order = 'F'),ravel(zz, order = 'F'))).transpose()
		R = DistanceVector(x,x[0,:].transpose(),theta)

	else:
		print 'Wrong dimension'

	row = kernel(R)
	return row, x

def ToeplitzProduct(x,row,N):

	dim = N.size 
	
	if dim == 1:
		circ = concatenate((row,row[-2:0:-1]))
		padded = concatenate((x,zeros(n-2))) 

		result = fft.ifft(fft.fft(circ)*fft.fft(padded))
		result = real(result[0:N[0]])
		
	elif dim == 2:
		circ = reshape(row,(N[0],N[1]),order = 'F')
		circ = concatenate((circ,circ[:,-2:0:-1]),axis=1)
		circ = concatenate((circ,circ[-2:0:-1,:]),axis=0)
		
		n = shape(circ)
		padded = reshape(x,(N[0],N[1]),order = 'F')		

		result = fft.ifft2(fft.fft2(circ)*fft.fft2(padded,n))
		result = real(result[0:N[0],0:N[1]])
		result = reshape(result,-1,order = 'F')

	elif dim ==3:
		circ = reshape(row,(N[0],N[1],N[2]),order = 'F')
		circ = concatenate((circ,circ[:,:,-2:0:-1]),axis=2)
		circ = concatenate((circ,circ[:,-2:0:-1,:]),axis=1)
		circ = concatenate((circ,circ[-2:0:-1,:,:]),axis=0)
		
		n = shape(circ)
		padded = reshape(x,N,order = 'F')		

		result = fft.ifftn(fft.fftn(circ)*fft.fftn(padded,n))
		result = real(result[0:N[0],0:N[1],0:N[2]])
		result = reshape(result,-1, order = 'F')


	else: 
		print 'Wrong dimension'
	
	return result


def Realizations(row,N):
	dim = N.size
	if dim == 1:
		circ = concatenate((row,row[-2:0:-1]))
		n = circ.shape

		eps = random.normal(0,1,n) + 1j*random.normal(0,1,n)
		res = fft.ifft(sqrt(fft.fft(circ))*eps)*sqrt(n)

		r1 = real(res[0:N[0]]);	r2 = imag(res[0:N[0]]);
		
	elif dim == 2:
		circ = reshape(row,(N[0],N[1]),order = 'F')
		circ = concatenate((circ,circ[:,-2:0:-1]),axis=1)
		circ = concatenate((circ,circ[-2:0:-1,:]),axis=0)
		
		n = shape(circ)
		eps = random.normal(0,1,n) + 1j*random.normal(0,1,n)		

		res = fft.ifft2(sqrt(fft.fft2(circ))*eps)*sqrt(n[0]*n[1])
		res = res[0:N[0],0:N[1]]
		res = reshape(res,-1,order = 'F')
		
		r1 = real(res); 	r2 = imag(res);	

	elif dim == 3:
		circ = reshape(row,(N[0],N[1],N[2]),order = 'F')
		circ = concatenate((circ,circ[:,:,-2:0:-1]),axis=2)
		circ = concatenate((circ,circ[:,-2:0:-1,:]),axis=1)
		circ = concatenate((circ,circ[-2:0:-1,:,:]),axis=0)

		
		n = shape(circ)
		eps = random.normal(0,1,n) + 1j*random.normal(0,1,n)		

		res = fft.ifftn(sqrt(fft.fftn(circ))*eps)*sqrt(n[0]*n[1]*n[2])
		res = res[0:N[0],0:N[1],0:N[2]]
		res = reshape(res,-1,order = 'F')
		
		r1 = real(res); 	r2 = imag(res);	

	return r1, r2, eps



if __name__ == '__main__':

	def kernel(R):
		return exp(-R)


	
	dim = 3
	N = array([5, 5, 5])

	row, pts = CreateRow(zeros(dim),ones(dim),N,kernel,
			ones((dim),dtype='d'))	

	#n = pts.shape
	#for i in arange(n[0]):
	#	print pts[i,0], pts[i,1]
		
	if dim == 2:
		v = random.rand(N[0],N[1])
	elif dim == 3 :
		v = random.rand(N[0],N[1],N[2])

	res = ToeplitzProduct(v,row,N)	
	r1, r2, ep = Realizations(row,N)

	sio.savemat('Q.mat',{'row':row,'pts':pts,'N':N,'r1':r1,'r2':r2,'ep':ep,'v':v,'res':res})

