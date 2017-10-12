from dolfin import *
import numpy as np

from scipy.sparse.linalg import factorized
from ublas2sparse import convertcsr, convertcsc
import scipy
from scipy.sparse import csr_matrix, kron, eye

class EllipticPDE:
	def __init__(self, N):

		#Mesh discretization
		self.N = N

		self.solver = None
		self.solverT = None

		#Sources
		self.sources = None
		
		#Measurements 
		self.H = None	
	
	def Setup(self):
		
		#Call initialization functions
		#Written in such a way that these functions can be overloaded
		self.FunctionSpace()
		self.BilinearForms()
		self.BoundaryConditions()

		return
	
	def FunctionSpace(self):	

		#self.mesh = UnitSquareMesh(N[0],N[1])
		N = self.N
		self.mesh = UnitCircle(N[0])
		self.V = FunctionSpace(self.mesh,"Lagrange",1)

		#Define variational problem
		self.u = TrialFunction(self.V)
		self.v = TestFunction(self.V)

	
		return


	def BilinearForms(self):	
		#Define the variable coefficient 
		self.sk = interpolate(Constant(0.0), self.V)
		self.ds = TrialFunction(self.V)
		self.ns = self.V.dim()
		self.m = self.V.dim()	
		
		#Create the stiffness matrix
		self.a = inner(exp(self.sk)*nabla_grad(self.u),nabla_grad(self.v))*dx
		
		#Create the bilinear form for the derivatives
		self.uk = Function(self.V)
		self.g = inner(exp(self.sk)*self.ds*grad(self.uk),grad(self.v))*dx

		
		self.vk = Function(self.V)
		self.gk = inner(exp(self.sk)*self.ds*grad(self.uk),grad(self.vk))*dx

		return
	def BoundaryConditions(self):
		#define the boundary conditions
		u0 = Expression('0.0')

		#Construct the boundary conditions
		def boundary(x, on_boundary):
			return on_boundary
		
		self.bc = DirichletBC(self.V, u0, boundary)

	def BuildMatrices(self, sk, ss = None):
	
		#Build forward matrices
		self.sk.vector()[:] = sk.ravel()
		
		A  = assemble(self.a)
		self.bc.apply(A)
	
		#ublas to csr
		A = convertcsc(A)
		
		#LU solver
		self.solver = factorized(A)

		AT = (A.T).tocsc()

		self.solvert = factorized(AT)
		return

	def Sources(self, pts, flrates = 1.):
		"""
			A delta source at the given well location
		"""
		ns = self.ns
		self.nr = len(pts) 
		
		self.sources = np.zeros((ns,self.nr), dtype = 'd')
		
		#Sources Parameters 
                R = 1.0;	sigma = 0.01;	    x0 = 0.5; 	y0 = 0.5;
                f = Expression('fl*exp(-0.5*(pow((R*x[0] - x0)/sigma, 2)) '
                                '     - 0.5*(pow((R*x[1] - y0)/sigma, 2)))',
                                fl=flrates, R=R, x0=x0, y0=y0, sigma=sigma)
                        		
		L = f*self.v*dx
		
		b = Vector(ns)
		for i in np.arange(self.nr):
			f.x0 = pts[i,0];	f.y0 = pts[i,1];	
			b = assemble(L);
			self.bc.apply(b)
			self.sources[:,i] = b.array()

		return		

	def BuildMeasurementOperator(self, pts):
		"""
			Observation locations.
		"""
		#In space
		ns = self.ns
		nm = len(pts)

		self.nm = nm
		H = np.zeros((nm,ns), dtype = 'd')

		coord = self.mesh.coordinates()

		b = Vector(ns)
		for i in np.arange(self.nm):
			b.zero()

			temp = coord - pts[i][:]
                        posx = np.argmin(np.add.reduce((temp*temp),axis=1))
			
			b[posx] = 1.0
			H[i,:] = np.copy(b.array())


		nr = self.nr
		self.H = kron(eye(nr,nr),csr_matrix(H))

		self.n = nm*nr	

		return 


	def HMult(self, x):
		return self.H*x


	def ForwardSolve(self, b, transpose = False):
		"""
			Solve A(s) x = b or A(s)'x = b 
			Since it uses symmetric form both *should* yield same results
		"""
		nr = self.nr
		ns = self.ns	
		
		x = np.zeros((ns*nr,), dtype = 'd')
		for i in np.arange(nr):
			x[i*ns:(i+1)*ns] = self.solver(b[:,i]) if not transpose else self.solvert(b[:,i])
	
		return x
	
	def DerivativeMult(self, x, phi, transpose = False):
		"""
			G = d(A(s)u)/ds
			Compute y = G(s) x  or y = G(s)'x  
			phi is the current solution of the forward problem
		"""
		ns = self.ns

		self.uk.vector()[:] = phi

		G = assemble(self.g)
		self.bc.apply(G)		

		G = convertcsr(G)
	
		if not transpose:	
			return G*x		
		else:	
			return G.T*x

		return

	def JacobianMult(self, x, phi, transpose = False):
		"""
			x 	= vector to be multiplied
			phi	= current solution to the forward problem

			Call BuildMatrices before calling this function
		"""

		nm = self.nm
		nr = self.nr
		ns = self.ns


		if not transpose:	#J*x
			Jx = np.zeros((nm*nr,), dtype = 'd')	# nm*nr = n	
	
			Gx = np.zeros((ns,nr), dtype = 'd')
			for i in np.arange(nr):		#Loop over number of sources
				Gx[:,i] = self.DerivativeMult(x, phi[i*ns:(i+1)*ns], transpose = False)
				
			AinvGx	 = self.ForwardSolve(Gx)
			
			Jx = -self.H*AinvGx.ravel()		
		
			return Jx
		else:
			Jtx = np.zeros((ns,), dtype = 'd')	
			Htx = -self.H.T*x
				
			AtinvHtx = self.ForwardSolve(np.reshape(Htx, (ns,nr), order = 'F'), transpose = True)
				
			for i in np.arange(nr):
				GtAtinvHtx = self.DerivativeMult(AtinvHtx[i*ns:(i+1)*ns], phi[i*ns:(i+1)*ns],  transpose = True)
				Jtx += GtAtinvHtx					

			
			return Jtx

	def ConstructJacobian(self, sc, phi):

		nm = self.nm
		nr = self.nr
		ns = self.ns

		self.BuildMatrices(sc)
		v = Vector(ns)
	
		ei = np.zeros((ns,), dtype = 'd')
		J = np.zeros((nm*nr, ns), dtype = 'd')
		for i in np.arange(nr):
			self.uk.vector()[:] = phi[i*ns:(i+1)*ns]
			
			for j in np.arange(nm):
				ei =  np.copy(self.H.getrow(j).todense().T)
				ei *= -1.0
				psi = self.solvert( np.squeeze(ei))
				self.vk.vector()[:] = psi
				
				assemble(self.gk, v)
				J[i*nm+j,:] = v.array()
				
		return J
		

def CreateSources(pde, n):
	x = np.linspace(0.35, 0.65, n)
	X, Y = np.meshgrid(x, x)
	
	pts = np.vstack((X.ravel(), Y.ravel())).transpose()
	pde.Sources(pts)
	
	return

def CreateMeasurements(pde, n):
	
	x = np.linspace(0.35, 0.65, n)
	X, Y = np.meshgrid(x, x)
	
	pts = np.vstack((X.ravel(), Y.ravel())).transpose()
	pde.BuildMeasurementOperator(pts)
	

	return

if __name__ == '__main__':

	parameters["linear_algebra_backend"] = "uBLAS"
	
	N = np.array([20,50])
	pde = EllipticPDE(N)

	pde.Setup()

	ns = pde.ns
	sk = np.zeros((ns,), dtype = 'd')

	pde.BuildMatrices(sk, ss = None)
	

	nr = 3
	nm = 3 
	CreateSources(pde, nr)
	CreateMeasurements(pde, nm)

	phi = pde.ForwardSolve(pde.sources)
	from time import time	
	start = time()
	J = pde.ConstructJacobian(sk, phi)
	print "Time Taken to build JAcobian is %g", time() - start

	x = np.random.randn(ns)
	Jx = np.dot(J,x)
	Jxmult = pde.JacobianMult(x, phi)
	print np.linalg.norm(Jx-Jxmult)/np.linalg.norm(Jx)


	x = np.random.randn((nr**2)*(nm**2))
	Jtx = np.dot(J.T,x)

	start = time()
	Jtxmult = pde.JacobianMult(x, phi, transpose = True)
	print "Time Taken to build JAcobian is %g", time() - start
	print np.linalg.norm(Jtx-Jtxmult)/np.linalg.norm(Jtx)
	

	sens = False	
	if sens == True:
		import matplotlib.pyplot as plt
		from matplotlib.tri import Triangulation
	
		pts = pde.mesh.coordinates()
		cells = pde.mesh.cells()

		tri = Triangulation(pts[:,0], pts[:,1], triangles = cells)

		plt.figure(1)
		plt.tripcolor(tri, pde.sources[:,1], shading = 'gouraud', cmap = plt.cm.rainbow)
		plt.title('Sensitivity')	
