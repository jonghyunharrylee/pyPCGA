from ufl import div, dx, ds, FiniteElement, grad, inner, VectorElement, pi, sym, jump, avg
from dolfin import (Constant, Expression, Mesh, MeshFunction, UnitCubeMesh, SubDomain,
                    near, Identity, tr, det, ln, dot, derivative, VectorFunctionSpace, assemble, File,
                    Measure, BoxMesh, Point, FunctionSpace, TensorFunctionSpace, FacetNormal, Function,
                    CellVolume, FacetArea, XDMFFile)
from multiphenics import (block_derivative, BlockDirichletBC, BlockElement, BlockFunction, BlockFunctionSpace,
                          BlockNonlinearProblem, BlockPETScSNESSolver, block_split, BlockTestFunction,
                          BlockTrialFunction, block_assemble, block_solve, block_assign, DirichletBC)
# from rbnics.backends import copy, TimeSeries
# from rbnics.backends.abstract import TimeSeries as AbstractTimeSeries
# from rbnics.backends.dolfin.wrapping.function_save import SolutionFileXDMF as SolutionFile
# import rbnics.backends.multiphenics.wrapping.block_function_space  # noqa: F401
# from rbnics.utils.io import TextIO

from multiprocessing import Pool

import numpy as np
import csv
import os
import math

from itertools import count

os.environ['OMP_NUM_THREADS'] = '1'

components = ["u", "p"]

def mu_str(mu):
    return "_".join("{:.3f}".format(mu_) for mu_ in mu)


# def solution_files(mu, solution_folder):
#     return {c: SolutionFile(solution_folder, "solution_" + c + "__" + mu_str(mu)) for c in components}


# def write_solution(mu, solution_folder, from_):
#     assert isinstance(from_, AbstractTimeSeries)
#     TextIO.save_file(from_._time_interval, solution_folder, "time_interval__" + mu_str(mu) + ".txt")
#     TextIO.save_file(from_._time_step_size, solution_folder, "time_step_size__" + mu_str(mu) + ".txt")
#     files = solution_files(mu, solution_folder)
#     for (t_index, solution_t) in enumerate(from_):
#         for (c, f) in files.items():
#             f.write(solution_t.sub(c), c, t_index)


# def read_solution(mu, solution_folder, to):
#     assert isinstance(to, BlockFunction)
#     time_interval = TextIO.load_file(solution_folder, "time_interval__" + mu_str(mu) + ".txt")
#     time_step_size = TextIO.load_file(solution_folder, "time_step_size__" + mu_str(mu) + ".txt")
#     time_series = TimeSeries(time_interval, time_step_size)
#     files = solution_files(mu, solution_folder)
#     for (t_index, _) in enumerate(time_series.expected_times()):
#         for (c, f) in files.items():
#             files[c].read(to.sub(c), c, t_index)
#         to.apply("from subfunctions")
#         time_series.append(copy(to))
#     return time_series


def read_mesh():
    mesh = Mesh("data/biot_2mat.xml")
    subdomains = MeshFunction("size_t", mesh, "data/biot_2mat_physical_region.xml")
    boundaries = MeshFunction("size_t", mesh, "data/biot_2mat_facet_region.xml")
    return (mesh, subdomains, boundaries)


def generate_block_function_space(mesh):
    # Block function space
    V_element = VectorElement("CG", mesh.ufl_cell(), 2)
    Q_element = FiniteElement("DG", mesh.ufl_cell(), 1)
    W_element = BlockElement(V_element, Q_element)
    #return BlockFunctionSpace(mesh, W_element, components=components)
    return BlockFunctionSpace(mesh, W_element)

def E_nu_to_mu_lmbda(E, nu):
    mu = E/(2*(1.0+nu))
    lmbda = (nu*E)/((1-2*nu)*(1+nu))
    return (mu, lmbda)

def K_nu_to_E(K, nu):
    return 3*K*(1-2*nu)

def Ks_cal(alpha,K):
    if alpha == 1.0:
        Ks = 1e35
    else:
        Ks = K/(1.0-alpha)
    return Ks

def init_scalar_parameter(p,p_value,index,sub):
    for cell_no in range(len(sub.array())):
        subdomain_no = sub.array()[cell_no]
        if subdomain_no == index:
            p.vector()[cell_no] = p_value
    return p

def init_tensor_parameter(p,p_value,index,sub,dim):
    for cell_no in range(len(sub.array())):
        subdomain_no = sub.array()[cell_no]
        if subdomain_no == index:
            k_j = 0
            for k_i in range(cell_no*np.power(dim,2), \
                cell_no*np.power(dim,2)+np.power(dim,2)):
                p.vector()[k_i] = p_value[k_j]
                k_j = k_j + 1
    return p

def strain(u):
    return sym(grad(u))

def avg_w(x,w):
    return (w*x('+')+(1-w)*x('-'))

def k_normal(k,n):
    return dot(dot(np.transpose(n),k),n)

def k_plus(k,n):
    return dot(dot(n('+'),k('+')),n('+'))

def k_minus(k,n):
    return dot(dot(n('-'),k('-')),n('-'))

def weight_e(k,n):
    return (k_minus(k,n))/(k_plus(k,n)+k_minus(k,n))

def k_e(k,n):
    return (2*k_plus(k,n)*k_minus(k,n)/(k_plus(k,n)+k_minus(k,n)))

def k_har(k):
    return (2*k*k/(k+k))

def weight_k_homo(k):
    return (k)/(k+k)

def init_from_file_parameter(p,index,sub,filename):
    with open(filename) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',',quoting=csv.QUOTE_NONNUMERIC)
        i = 0
        for row in readCSV:
            p.vector()[i] = row[0]
            i +=1
    return p

def init_from_file_parameter_scalar_to_tensor(p,index,sub,filename):
    with open(filename) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',',quoting=csv.QUOTE_NONNUMERIC)
        i = 0
        idx = 0
        for row in readCSV:
            p.vector()[i] = math.pow(10, row[0])
            i +=1
            p.vector()[i] = 0.
            i +=1
            p.vector()[i] = 0.
            i +=1
            p.vector()[i] = math.pow(10, row[0])
            i +=1
    return p

def init_from_var_scalar_to_tensor(p,index,sub,s):
    i = 0
    idx = 0
    for row in s:
        p.vector()[i] = math.pow(10, row[0])
        i +=1
        p.vector()[i] = 0.
        i +=1
        p.vector()[i] = 0.
        i +=1
        p.vector()[i] = math.pow(10, row[0])
        i +=1
    return p

class Model:
    #_ids = count(0)

    def __init__(self, params=None):
        #self.id = next(self._ids)
        self.idx = 0
        
        self.pts_fem = None
        self.pts = None
        self.pty = None
        self.logk_idx = None

        from psutil import cpu_count  # physcial cpu counts
        self.ncores = cpu_count(logical=False)
        if params is not None:
            if 'ncores' in params:
                self.ncores = params['ncores']
            
            if 'ptx' in params:
                self.ptx = params['ptx']
            else:
                ValueError('please provide forward_params[ptx]')
            
            if 'pty' in params:
                self.pty = params['pty']
            else:
                ValueError('please provide forward_params[pty]')
            
            if 'pts_fem' in params:
                self.pts_fem = params['pts_fem']
            else:
                ValueError('please provide forward_params[pts_fem]')
            
            if 'logk_idx' in params:
                self.logk_idx = params['logk_idx']
            else:
                ValueError('please provide forward_params[logk_idx]')

    def s_to_s_fem(self,s):
        # simple nearest neighbor interpolation
        return s[self.logk_idx]

    def model_run(self, mu, idx=0):
        # print("Performing truth solve at mu =", mu)

        # print(mu.shape)
        ##print('ID: ', self.id, ' mean: ', np.mean(mu))
        #print('ID: ', idx, ' mean: ', np.mean(mu))

        # from structured grids to FEM grids
        mu = self.s_to_s_fem(mu)

        time_interval = (0, 1200)
        time_step_size = 400

        (mesh, subdomains, boundaries) = read_mesh()
        W = generate_block_function_space(mesh)

        PM = FunctionSpace(mesh, 'DG', 0)
        TM = TensorFunctionSpace(mesh, 'DG', 0)

        I = Identity(mesh.topology().dim())

        dx = Measure('dx', domain=mesh, subdomain_data=subdomains)
        ds = Measure('ds', domain=mesh, subdomain_data=boundaries)
        dS = Measure('dS', domain=mesh, subdomain_data=boundaries)

        # Test and trial functions
        vq = BlockTestFunction(W)
        (v, q) = block_split(vq)
        up = BlockTrialFunction(W)
        (u, p) = block_split(up)

        w = BlockFunction(W)
        w0 = BlockFunction(W)
        (u0, p0) = block_split(w0)

        n = FacetNormal(mesh)
        vc = CellVolume(mesh)
        fc = FacetArea(mesh)

        h = vc/fc
        h_avg = (vc('+') + vc('-'))/(2*avg(fc))

        penalty1 = 1.0
        penalty2 = 10.0
        theta = 1.0

        # Constitutive parameters
        K = 1000.e3
        nu = 0.25
        E = K_nu_to_E(K, nu) # Pa 14

        (mu_l, lmbda_l) = E_nu_to_mu_lmbda(E, nu)

        f_stress_y = Constant(-1.e3)

        f = Constant((0.0, 0.0)) #sink/source for displacement
        g = Constant(0.0) #sink/source for velocity

        p_D1 = 0.0
        p_D2 = 1000.0

        alpha = 1.0

        rho1 = 1000.0
        vis1 = 1.e-3
        cf1 = 1.e-10
        phi1 = 0.2


        cf1 = 1e-10
        ct1 = phi1*cf1


        rho2 = 1000.0
        vis2 = 1.e-3
        cf2 = 1.e-10
        phi2 = 0.2

        cf2 = 1e-10
        ct2 = phi2*cf2

        rho_values = [rho1, rho2]
        vis_values = [vis1, vis2]
        cf_values = [cf1, cf2]
        phi_values = [phi1, phi2]
        ct_values = [ct1, ct2]

        rho = Function(PM)
        vis = Function(PM)
        phi = Function(PM)
        ct = Function(PM)
        k = Function(TM)

        rho = init_scalar_parameter(rho,rho_values[0],500,subdomains)
        vis = init_scalar_parameter(vis,vis_values[0],500,subdomains)
        phi = init_scalar_parameter(phi,phi_values[0],500,subdomains)
        ct = init_scalar_parameter(ct,ct_values[0],500,subdomains)

        rho = init_scalar_parameter(rho,rho_values[1],501,subdomains)
        vis = init_scalar_parameter(vis,vis_values[1],501,subdomains)
        phi = init_scalar_parameter(phi,phi_values[1],501,subdomains)
        ct = init_scalar_parameter(ct,ct_values[1],501,subdomains)

        k = init_from_var_scalar_to_tensor(k,0.,0.,mu)
        #xdmk = XDMFFile(mesh.mpi_comm(), "results/permeability_" + str(self.id) + ".xdmf")
        #xdmk = XDMFFile(mesh.mpi_comm(), "results/permeability_" + str(idx) + ".xdmf")
        #k.rename("perm", "permeability_eg")
        #xdmk.write(k, 0.)

        #T = time_interval
        T = 1200.
        t = 0.0
        dt = time_step_size


        #DirichletBC - equilibrate
        bcd1 = DirichletBC(W.sub(0).sub(0), Constant(0.0), boundaries, 1) # No normal displacement for solid on left side
        bcd3 = DirichletBC(W.sub(0).sub(0), Constant(0.0), boundaries, 3) # No normal displacement for solid on right side
        bcd4 = DirichletBC(W.sub(0).sub(1), Constant(0.0), boundaries, 4) # No normal displacement for solid on bottom side
        bcs = BlockDirichletBC([[bcd1, bcd3, bcd4], []])

        a = inner(2*mu_l*strain(u)+lmbda_l*div(u)*I, sym(grad(v)))*dx

        b = inner(-alpha*p*I,sym(grad(v)))*dx

        c = rho*alpha*div(u)*q*dx

        d = phi*rho*ct*p*q*dx + dt*dot(rho*k/vis*grad(p),grad(q))*dx \
            - dt*dot(avg_w(rho*k/vis*grad(p),weight_e(k,n)), jump(q, n))*dS \
            - theta*dt*dot(avg_w(rho*k/vis*grad(q),weight_e(k,n)), jump(p, n))*dS \
            + dt*penalty1/h_avg*avg(rho)*k_e(k,n)/avg(vis)*dot(jump(p, n), jump(q, n))*dS \
            - dt*dot(rho*k/vis*grad(p),q*n)*ds(2) \
            - dt*dot(rho*k/vis*grad(q),p*n)*ds(2) \
            + dt*(penalty2/h*rho/vis*dot(dot(n,k),n)*dot(p*n,q*n))*ds(2) \
            - dt*dot(rho*k/vis*grad(p),q*n)*ds(4) \
            - dt*dot(rho*k/vis*grad(q),p*n)*ds(4) \
            + dt*(penalty2/h*rho/vis*dot(dot(n,k),n)*dot(p*n,q*n))*ds(4)


        lhs = [[a, b],
               [c, d]]

        f_u = inner(f,v)*dx\
            + dot(f_stress_y*n,v)*ds(2)

        f_p = rho*alpha*div(u0)*q*dx \
            + phi*rho*ct*p0*q*dx + dt*g*q*dx \
            - dt*dot(p_D1*n,rho*k/vis*grad(q))*ds(2) \
            + dt*(penalty2/h*rho/vis*dot(dot(n,k),n)*dot(p_D1*n,q*n))*ds(2) \
            - dt*dot(p_D2*n,rho*k/vis*grad(q))*ds(4) \
            + dt*(penalty2/h*rho/vis*dot(dot(n,k),n)*dot(p_D2*n,q*n))*ds(4)

        rhs = [f_u, f_p]

        # Perform a fake loop over time.
        # up_over_time = TimeSeries(time_interval, time_step_size)
        # for _ in up_over_time.expected_times():
        while t < T:
            t += dt
            #print('solving time of: ', t)
            AA = block_assemble(lhs)
            FF = block_assemble(rhs)
            bcs.apply(AA)
            bcs.apply(FF)
            block_solve(AA, w.block_vector(), FF, "mumps")
            block_assign(w0, w)
            # solver.solve()
            #up_over_time.append(copy(w))

        obs_locmat = np.loadtxt('pressure_idx.csv')
        obs_locmat = np.ma.make_mask(obs_locmat)
        # up_over_time_rev = up_over_time[-1][1].vector()[obs_locmat]
        up_over_time_rev = w[1].vector()[obs_locmat]
        
        #np.savetxt('pressure_out.csv',up_over_time_rev)
        
        return up_over_time_rev.reshape(-1)

#        up_over_time = self.truth_solve(s)
#        return up_over_time


    def run(self, s, par, ncores=None):
        if ncores is None:
            ncores = self.ncores

        method_args = range(s.shape[1])
        args_map = [(s[:, arg:arg + 1], arg) for arg in method_args]

        if par:
            pool = Pool(processes=ncores)
            simul_obs = pool.map(self, args_map)
        else:
            simul_obs = []
            for item in args_map:
                simul_obs.append(self(item))

        return np.array(simul_obs).T # make it 2D

    def __call__(self, args):
        return self.model_run(args[0], args[1])

if __name__ == '__main__':
    import numpy as np
    from time import time
    import poro

    # load true value for comparison purpose
    #s_true_fem = np.loadtxt('het_9999.csv', delimiter=',')
    #s_true = np.loadtxt('het_structure.csv', delimiter=',')

    obs_true_all = np.loadtxt('pressure.csv', delimiter=',')
    obs_locmat = np.loadtxt('pressure_idx.csv', delimiter=',')
    obs_locmat = np.ma.make_mask(obs_locmat)
    obs_true = obs_true_all[obs_locmat]
    
    s_true = np.loadtxt('s_true.txt').reshape(-1,1)
    m = s_true.shape[0]
    mx = 128
    my = 128
    assert(m == mx*my)

    pts_fem = np.loadtxt('dof_perm_dg0.csv', delimiter=',')
    ptx = np.linspace(0,1,mx)
    pty = np.linspace(0,1,my)

    logk_idx = np.loadtxt('logk_idx.txt').astype(int)

    forward_params = {'ptx': ptx, 'pty': pty, 'pts_fem': pts_fem, 'logk_idx': logk_idx}

    model = poro.Model(forward_params)
    
    print('(1) single run')

    from time import time
    stime = time()

    par = False # no parallelization 
    
    simul_obs = model.run(s_true,par)
    print('simulation run: %f sec' % (time() - stime))
    
    # start with 5% error (of max(simul_obs_true) = 1000)
    obs = simul_obs + 50.0*np.random.randn(simul_obs.shape[0],simul_obs.shape[1])
    
    np.savetxt('obs.txt',obs)
#    np.savetxt('obs_true.txt',simul_obs)

#    import sys
#    sys.exit(0)

    # ncores = 2
    # nrelzs = 2
    
    # print('(2) parallel run with ncores = %d' % ncores)
    # par = True # parallelization false
    # srelz = np.zeros((np.size(s_true,0),nrelzs),'d')
    # for i in range(nrelzs):
    #     srelz[:,i:i+1] = s_true + 0.1*np.random.randn(np.size(s_true,0),1)
    
    # simul_obs_all = model.run(srelz,par, ncores = ncores)
    
    # print(simul_obs_all)

    # use all the physcal cores if not specify ncores
    #print('(3) parallel run with all the physical cores')
    #simul_obs_all = mymodel.run(srelz,par)
    #print(simul_obs_all)