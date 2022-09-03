'''
    toeplitz matrix-vector mutliplication adapted from Arvind Saibaba's code
'''
import numpy as np
#from IPython.core.debugger import Tracer; debug_here = Tracer()

__all__ = ['CreateRow', 'ToeplitzProduct', 'Realizations']

def DistanceVector(x, y, theta):
    dim = x.shape[1]
    DM = np.zeros(x.shape[0])

    if dim == 1:
        DM = (x[:] - y) ** 2. / theta ** 2.
    else:
        for i in np.arange(dim):
            DM += (x[:, i] - y[i]) ** 2. / theta[i] ** 2.

    DM = np.sqrt(DM)
    return DM


def CreateRow(xmin, xmax, N, kernel, theta):
    """
    Create row column of covariance matrix
    """
    dim = N.size

    if dim == 1:
        x = np.linspace(xmin[0], xmax[0], N[0])
        x = x.reshape(-1,1) # make it 2D for consistency
        R = DistanceVector(x, x[0], theta)
    elif dim == 2:
        x1 = np.linspace(xmin[0], xmax[0], N[0])
        x2 = np.linspace(xmin[1], xmax[1], N[1])

        xx, yy = np.meshgrid(x1, x2, indexing='ij')

        x = np.vstack((np.ravel(xx, order='F'), np.ravel(yy, order='F'))).transpose()
        R = DistanceVector(x, x[0, :].transpose(), theta)

    elif dim == 3:
        x1 = np.linspace(xmin[0], xmax[0], N[0])
        x2 = np.linspace(xmin[1], xmax[1], N[1])
        x3 = np.linspace(xmin[2], xmax[2], N[2])

        #xx, yy, zz = np.meshgrid(x1, x2, x3, indexing='ij')
        xx, yy, zz = np.meshgrid(x1, x2, x3, indexing='ij')

        x = np.vstack((np.ravel(xx, order='F'), np.ravel(yy, order='F'), np.ravel(zz, order='F'))).transpose()
        R = DistanceVector(x, x[0, :].transpose(), theta)

    else:
        raise ValueError("Support 1,2 and 3 dimensions")

    row = kernel(R)

    return row, x


def ToeplitzProduct(x, row, N):
    ''' Toeplitz matrix times x

    :param x: x for Qx
    :param row: from CreateRow
    :param N: size in each dimension ex) N = [2,3,4]
    :return: Qx
    '''
    dim = N.size

    if dim == 1:
        circ = np.concatenate((row, row[-2:0:-1])).reshape(-1)
        padded = np.concatenate((x, np.zeros(N[0] - 2)))
        result = np.fft.ifft(np.fft.fft(circ) * np.fft.fft(padded))
        result = np.real(result[0:N[0]])

    elif dim == 2:
        circ = np.reshape(row, (N[0], N[1]), order='F')
        circ = np.concatenate((circ, circ[:, -2:0:-1]), axis=1)
        circ = np.concatenate((circ, circ[-2:0:-1, :]), axis=0)

        n = np.shape(circ)
        padded = np.reshape(x, (N[0], N[1]), order='F')

        result = np.fft.ifft2(np.fft.fft2(circ) * np.fft.fft2(padded, n))
        result = np.real(result[0:N[0], 0:N[1]])
        result = np.reshape(result, -1, order='F')

    elif dim == 3:
        circ = np.reshape(row, (N[0], N[1], N[2]), order='F')
        circ = np.concatenate((circ, circ[:, :, -2:0:-1]), axis=2)
        circ = np.concatenate((circ, circ[:, -2:0:-1, :]), axis=1)
        circ = np.concatenate((circ, circ[-2:0:-1, :, :]), axis=0)

        n = np.shape(circ)
        padded = np.reshape(x, N, order='F')

        result = np.fft.ifftn(np.fft.fftn(circ) * np.fft.fftn(padded, n))
        result = np.real(result[0:N[0], 0:N[1], 0:N[2]])
        result = np.reshape(result, -1, order='F')
    else:
        raise ValueError("Support 1,2 and 3 dimensions")

    return result


def Realizations(row, N):
    dim = N.size
    if dim == 1:
        circ = np.concatenate((row, row[-2:0:-1]))
        n = circ.shape

        eps = np.random.normal(0, 1, n) + 1j * np.random.normal(0, 1, n)
        res = np.fft.ifft(np.sqrt(np.fft.fft(circ)) * eps) * np.sqrt(n)

        r1 = np.real(res[0:N[0]])
        r2 = np.imag(res[0:N[0]])

    elif dim == 2:
        circ = np.reshape(row, (N[0], N[1]), order='F')
        circ = np.concatenate((circ, circ[:, -2:0:-1]), axis=1)
        circ = np.concatenate((circ, circ[-2:0:-1, :]), axis=0)

        n = np.shape(circ)
        eps = np.random.normal(0, 1, n) + 1j * np.random.normal(0, 1, n)

        res = np.fft.ifft2(np.sqrt(np.fft.fft2(circ)) * eps) * np.sqrt(n[0] * n[1])
        res = res[0:N[0], 0:N[1]]
        res = np.reshape(res, -1, order='F')

        r1 = np.real(res)
        r2 = np.imag(res)

    elif dim == 3:
        circ = np.reshape(row, (N[0], N[1], N[2]), order='F')
        circ = np.concatenate((circ, circ[:, :, -2:0:-1]), axis=2)
        circ = np.concatenate((circ, circ[:, -2:0:-1, :]), axis=1)
        circ = np.concatenate((circ, circ[-2:0:-1, :, :]), axis=0)

        n = np.shape(circ)
        eps = np.random.normal(0, 1, n) + 1j * np.random.normal(0, 1, n)

        res = np.fft.ifftn(np.sqrt(np.fft.fftn(circ)) * eps) * np.sqrt(n[0] * n[1] * n[2])
        res = res[0:N[0], 0:N[1], 0:N[2]]
        res = np.reshape(res, -1, order='F')

        r1 = np.real(res)
        r2 = np.imag(res)
    else:
        raise ValueError("Support 1,2 and 3 dimensions")

    return r1, r2, eps

if __name__ == '__main__':

    import numpy as np


    def kernel(R):
        return 0.01 * np.exp(-R)

    #dim = 1
    #N = np.array([5])
    #dim = 2
    #N = np.array([2, 3])
    dim = 3
    N = np.array([5, 6, 7])

    row, pts = CreateRow(np.zeros(dim), np.ones(dim), N, kernel, np.ones((dim), dtype='d'))
    n = pts.shape
    #for i in np.arange(n[0]):
    #    print(pts[i, 0], pts[i, 1])
    if dim == 1:
        v = np.random.rand(N[0])
    elif dim == 2:
        v = np.random.rand(N[0]*N[1])
    elif dim == 3:
        v = np.random.rand(N[0]*N[1]*N[2])

    res = ToeplitzProduct(v, row, N)

    r1, r2, ep = Realizations(row, N)
    # import scipy.io as sio
    # sio.savemat('Q.mat',{'row':row,'pts':pts,'N':N,'r1':r1,'r2':r2,'ep':ep,'v':v,'res':res})

    from .dense import GenerateDenseMatrix

    mat = GenerateDenseMatrix(pts, kernel)
    res1 = np.dot(mat, v)

    print("rel. error %g for cov. mat. row (CreateRow)" % (np.linalg.norm(mat[0,:] - row) / np.linalg.norm(mat[0,:])))
    print("rel. error %g" % (np.linalg.norm(res - res1) / np.linalg.norm(res1)))
    #print(mat[0,:])
    #print(row)
    #print(res1)
    #print(np.linalg.norm(res1))
