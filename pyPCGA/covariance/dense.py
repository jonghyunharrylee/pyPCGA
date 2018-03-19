from __future__ import print_function
import numpy as np

__all__ = ['GenerateDenseMatrix']

#Compute O(N^2) interactions    
def GenerateDenseMatrix(pts, kernel):
    nx = np.size(pts,0)
    ptsx = pts

    if nx == 1:
        ptsx = pts[np.newaxis,:]

    dim = np.size(pts,1)
    R = np.zeros((nx,nx),'d')

    for i in np.arange(dim):
        X, Y = np.meshgrid(ptsx[:,i],ptsx[:,i], indexing='ij')
        R += (X.transpose()-Y.transpose())**2. 

    return kernel(np.sqrt(R))         
