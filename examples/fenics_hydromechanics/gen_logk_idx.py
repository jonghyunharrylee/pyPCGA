import matplotlib.pyplot as plt
import numpy as np

'''
generate lnk_idx to link s_true to s_true_fem. this is needed in poro.py using uniform grid lnk
'''
s_true_fem = np.loadtxt('het_9999.csv', delimiter=',')
s_true2d = np.loadtxt('het_structure.csv', delimiter=',')
pts_fem = np.loadtxt('dof_perm_dg0.csv', delimiter=',')

s_true = np.loadtxt('s_true.txt')

x = np.linspace(0, 1., 128)
y = np.linspace(0, 1., 128)
grid_x, grid_y = np.meshgrid(x, y)

idx = []
i = 0

for pts in pts_fem:
    ptx = pts[0]
    pty = pts[1]
    x_idx = np.abs(x - ptx).argmin()
    y_idx = np.abs(y - pty).argmin()
    idx.append(x_idx*128 + y_idx)

    i = i + 1

np.savetxt('lnk_idx.txt',np.array(idx),fmt='%d')