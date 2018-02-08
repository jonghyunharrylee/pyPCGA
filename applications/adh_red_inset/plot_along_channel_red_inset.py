import matplotlib
# matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.ticker import ScalarFormatter, FormatStrFormatter
from scipy.io import loadmat

mydir = "./"

onlyfiles = [f for f in os.listdir(mydir) if os.path.isfile(os.path.join(mydir, f)) and f[:4] == "shat"]
myfile = onlyfiles[-1]
print("read %s" % (mydir + myfile))
elev = np.loadtxt(mydir + myfile)
postv = np.loadtxt(mydir + "postv.txt")
poststd = np.sqrt(postv)

triangles = np.loadtxt("triangles.txt")
meshnode = np.loadtxt("meshnode.txt")
velocity_obs_loc = np.loadtxt("./observation_loc_N250_M8_J1_I10.dat")


matplotlib.rcParams.update({'font.size': 20})
matplotlib.rcParams['legend.loc'] = 'best'

def calculate_centerline(regmesh, nx, ny, j0=5, data_array=None):
    # x_regular= regmesh.nodeArray[:,0].reshape(ny,nx).T; y_regular=regmesh.nodeArray[:,1].reshape(ny,nx).T;
    x_regular = regmesh[:, 0].reshape(ny, nx).T
    y_regular = regmesh[:, 1].reshape(ny, nx).T
    if data_array is None:
        # val_regular= regmesh.nodeArray[:,2].reshape(ny,nx).T;
        val_regular = regmesh[:, 2].reshape(ny, nx).T
    else:
        val_regular = data_array.reshape(ny, nx).T
    val_center = val_regular[:, j0].copy()
    guess_length = np.zeros((nx,), 'd')
    for i in range(1, nx):
        guess_length[i] = guess_length[i - 1] + np.sqrt((x_regular[i, j0] - x_regular[i - 1, j0]) ** 2 +
                                                        (y_regular[i, j0] - y_regular[i - 1, j0]) ** 2)
    return val_center, guess_length

'''
def calculate_thalweg(est, nx, idx, coord, data_array=None):
    x_regular = coord[:, 0];
    y_regular = coord[:, 1];
    # val_regular= est[:];

    val_center = est[idx[:] - 1].copy()
    guess_length = np.zeros((nx,), 'd')
    for i in range(1, nx):
        guess_length[i] = guess_length[i - 1] + np.sqrt((x_regular[i] - x_regular[i - 1]) ** 2 +
                                                        (y_regular[i] - y_regular[i - 1]) ** 2)
    return val_center, guess_length
'''

def calculate_transverse_mean(regmesh, nx, ny, data_array=None):
    x_regular = regmesh[:, 0].reshape(ny, nx).T
    y_regular = regmesh[:, 1].reshape(ny, nx).T
    if data_array is None:
        val_regular = regmesh[:, 2].reshape(ny, nx).T
    else:
        val_regular = data_array.reshape(ny, nx).T

    diff_x = x_regular[:, 1:] - x_regular[:, 0:-1]
    diff_y = y_regular[:, 1:] - y_regular[:, 0:-1]
    diff_x *= diff_x
    diff_y *= diff_y
    trans_delta = np.sqrt(diff_x + diff_y)
    trans_length = np.sum(trans_delta, 1)

    val_mean = 0.5 * val_regular[:, 0] * trans_delta[:, 0]
    val_mean += np.sum(
        0.5 * val_regular[:, 1:-1] * trans_delta[:, :-1] + 0.5 * val_regular[:, 1:-1] * trans_delta[:, 1:], 1)
    val_mean += 0.5 * val_regular[:, -1] * trans_delta[:, -1]
    val_mean /= trans_length
    return val_mean, trans_length


def calculate_crossection_mean(regmesh, nx, ny, i0, nx_avg=5, data_array=None):
    x_regular = regmesh[:, 0].reshape(ny, nx).T
    y_regular = regmesh[:, 1].reshape(ny, nx).T
    if data_array is None:
        val_regular = regmesh[:, 2].reshape(ny, nx).T
    else:
        val_regular = data_array.reshape(ny, nx).T
    ist = max(0, i0 - nx_avg)
    ien = min(nx, i0 + nx_avg)
    val_window = val_regular[ist:ien, :]
    diff_x = x_regular[ist + 1:ien, :] - x_regular[ist:ien - 1, :]
    diff_y = y_regular[ist + 1:ien, :] - y_regular[ist:ien - 1, :]
    diff_x *= diff_x
    diff_y *= diff_y
    cross_delta = np.sqrt(diff_x + diff_y)
    cross_length = np.sum(cross_delta, 0)

    val_mean = 0.5 * val_window[0, :] * cross_delta[0, :]
    val_mean += np.sum(0.5 * val_window[1:-1, :] * cross_delta[:-1, :] + 0.5 * val_window[1:-1, :] * cross_delta[1:, :],
                       0)
    val_mean += 0.5 * val_window[-1, :] * cross_delta[-1, :]
    val_mean /= cross_length
    guess_cross = np.zeros((ny,), 'd')
    for j in range(1, ny):
        guess_cross[j] = guess_cross[j - 1] + np.sqrt((x_regular[i0, j] - x_regular[i0, j - 1]) ** 2 +
                                                      (y_regular[i0, j] - y_regular[i0, j - 1]) ** 2)

    return val_mean, cross_length, guess_cross


# np.copyto(meshnode,mesh_true)

mesh_true = np.array(meshnode)
mesh_est = np.array(meshnode)
mesh_est_u = np.array(meshnode)
mesh_est_l = np.array(meshnode)
mesh_est[:, 2] = elev[:]
mesh_est_u[:,2] = elev + 1.96*poststd[:]
mesh_est_l[:,2] = elev - 1.96*poststd[:]

# centerline
nx = 1001
ny = 51

fig, axes = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(20, 16), dpi=100)

z_center, guess_length = calculate_centerline(mesh_true, nx, ny, 25)
z_center, trans_length = calculate_transverse_mean(mesh_true, nx, ny)
z_est, trans_length = calculate_transverse_mean(mesh_est, nx, ny)
z_est_u, trans_length = calculate_transverse_mean(mesh_est_u, nx, ny)
z_est_l, trans_length = calculate_transverse_mean(mesh_est_l, nx, ny)

axes[0,0].plot(guess_length, z_center, 'r.-' ,label='true')
axes[0,0].plot(guess_length, z_est, 'k.-' ,label='estimate')
axes[0,0].plot(guess_length, z_est_u, 'k--' ,label='95% credible interval')
axes[0,0].plot(guess_length, z_est_l, 'k--')

axes[0,0].set_title('(a) transverse avg. with std(error) = 0.05 m/s', fontsize=24, loc='left')
axes[0,0].legend(loc="upper right")
axes[0,0].set_ylabel('$z_b$ [m]')

z_center, guess_length = calculate_centerline(mesh_true, nx, ny, 25)
z_est, guess_length = calculate_centerline(mesh_est, nx, ny, 25)
z_est_u, guess_length = calculate_centerline(mesh_est_u, nx, ny, 25)
z_est_l, guess_length = calculate_centerline(mesh_est_l, nx, ny, 25)

axes[0,1].plot(guess_length, z_center, 'r.-', guess_length, z_est, 'k.-', guess_length, z_est_u, 'k--', guess_length, z_est_l, 'k--')
axes[0,1].set_title('(b) Centerline with std(error) = 0.05 m/s', fontsize=24, loc='left')

z_center, guess_length = calculate_centerline(mesh_true, nx, ny, 20)
z_est, guess_length = calculate_centerline(mesh_est, nx, ny, 20)
z_est_u, guess_length = calculate_centerline(mesh_est_u, nx, ny, 20)
z_est_l, guess_length = calculate_centerline(mesh_est_l, nx, ny, 20)
axes[1,0].plot(guess_length, z_center, 'r.-', guess_length, z_est, 'k.-', guess_length, z_est_u, 'k--', guess_length, z_est_l, 'k--')
#axes[1].set_ylim([-6.0, 6.0])
axes[1,0].set_title('(c) j = 20 with std(error) = 0.05 m/s', fontsize=24, loc='left')
axes[1,0].set_ylabel('$z_b$ [m]')
axes[1,0].set_xlabel('along channel [m]', fontsize=25)

z_center, guess_length = calculate_centerline(mesh_true, nx, ny, 30)
z_est, guess_length = calculate_centerline(mesh_est, nx, ny, 30)
z_est_u, guess_length = calculate_centerline(mesh_est_u, nx, ny, 30)
z_est_l, guess_length = calculate_centerline(mesh_est_l, nx, ny, 30)
axes[1,1].plot(guess_length, z_center, 'r.-', guess_length, z_est, 'k.-', guess_length, z_est_u, 'k--', guess_length, z_est_l, 'k--')
axes[1,1].set_title('(d) j = 30 with std(error) = 0.05 m/s', fontsize=24, loc='left')
axes[1,1].set_xlabel('along channel [m]', fontsize=25)
#plt.tight_layout()

plt.savefig('red_inset_alongchannel.png', bbox_inches='tight', pad_inches=0, dpi=300)
#plt.show()
plt.close()


fig, axes = plt.subplots(3, 1, figsize=(8, 20), dpi=100)

z_center, cross_length, guess_cross = calculate_crossection_mean(mesh_true, nx, ny, 153)
z_est, cross_length, guess_cross = calculate_crossection_mean(mesh_est, nx, ny, 153)
z_est_u, cross_length, guess_cross = calculate_crossection_mean(mesh_est_u, nx, ny, 153)
z_est_l, cross_length, guess_cross = calculate_crossection_mean(mesh_est_l, nx, ny, 153)
axes[0].plot(guess_cross, z_center, 'r.-', label='true')
axes[0].plot(guess_cross, z_est, 'k.-', label='estimate')
axes[0].plot(guess_cross, z_est_u, 'k--', label='95% credible interval')
axes[0].plot(guess_cross, z_est_l, 'k--')
axes[0].set_title('(a) i = 153 with std(error) = 0.05 m/s', fontsize=24, loc='left')
axes[0].legend(loc="upper right")


z_center, cross_length, guess_cross = calculate_crossection_mean(mesh_true, nx, ny, 200)
z_est, cross_length, guess_cross = calculate_crossection_mean(mesh_est, nx, ny, 200)
z_est_u, cross_length, guess_cross = calculate_crossection_mean(mesh_est_u, nx, ny, 200)
z_est_l, cross_length, guess_cross = calculate_crossection_mean(mesh_est_l, nx, ny, 200)
axes[1].plot(guess_cross, z_center, 'r.-', guess_cross, z_est, 'k.-', guess_cross, z_est_u, 'k--',guess_cross, z_est_l, 'k--')
axes[1].set_title('(b) i = 200 with std(error) = 0.05 m/s', fontsize=24, loc='left')
axes[1].set_ylabel('$z_b$ [m]')

z_center, cross_length, guess_cross = calculate_crossection_mean(mesh_true, nx, ny, 753)
z_est, cross_length, guess_cross = calculate_crossection_mean(mesh_est, nx, ny, 753)
z_est_u, cross_length, guess_cross = calculate_crossection_mean(mesh_est_u, nx, ny, 753)
z_est_l, cross_length, guess_cross = calculate_crossection_mean(mesh_est_l, nx, ny, 753)
axes[2].plot(guess_cross, z_center, 'r.-', guess_cross, z_est, 'k.-', guess_cross, z_est_u, 'k--',guess_cross,  z_est_l, 'k--')
axes[2].set_title('(c) i = 753 with std(error) = 0.05 m/s', fontsize=24, loc='left')
axes[2].set_xlabel('across channel [m]', fontsize=25)
#plt.tight_layout()

plt.savefig('red_inset_acrosschannel.png', bbox_inches='tight', pad_inches=0, dpi=300)
#plt.show()
plt.close()
