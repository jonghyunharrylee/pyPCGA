#! /usr/bin/env python

"""
Collect some plotting routines for Red River
"""
from matplotlib import pyplot as plt
import numpy as np

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

def plot_bathymetry_physical_domain(meshnode,triangles,title=None,ax=None):
    """
    Plot bathymetry given by meshnode on the triangular domain given by meshnode and triangles

    creates a new axis if none is specified
    """

    if ax is None:
        ax = plt.gca()

    im = ax.tripcolor(meshnode[:, 0], meshnode[:, 1], triangles,meshnode[:, 2], cmap=plt.get_cmap('jet'),
                       label='_nolegend_')
    ax.set_xlabel("Easting [m]")
    ax.set_ylabel("Northing [m]")
    ax.set_aspect('equal', adjustable='box')
    cbar = plt.colorbar(im, ax=ax,fraction=0.025, pad=0.05)
    cbar.set_label('Elevation [m]')

    ax.grid()
    ax.set_axisbelow(True)
    plt.tight_layout()
    if title is not None:
        ax.set_title(title)
 
    return ax

def plot_field_on_physical_domain(meshnode,triangles,s,title=None,ax=None,xlabel="Easting [m]",
                                  ylabel="Northing [m]",slabel='Elevation',set_aspect=True):
    """
    Plot s on the triangular domain given by meshnode and triangles

    creates a new axis if none is specified
    """

    if ax is None:
        ax = plt.gca()

    im = ax.tripcolor(meshnode[:, 0], meshnode[:, 1], triangles,s, cmap=plt.get_cmap('jet'),
                       label='_nolegend_')
    if xlabel is not None: ax.set_xlabel(xlabel)
    if ylabel is not None: ax.set_ylabel(ylabel)
    if set_aspect: ax.set_aspect('equal', adjustable='box')
    cbar = plt.colorbar(im, ax=ax,fraction=0.025, pad=0.05)
    cbar.set_label(slabel)

    ax.grid()
    ax.set_axisbelow(True)
    plt.tight_layout()
    if title is not None:
        ax.set_title(title)
 
    return ax

def plot_field_physical_domain_comparison_1x2(meshnode,triangles,s_list,titles):
    assert len(s_list) == 2
    assert len(titles) == 2
    fig, axes = plt.subplots(1, 2, sharey=True, figsize=(18, 8), dpi=100)

    for ip,s in enumerate(s_list):
        axes[ip] = plot_field_on_physical_domain(meshnode,triangles,s,titles[ip],ax=axes[ip],set_aspect=False)
        
    return fig,axes

def plot_observations_comparison(obs,simul_obs,title='obs. vs. simul.',ax=None):
    """
    Compare simulated observations with true observations
    """
    if ax is None:
        ax = plt.gca()
    
    nobs = obs.shape[0]
    ax.set_title(title)
    ax.plot(obs[0::2], simul_obs[0::2], 'bo')
    ax.plot(obs[1::2], simul_obs[1::2], 'ro')
    ax.legend(('$v_x$','$v_y$'))
    ax.set_xlabel('observed velocity')
    ax.set_ylabel('simulated velocity')
    minobs = min(obs.min(),simul_obs.min())
    maxobs = max(obs.max(),simul_obs.max())
    
    ax.plot(np.linspace(minobs, maxobs, 20), np.linspace(minobs, maxobs, 20), 'k-')
    ax.axis('equal')
    ax.set_xlim([np.floor(minobs), np.ceil(maxobs)])
    ax.set_ylim([np.floor(minobs), np.ceil(maxobs)])

    return ax

def plot_streamwise_section_comparison(mesh_true,mesh_est,mesh_est_u,mesh_est_l,
                                       nx,ny,J,use_transverse_average=False,title=None,ax=None,
                                       ylabel='$z_b$ [m]',xlabel='streamwise distance [m]'):
    if ax is None:
        ax = plt.gca()

    if use_transverse_average:
        z_center, guess_length = calculate_centerline(mesh_true, nx, ny, int(ny/2))
        z_center, trans_length = calculate_transverse_mean(mesh_true, nx, ny)
        z_est, trans_length = calculate_transverse_mean(mesh_est, nx, ny)
        z_est_u, trans_length = calculate_transverse_mean(mesh_est_u, nx, ny)
        z_est_l, trans_length = calculate_transverse_mean(mesh_est_l, nx, ny)
    else:
        z_center, guess_length = calculate_centerline(mesh_true, nx, ny, J)
        z_est, guess_length = calculate_centerline(mesh_est, nx, ny, J)
        z_est_u, guess_length = calculate_centerline(mesh_est_u, nx, ny, J)
        z_est_l, guess_length = calculate_centerline(mesh_est_l, nx, ny, J)

    ax.plot(guess_length, z_center, 'r.-' ,label='true')
    ax.plot(guess_length, z_est, 'k.-' ,label='estimate')
    ax.plot(guess_length, z_est_u, 'k--' ,label='95% credible interval')
    ax.plot(guess_length, z_est_l, 'k--')
    if title is not None:
        ax.set_title(title, fontsize=24, loc='left')
    ax.legend(loc='best')
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    return ax

def plot_streamwise_section_comparison_2x2(mesh_true,mesh_est,mesh_est_u,mesh_est_l,nx,ny,J_list,
                                           title_suffix = 'with std(error) = 0.05 m/s',
                                           ylabel='$z_b$ [m]',xlabel='streamwise distance [m]'):
    assert len(J_list) == 4

    fig,axes = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(20, 16), dpi=100)

    for ip in range(2):
        for jp in range(2):
            J = J_list[ip*2 + jp]
            if J is None:
                use_transverse_average = True
                title = 'transverse avg. ' + title_suffix
            else:
                use_transverse_average = False
                title='{0:3.0f}% slice '.format(J)
                title += title_suffix
            axes[ip,jp] = plot_streamwise_section_comparison(mesh_true,mesh_est,mesh_est_u,mesh_est_l,nx,ny,J,
                                                             use_transverse_average=use_transverse_average,
                                                             title=title,ylabel=ylabel,xlabel=xlabel,ax=axes[ip,jp])
    return fig,axes

def plot_channel_crosssection_comparison(mesh_true,mesh_est,mesh_est_u,mesh_est_l,
                                         nx,ny,I,title=None,ax=None,
                                         ylabel='$z_b$ [m]',xlabel='cross-channel distance [m]',
                                         loc='best'):
    if ax is None:
        ax = plt.gca()

    z_center, cross_length, guess_cross = calculate_crossection_mean(mesh_true, nx, ny, I)
    z_est, cross_length, guess_cross = calculate_crossection_mean(mesh_est, nx, ny, I)
    z_est_u, cross_length, guess_cross = calculate_crossection_mean(mesh_est_u, nx, ny, I)
    z_est_l, cross_length, guess_cross = calculate_crossection_mean(mesh_est_l, nx, ny, I)

    ax.plot(guess_cross, z_center, 'r.-', label='true')
    ax.plot(guess_cross, z_est, 'k.-', label='estimate')
    ax.plot(guess_cross, z_est_u, 'k--', label='95% credible interval')
    ax.plot(guess_cross, z_est_l, 'k--')
    ax.set_title(title, fontsize=24, loc='left')
    if loc is not None:
        ax.legend(loc=loc)
    if ylabel is not None:  ax.set_ylabel(ylabel)
    if xlabel is not None:  ax.set_xlabel(xlabel)


    return ax

def plot_channel_crosssection_comparison_3x1(mesh_true,mesh_est,mesh_est_u,mesh_est_l,
                                             nx,ny,I_list,title_suffix='with std(error)= 0.05 [m/s]',
                                             ylabel='$z_b$ [m]',xlabel='cross-channel distance [m]'):
    assert len(I_list) == 3

    fig, axes = plt.subplots(3, 1, figsize=(8, 20), dpi=100)
    alpha_label = ['a','b','c']
    for ip,I in enumerate(I_list):
        title = "({0}) {1:3.1f}% cross-section ".format(alpha_label[ip],I/float(nx)*100)
        title += title_suffix
        loc = "upper right"
        if ip > 0: loc = None
        xlabel_ax=None
        if ip == len(I_list)-1: xlabel_ax=xlabel
        axes[ip] = plot_channel_crosssection_comparison(mesh_true,mesh_est,mesh_est_u,mesh_est_l,
                                                        nx,ny,I,title=title,
                                                        ylabel=ylabel,xlabel=xlabel_ax,
                                                        loc=loc,ax=axes[ip])

    return fig,axes
