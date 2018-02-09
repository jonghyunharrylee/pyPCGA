#! /usr/bin/env python

"""
Collect some plotting routines for Red River
"""
from matplotlib import pyplot as plt
import numpy as np

def plot_bathymetry_physical_domain(meshnode,triangles,s,title=None,ax=None):
    """
    Plot s on the triangular domain given by meshnode and triangles

    creates a new axis if none is specified
    """

    if ax is None:
        ax = plt.gca()

    im = ax.tripcolor(meshnode[:, 0], meshnode[:, 1], triangles, meshnode[:, 2], cmap=plt.get_cmap('jet'),
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

def plot_observations_comparison(obs,simul_obs,ax=None):
    """
    Compare simulated observations with true observations
    """
    if ax is None:
        ax = plt.gca()
    
    nobs = obs.shape[0]
    ax.set_title('obs. vs simul.')
    ax.plot(obs, simul_obs, '.')
    ax.set_xlabel('observation')
    ax.set_ylabel('simulation')
    minobs = np.vstack((obs, simul_obs)).min(0)
    maxobs = np.vstack((obs, simul_obs)).max(0)
    
    ax.plot(np.linspace(minobs, maxobs, 20), np.linspace(minobs, maxobs, 20), 'k-')
    ax.axis('equal')
    ax.set_xlim([np.floor(minobs), np.ceil(maxobs)])
    ax.set_ylim([np.floor(minobs), np.ceil(maxobs)])

    return ax
