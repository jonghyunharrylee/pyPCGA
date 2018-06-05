#! /usr/bin/env python
"""
A few quick and dirty routines hardwired for plotting stwave example
"""
import numpy as np
import matplotlib.pyplot as plt

def get_transect(nx,ny,dx,dy,s_true2d,post_std2d,ix):
    linex = np.arange(1,nx+1)*dx
    line1_true = s_true2d[ny-ix+1,:]
    line1 = s_hat2d[ny-ix+1,:]
    line1_u = s_hat2d[ny-ix+1,:] + 1.96*post_std2d[ny-ix+1,:]
    line1_l = s_hat2d[ny-ix+1,:] - 1.96*post_std2d[ny-ix+1,:]

    return linex,line1_true,line1,line1_u,line1_l


def plot_transect(nx,ny,dx,dy,s_true2d,post_std2d,ix,axes=None,linewidth=3):
    if axes is None:
        axes = plt.gca()
    linex,line1_true,line1,line1_u,line1_l = get_transect(nx,ny,dx,dy,s_true2d,post_std2d,ix)
    axes.plot(linex, np.flipud(-line1_true),'r-', label='True',linewidth=linewidth)
    axes.plot(linex, np.flipud(-line1),'k-', label='Estimated',linewidth=linewidth)
    axes.plot(linex, np.flipud(-line1_u),'k--', label='95% credible interval',linewidth=linewidth)
    axes.plot(linex, np.flipud(-line1_l),'k--',linewidth=linewidth)
    #axes.plot(linex, np.flipud(-line1_X),'b--', label='Drift/Trend')
    axes.set_title('(a) {0:4.1f} m'.format(ix*dy), loc='left')
    handles, labels = axes.get_legend_handles_labels()
    axes.legend(handles, labels)
    axes.set_xlabel('Offshore distance [m]')
    axes.set_ylabel('z [m]')
    return axes

def plot_1x1(nx,ny,dx,dy,s0,fig=None,tit0='Estimate',
             vmin=[-7.],vmax=[0.],cmap_type='jet'):
    if fig is None:
        fig,axes = plt.subplots(1,1, figsize=(15,5))
    else:
        axes = fig.add_subplot(1,1,1,adjustable='box',aspect=1)


    linex = np.arange(1,nx+1)*dx
    liney = np.arange(1,ny+1)*dy
    im0 = axes.imshow(np.flipud(np.fliplr(-s0)), extent=[0, linex[-1], 0, liney[-1]], vmin=vmin[0], vmax=vmax[0], 
                        cmap=plt.get_cmap(cmap_type))
    axes.set_title('(a) {0}'.format(tit0), loc='left')
    axes.set_aspect('equal')
    axes.set_xlabel('Offshore distance [m]')
    axes.set_ylabel('Alongshore distance [m]')
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.65, 0.15, 0.05, 0.7])
    fig.colorbar(im0, cax=cbar_ax)


    return fig,axes


def plot_2x1(nx,ny,dx,dy,s0,s1,fig=None,tit0='True',tit1='Estimate',
             vmin=[-7.,-7.],vmax=[0.,0.],cmap_type='jet',colorbar_flag=0):
    if fig is None:
        fig,axes = plt.subplots(1,2, figsize=(15,5))
    else:
        axes = []
        axes.append(fig.add_subplot(1,2,1,adjustable='box',aspect=1))
        axes.append(fig.add_subplot(1,2,2,adjustable='box',aspect=1))


    linex = np.arange(1,nx+1)*dx
    liney = np.arange(1,ny+1)*dy
    im0 = axes[0].imshow(np.flipud(np.fliplr(-s0)), extent=[0, linex[-1], 0, liney[-1]], vmin=vmin[0], vmax=vmax[0], 
                        cmap=plt.get_cmap(cmap_type))
    axes[0].set_title('(a) {0}'.format(tit0), loc='left')
    axes[0].set_aspect('equal')
    axes[0].set_xlabel('Offshore distance [m]')
    axes[0].set_ylabel('Alongshore distance [m]')
    im1 = axes[1].imshow(np.flipud(np.fliplr(-s1)), extent=[0, linex[-1], 0, liney[-1]], vmin=vmin[1], vmax=vmax[1], 
                   cmap=plt.get_cmap(cmap_type))
    axes[1].set_title('(b) Estimate'.format(tit1), loc='left')
    axes[1].set_xlabel('Offshore distance [m]')
    axes[1].set_aspect('equal')
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    if colorbar_flag==1:
        fig.colorbar(im1, cax=cbar_ax)
    else:
        fig.colorbar(im0, cax=cbar_ax)



    return fig,axes

