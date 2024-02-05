#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Script for debugging/making script for nicer maps
Created on Mon Feb  5 14:01:29 2024

@author: gliu

"""



import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import sys
from tqdm import tqdm
import copy
import glob

import matplotlib as mpl

#%% Import Custom Modules
amvpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/03_Scripts/" # amv module
scmpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/03_Scripts/stochmod/model/"

sys.path.append(amvpath)
sys.path.append(scmpath)

from amv import proc,viz
import scm
import amv.loaders as dl
import yo_box as ybx


#%% New packages to load?

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.path as mpath
import matplotlib.patches as patches
import matplotlib.ticker as mticker



#%% Other Setup
mpl.rcParams['font.family'] = 'Avenir'

bboxplot                    = [-80,0,0,65]
#%% Function Dev

# Set Defaults
centlon   =-40
centlat   = 35
precision = 40
dx        = 10
dy        = 5
frame_lw  = 2
frame_col = "k"
nrow      = 1
ncol      = 1
figsize   = (8,4.5)

#%%


def init_orthomap(bboxplot,centlon=-40,centlat=35,precision=40,
                  dx=10,dy=5,
                  frame_lw=2,frame_col="k",
                  nrow=1,ncol=1,figsize=(8,4.5)):
    # Intiailize Ortograpphic map over North Atlantic.
    # Based on : https://stackoverflow.com/questions/74124975/cartopy-fancy-box
    # The default lat/lon projection
    noProj = ccrs.PlateCarree(central_longitude=0)
    
    # Set Orthographic Projection
    myProj = ccrs.Orthographic(central_longitude=centlon, central_latitude=centlat)
    myProj._threshold = myProj._threshold/precision  #for higher precision plot
    
    # Initialize Figure
    fig,ax = plt.subplots(nrow,ncol,figsize=figsize,subplot_kw={'projection': myProj},
                          constrained_layout=True)
    
    # Get Line Coordinates
    xp,yp  = viz.get_box_coords(bboxplot,dx=dx,dy=dy)
    
    # Draw the line
    [ax_hdl] = ax.plot(xp,yp,
        color=frame_col, linewidth=frame_lw,
        transform=noProj)
    
    # Make a polygon and crop
    tx_path                = ax_hdl._get_transformed_path()
    path_in_data_coords, _ = tx_path.get_transformed_path_and_affine()
    polygon1s              = mpath.Path( path_in_data_coords.vertices)
    ax.set_boundary(polygon1s) # masks-out unwanted part of the plot
    

    
    mapdict={
        'noProj'     : noProj,
        'myProj'     : myProj,
        'line_coords': (xp,yp),
        'polygon'    : polygon1s,
        }
    return fig,ax,mapdict

#%%

bboxplot     = [-80,0,0,65]
fig,ax,mdict = viz.init_orthomap(1,1,bboxplot,figsize=(10,8.5),constrained_layout=True,)

# blabels     = viz.init_blabels()
# blabels['left']   = True
# blabels['bottom'] = False
# blabels['top'] = True
# blabels['right'] = True
# for ax in axs:

ax           = viz.add_coast_grid(ax,bboxplot,fill_color="lightgray",fontsize=20,
                                fix_lon=np.arange(-80,10,10),fix_lat=np.arange(0,70,10),grid_color="k")



# I couldn't figure this part out... (adding Zebra Stripes)
#vcode     = np.repeat([1,2,],len(path_in_data_coords.vertices)/2) # Path code: Should be same size as verticles?
#polygon1v = mpath.Path( path_in_data_coords.vertices, vcode)
# patch1s = patches.PathPatch(polygon1s, facecolor='none', ec="black", lw=7, zorder=100)
# patch1v = patches.PathPatch(polygon1v, facecolor='none', ec="white", lw=6, zorder=101)
# ax.add_patch(patch1s)
# ax.add_patch(patch1v)



