#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Visualize Re-emergence in CMIP6

Created on Fri Feb  2 13:51:13 2024

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


#%%

datpath     = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/proc/CMIP6/proc/"
figpath     = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/01_hfdamping/02_Figures/20240202/"
proc.makedir(figpath)


#%% Load ACFs

e       = 0
varname = "SSS"
nc      = "%sCESM2_%s_ACF_1850to2014_lag00to60_ens%02i.nc" % (datpath,varname,e+1)

ds  = xr.open_dataset(nc).isel(thres=2)[varname]
acf = ds.values # [lon x lat x month x lag]
lon = ds.lon.values
lat = ds.lat.values

T2 = np.sum(acf**2,3)

# DOESN'T SEEM TO BE WORKING?
# remidx_all = []
# for  kmonth in range(12):
#     reidx = proc.calc_remidx_simple(acf,kmonth,monthdim=-2,lagdim=-1)
#     remidx_all.append(reidx)
# remidx_all = np.array(remidx_all)

#%%
kmonth = 1

for kmonth in range(12):
    mons3 = proc.get_monstr(nletters=3)
    levels = np.arange(0,21,1)
    
    mpl.rcParams['font.family'] = 'JetBrains Mono'
    bboxplot                    = [-80,0,0,65]
    fig,ax                      = viz.geosubplots(1,1,figsize=(10,6))
    ax                          = viz.add_coast_grid(ax,bbox=bboxplot,fill_color="k")
    
    pcm = ax.contourf(lon,lat,T2[:,:,kmonth].T,cmap='cmo.dense',levels=levels)
    cl = ax.contour(lon,lat,T2[:,:,kmonth].T,colors='k',linewidths=0.8,linestyles='dotted',levels=levels)
    ax.set_title("CESM2 Salinity %s $T^2$ (Ens %02i) " % (mons3[kmonth],e+1))
    fig.colorbar(pcm,ax=ax,fraction=0.025,orientation='horizontal',pad=0.01,)
    
    savename = "%sCESM2_%s_T2_mon%02i_ens%02i.png" % (figpath,varname,kmonth+1,e+1)
    plt.savefig(savename,dpi=150,bbox_inches='tight')

#%% Try for CESM1

cesm1ncs  = ['HTR-FULL_SST_autocorrelation_thres0.nc','HTR-FULL_SSS_autocorrelation_thres0.nc']
cesm1path = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/proc/"
vnames = ["SST","SSS"]

ds_opn = [xr.open_dataset(cesm1path+nc) for nc in cesm1ncs] # [thres x ens x lag x mon x lat x lon]
ds_opn = [ds.isel(thres=2) for ds in ds_opn]

ds  = ds_opn[0]
lon = ds.lon.values
lat = ds.lat.values
lag = ds.lag.values
ens = ds.ens.values
mon = ds.mon.values 

#%% Plot for a variable

ds = ds_opn[0]

vv = 0
vname = vnames[vv]

acf      = ds_opn[vv][vname].values # [Ens Lag Mon Lat Lon]
T2_cesm1 = np.sum(acf**2,1) 

#%%

e      = 0
kmonth = 0

for kmonth in range(12):
    mons3 = proc.get_monstr(nletters=3)
    levels = np.arange(0,21,1)
    
    mpl.rcParams['font.family'] = 'JetBrains Mono'
    bboxplot                    = [-80,0,0,65]
    fig,ax                      = viz.geosubplots(1,1,figsize=(10,6))
    ax                          = viz.add_coast_grid(ax,bbox=bboxplot,fill_color="k")
    plotvar = T2_cesm1[e,kmonth,:,:]
    pcm = ax.contourf(lon,lat,plotvar,cmap='cmo.dense',levels=levels)
    cl = ax.contour(lon,lat,plotvar,colors='k',linewidths=0.8,linestyles='dotted',levels=levels)
    ax.set_title("CESM1 %s %s $T^2$ (Ens %02i) " % (vname,mons3[kmonth],e+1))
    fig.colorbar(pcm,ax=ax,fraction=0.025,orientation='horizontal',pad=0.01,)
    
    savename = "%sCESM1_%s_T2_mon%02i_ens%02i.png" % (figpath,varname,kmonth+1,e+1)
    plt.savefig(savename,dpi=150,bbox_inches='tight')

#%% Fancy Plot
# https://stackoverflow.com/questions/74124975/cartopy-fancy-box


import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.path as mpath
import matplotlib.patches as patches
import matplotlib.ticker as mticker


kmonth = 0
e      = 0
varname=vname

mons3 = proc.get_monstr(nletters=3)
levels = np.arange(0,21,1)

mpl.rcParams['font.family'] = 'Avenir'
bboxplot                    = [-80,0,0,65]



#def init_natl(centlon=None,centlat=None,precision=None):
# 
# Based on : https://stackoverflow.com/questions/74124975/cartopy-fancy-box

# Set Defaults
centlon   =-40
centlat   = 35
precision = 40

nrow      = 1
ncol      = 1
figsize   = (8,4.5)

bboxplot  = [-80,0,0,65]


# Set Defaults
if centlon is None:
    centlon=-40
if centlat is None:
    centlat=35
if precision is None:
    precision=40.
    
    

#%%
# The default lat/lon projection
noProj = ccrs.PlateCarree(central_longitude=0)

# Set Orthographic Projection
myProj = ccrs.Orthographic(central_longitude=centlon, central_latitude=centlat)
myProj._threshold = myProj._threshold/precision  #for higher precision plot

fig,ax = plt.subplots(nrow,ncol,figsize=figsize,subplot_kw={'projection': myProj},
                      constrained_layout=True)





#ax                         = viz.add_coast_grid(ax,bbox=bboxplot,fill_color="k")

# Zebra-border-line segments ...
#  four edges on separate lines of code
# 1: lower edge: Left - Right
# 2: Right edge: Bottom - Top
# 3: Upper edge: Right - Left
# 4: Left edge: Top - Bottom

def get_box_coords(bbox,dx=None,dy=None):
    
    if dx is None:
        dx = np.linspace(bbox[0],bbox[1],5)
        dx = dx[1] - dx[0]
    if dy is None:
        dy = np.linspace(bbox[2],bbox[3],5)
        dy = dy[1] - dy[0]
    
    # Lower Edge (Bot. Left --> Bot. Right)
    lower_x = np.arange(bbox[0],bbox[1]+dx,dx) # x-coord
    nx = len(lower_x) 
    lower_y = [bbox[2],]*nx # y-coord
    
    # Right Edge (Bot. Right ^^^ Top Right)
    right_y = np.arange(bbox[2],bbox[3]+dy,dy)
    ny = len(right_y)
    right_x = [bbox[1],]*ny
    
    # Upper Edge (Top Left <-- Top Right)
    upper_x = np.flip(lower_x)
    upper_y = [bbox[3],]*nx
    
    # Left Edge (Bot. Left vvv Top Left)
    left_y  = np.flip(right_y)
    left_x  = [bbox[0],]*ny
    
    x_coords = np.hstack([lower_x,right_x,upper_x,left_x])
    y_coords = np.hstack([lower_y,right_y,upper_y,left_y])
    
    return x_coords,y_coords

# Get Line Coordinates
xp,yp = get_box_coords(bboxplot,dx=10,dy=5)

# Draw the line
[ax_hdl] = ax.plot(xp,yp,
    color='black', linewidth=0.5,
    transform=noProj)

# Make a polygon and crop
tx_path                = ax_hdl._get_transformed_path()
path_in_data_coords, _ = tx_path.get_transformed_path_and_affine()
polygon1s              = mpath.Path( path_in_data_coords.vertices)
#vcode     = np.repeat([1,2,],len(path_in_data_coords.vertices)/2) # Path code: Should be same size as verticles?
#polygon1v = mpath.Path( path_in_data_coords.vertices, vcode)
ax.set_boundary(polygon1s) # masks-out unwanted part of the plot

# patch1s = patches.PathPatch(polygon1s, facecolor='none', ec="black", lw=7, zorder=100)
# patch1v = patches.PathPatch(polygon1v, facecolor='none', ec="white", lw=6, zorder=101)
# ax.add_patch(patch1s)
# ax.add_patch(patch1v)

#ax.zebra_frame(lw=5, crs=noProj, zorder=3)

#gl = ax.gridlines(draw_labels=True, x_inline=False, y_inline=False,ls='dotted',colors='w')

ax.add_feature(cfeature.LAND, zorder=1, edgecolor='k',color="k")
#ax.title.set_text("Map with zebra border line")


# Add Gridlines
grid_color="w"
fix_lon   = False
fix_lat   = False
gl = ax.gridlines(crs=noProj, draw_labels=True,
              linewidth=2, color=grid_color, alpha=0.5, linestyle="dotted",
              lw=0.75)

# # Remove the degree symbol
# if ignore_error:
#     #print("Removing Degree Symbol")
#     gl.xformatter = LongitudeFormatter(zero_direction_label=False,degree_symbol='')
#     gl.yformatter = LatitudeFormatter(degree_symbol='')
#     #gl.yformatter = LatitudeFormatter(degree_symbol='')
#     gl.rotate_labels = False

if fix_lon:
    gl.xlocator = mticker.FixedLocator(fix_lon)
if fix_lat:
    gl.ylocator = mticker.FixedLocator(fix_lat)
    
    
gl.left_labels   = True
gl.right_labels  = False
gl.top_labels    = False
gl.bottom_labels = True

# gl.top_labels=False
# gl.right_labels=False

#ax.add_feature(cartopy.feature.OCEAN, linewidth=.3, color='lightblue')
# viz.plot_box
pcm = ax.contourf(lon,lat,plotvar,cmap='cmo.dense',levels=levels,transform=noProj)
plt.show()
# [ax_hdl] = ax.plot(
#     [
#         -45, -40, -35, -30, -25, -20, -15, -10, -5,
#         -5,-5,-5,-5,-5,
#         -10,-15,-20,-25,-30,-35,-40,-45,
#         -45, -45, -45, -45, -45
#     ],
#     [
#         45, 45, 45, 45, 45, 45, 45, 45, 45,
#         50, 55, 60, 65, 70,
#         70,70,70,70,70,70,70,70,
#         65, 60, 55, 50, 45 
#     ],
#     color='black', linewidth=0.5,
#     transform=noProj)







# #fig,ax                      = viz.geosubplots(1,1,figsize=(10,6))


# ax                          = viz.add_coast_grid(ax,bbox=bboxplot,fill_color="k")
# plotvar = T2_cesm1[e,kmonth,:,:]
# pcm = ax.contourf(lon,lat,plotvar,cmap='cmo.dense',levels=levels)
# cl  = ax.contour(lon,lat,plotvar,colors='k',linewidths=0.8,linestyles='dotted',levels=levels)
# ax.set_title("CESM1 %s %s T$^2$ (Ens. %02i) " % (vname,mons3[kmonth],e+1),fontsize=18)
# fig.colorbar(pcm,ax=ax,fraction=0.025,orientation='horizontal',pad=0.01,)

# savename = "%sCESM1_%s_T2_mon%02i_ens%02i.png" % (figpath,varname,kmonth+1,e+1)
# plt.savefig(savename,dpi=150,bbox_inches='tight')




