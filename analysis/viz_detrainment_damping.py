#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Visualize Detrainment Damping (across all ensemble members)
Uses output of postprocessing script

Created on Wed Mar  6 13:15:12 2024

@author: gliu
"""


import numpy as np
import xarray as xr
import sys
import time
import matplotlib.pyplot as plt
import matplotlib as mpl
import cartopy.crs as ccrs
import os

# ----------------------------------
#%% Import custom modules and paths
# ----------------------------------

# Indicate the Machine!
machine = "Astraeus"

# First Load the Parameter File
sys.path.append("../")
import reemergence_params as rparams

# Paths and Load Modules
machine    = "Astraeus"
pathdict   = rparams.machine_paths[machine]

sys.path.append(pathdict['amvpath'])
sys.path.append(pathdict['scmpath'])
from amv import proc,viz
import scm
import amv.loaders as dl
import yo_box as ybx

# Set needed paths
figpath     = pathdict['figpath']
proc.makedir(figpath)
input_path  = pathdict['input_path']
output_path = pathdict['output_path']

# ----------------------------------
#%% User Edits
# ----------------------------------

fn       = "CESM1_HTR_FULL_SSS_Expfit_lbdd_monvar_detrendensmean_lagmax3_EnsAll.nc"
varname  = "SSS"
loadpath = input_path + "damping/" 

# Constants
dt    = 3600*24*30 # Timestep [s]
cp    = 3850       # 
rho   = 1026       # Density [kg/m3]
B     = 0.2        # Bowen Ratio, from Frankignoul et al 1998
L     = 2.5e6      # Specific Heat of Evaporation [J/kg], from SSS model document


debug = False


#%% Load in the data array

ds = xr.open_dataset(loadpath+fn) # (ens, mon, lat, lon)

#%% Visualize some of the inputs

# Set up mapping template
# Plotting Params
mpl.rcParams['font.family'] = 'JetBrains Mono'
bboxplot                    = [-80,0,20,65]
proj                        = ccrs.PlateCarree()
lon                         = ds.lon.values
lat                         = ds.lat.values
mons3                       = proc.get_monstr()

plotmon                     = np.roll(np.arange(12),1)

fsz_title= 26
fsz_axis = 22
fsz_lbl  = 10



#%% 
im = 0
ie = 0

lbd_d = ds.lbd_d

vlms_base      = [0,0.5]#None#,[0,.3]
vlms_diff      = [0,0.5]#None#,[-.2,.2]
im             = 0

cints_base     = [12,24,36,48,60,72]
cints_diff     = [-12,-6,-3,0,3,6,12]
linecol        = "k"


cmap = 'inferno'
cints = cints_base

fig,axs,mdict                = viz.init_orthomap(1,2,bboxplot,figsize=(14,8),constrained_layout=True,)
for a,ax in enumerate(axs):
    
    ax                          = viz.add_coast_grid(ax,bbox=bboxplot,fill_color="lightgray")
    
    if a == 0:
        plotvar = lbd_d.isel(ens=ie,mon=im) #* -1 # x1 was for case when it was wrong computed
        title = "Ens %02i" % (ie+1)
    elif a == 1:
        plotvar = lbd_d.isel(mon=im).mean('ens')# * -1
        title = "Ens Avg" 
    
    # if a <2:
    #     plotvar = ds_all[a].lbd_d.isel(mon=im) 
    #     cmap    = "inferno"
    #     title   = fnnames[a]
    #     vlms    = vlms_base
    #     cints   = cints_base
    #     linecol = "w"
    
    # else:
    #     plotvar = (-ds_all[1].lbd_d.isel(mon=im) - -ds_all[0].lbd_d.isel(mon=im))
    #     cmap    = 'cmo.balance'
    #     title   =  "%s - %s" % (fnnames[1],fnnames[0])
    #     vlms    = vlms_diff
    #     cints   = cints_diff
    #     linecol = "k"
    
    if a == 0:
        vlms = vlms_base#[-.5,0]
    elif a == 1:
        vlms = vlms_base#[0,.2]
    
    ax.set_title(title)
    if vlms is None:
        pcm     = ax.pcolormesh(lon,lat,plotvar,transform=proj,cmap=cmap)
    else:
        pcm     = ax.pcolormesh(lon,lat,plotvar,transform=proj,vmin=vlms[0],vmax=vlms[1],cmap=cmap)
    
    # Contours
    if a <2:
        cl      = ax.contour(lon,lat,1/plotvar,levels=cints,colors=linecol,linewidths=.55,transform=proj)
        ax.clabel(cl,fontsize=8)
        
    # Colorbar
    cb      = fig.colorbar(pcm,ax=ax,orientation='horizontal',fraction=0.02)
    cb.set_label("Detrainment Damping (1/mon)")
    
plt.suptitle("%s. Detrainment Damping Comparison" % (mons3[im]),y=0.60,fontsize=16)




#%% Check for all ensembl members

im            = 1
nens          = 42

vlms          = [0,.5]
fig,axs,mdict = viz.init_orthomap(6,7,bboxplot,figsize=(30,24),constrained_layout=True,)

for e in range(nens):
    
    ax = axs.flatten()[e]
    ax = viz.add_coast_grid(ax,bbox=bboxplot,fill_color="lightgray")
    ax.set_title("Ens.%02i" % (e+1),fontsize=fsz_title)

    plotvar = lbd_d.isel(ens=e,mon=im) # * -1
    
    if vlms is None:
        pcm     = ax.pcolormesh(lon,lat,plotvar,transform=proj,cmap=cmap)
    else:
        pcm     = ax.pcolormesh(lon,lat,plotvar,transform=proj,vmin=vlms[0],vmax=vlms[1],cmap=cmap)

cb      = fig.colorbar(pcm,ax=axs.flatten(),orientation='horizontal',fraction=0.02)
cb.set_label("Detrainment Damping (1/mon)")

savename = "%sDetrainment_Damping_%s_mon%02i_AllEns.png" % (figpath,varname,im+1)
plt.savefig(savename,dpi=150,bbox_inches='tight')


#%% Look at the ensemble mean

lbdd_ensmean = lbd_d.mean('ens')
lbdd_ensmean.isel(mon=1).plot(vmin=0,vmax=0.5)

lbd_d.isel(mon=1,ens=32).plot(vmin=0,vmax=0.5)

#%% manual ensmean

lbddval= lbd_d.values # (42, 12, 48, 65)
lbddval[32,:,:,:] = np.nan

lbdd_ensmean1 = np.nanmean(lbddval,0)
 
 
plt.pcolormesh(lbdd_ensmean1[1,:,:],vmin=0,vmax=0.5),plt.colorbar()
#%%