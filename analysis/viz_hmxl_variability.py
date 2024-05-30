#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Visualize MLD variability

Created on Wed May 29 13:55:00 2024

@author: gliu

"""




import xarray as xr
import numpy as np
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import cartopy.crs as ccrs

import matplotlib.pyplot as plt
import sys
import glob
import os

import tqdm
import time

# ----------------------------------
# %% Import custom modules and paths
# ----------------------------------

# Import re-eergemce parameters

# Indicate the Machine!
machine = "Astraeus"

# First Load the Parameter File
cwd = os.getcwd()
sys.path.append(cwd+ "/..")
import reemergence_params as rparams

# Paths and Load Modules
pathdict = rparams.machine_paths[machine]

sys.path.append(pathdict['amvpath'])
sys.path.append(pathdict['scmpath'])

# Set needed paths
figpath     = pathdict['figpath']
input_path  = pathdict['input_path']
output_path = pathdict['output_path']
procpath    = pathdict['procpath']
rawpath     = pathdict['raw_path']

#%% Import Custom Modules

# Import AMV Calculation
from amv import proc,viz
import amv.loaders as dl

# Import stochastic model scripts
import scm


#%% Other User Edits


proc.makedir(figpath)

# Plotting Params
mpl.rcParams['font.family'] = 'JetBrains Mono'
bboxplot                    = [-80,0,20,65]
proj                        = ccrs.PlateCarree()
#lon                         = ds.lon.values
#lat                         = ds.lat.values
mons3                       = proc.get_monstr()

# Font Sizes
fsz_title= 25
fsz_axis = 20
fsz_ticks= 18



#%% Load Mixed Layer Depth


ncname = "CESM1LE_HMXL_NAtl_19200101_20050101_bilinear.nc"
ds_mld = xr.open_dataset(rawpath+ncname).load()

#%% Compute the seasonal cycle and standard deivation

dsmon       = ds_mld.HMXL.groupby('time.month')
mldcycle    = dsmon.mean('time')
mldvary     = dsmon.std('time')


mldratio = mldvary/mldcycle

#%% Load other varibles to plot
dsref         = mldcycle

# Settings and load
bsf           = dl.load_bsf()

# Load Land Ice Mask
icemask       = xr.open_dataset(input_path + "/masks/CESM1LE_HTR_limask_pacificmask_enssum_lon-90to20_lat0to90.nc")

# Resize
bsf,icemask,_ = proc.resize_ds([bsf,icemask,dsref])
bsf_savg      = proc.calc_savg_mon(bsf)

#
mask          = icemask.MASK.squeeze()
mask_plot     = xr.where(np.isnan(mask),0,mask)#mask.copy()


#%% Plot how it varies
cints_bsf = np.arange(-50,60,10)
plot_im   = np.roll(np.arange(12),1)
fig,axs,_ = viz.init_orthomap(4,3,bboxplot,figsize=(16,14),)
vlms      = [0,.8]

for ii in range(12):
    ax = axs.flatten()[ii]
    im = plot_im[ii]
    ax = viz.add_coast_grid(ax,bbox=bboxplot,fill_color="lightgray")
    ax.set_title(mons3[im],fontsize=fsz_axis)
    
    plotvar = mldratio.mean('ensemble').isel(month=im)
    pcm     = ax.pcolormesh(plotvar.lon,plotvar.lat,plotvar,transform=proj,vmin=vlms[0],vmax=vlms[1],cmap='cmo.deep_r')
    
    
    # Plot BSF
    plotbsf = bsf.isel(mon=im).BSF * mask
    cl      = ax.contour(plotbsf.lon,plotbsf.lat,plotbsf,
                         levels=cints_bsf,colors="k",linewidths=0.75,transform=proj)
    
    
    # Plot Mask
    cl2      = ax.contour(mask.lon,mask.lat,mask_plot,
                         levels=[0,1,2],colors="w",linewidths=2,transform=proj)
        
        
cb = fig.colorbar(pcm,ax=axs.flatten(),orientation='horizontal',pad=0.01,fraction=0.02)
cb.set_label(r"MLD Ratio $\frac{\sigma_{h}}{\overline{h}}$",fontsize=fsz_title)
    

savename = "%sMLD_Ratio_HMXL_AllMon.png" % (figpath)
plt.savefig(savename,bbox_inches='tight',dpi=150)


#%% Make a plot of detrainment depth

ncnamedt   = input_path + "mld/CESM1_HTR_FULL_hdetrain_NAtl.nc"
ds_detrain = xr.open_dataset(ncnamedt).h.load()
cint_mld = np.arange(100,1500,100)


fig,axs,_ = viz.init_orthomap(4,3,bboxplot,figsize=(16,18),)
vlms      = [0,100]

for ii in range(12):
    ax = axs.flatten()[ii]
    im = plot_im[ii]
    ax = viz.add_coast_grid(ax,bbox=bboxplot,fill_color="lightgray")
    ax.set_title(mons3[im],fontsize=fsz_axis)
    
    plotvar = ds_detrain.mean('ens').isel(mon=im)
    pcm     = ax.pcolormesh(plotvar.lon,plotvar.lat,plotvar,transform=proj,vmin=vlms[0],vmax=vlms[1],cmap='cmo.dense')
    cl      = ax.contour(plotvar.lon,plotvar.lat,plotvar,transform=proj,levels=cint_mld,colors="w")
    
    # Plot BSF
    # plotbsf = bsf.isel(mon=im).BSF * mask
    # cl      = ax.contour(plotbsf.lon,plotbsf.lat,plotbsf,
    #                      levels=cints_bsf,colors="k",linewidths=0.75,transform=proj)
    
    
    # Plot Mask
    cl2      = ax.contour(mask.lon,mask.lat,mask_plot,
                         levels=[0,1,2],colors="cornflowerblue",linewidths=2,transform=proj,ls='dotted')
        
        
cb = fig.colorbar(pcm,ax=axs.flatten(),orientation='horizontal',pad=0.01,fraction=0.02)
cb.set_label(r"Detrain depth [m]",fontsize=fsz_title)
    

savename = "%sHdetrain_EnsAvg_AllMon.png" % (figpath)
plt.savefig(savename,bbox_inches='tight',dpi=150)


#%% Focus on OND

selmons  = [9,10,11]
cint_mld = np.arange(100,1500,50)
fig,ax,_ = viz.init_orthomap(1,1,bboxplot,figsize=(12,8),)
ax       = viz.add_coast_grid(ax,bbox=bboxplot,fill_color="lightgray")


ax.set_title("Detrainment Depths (SON)",fontsize=26)
plotvar = ds_detrain.mean('ens').isel(mon=selmons).mean('mon')
pcm     = ax.pcolormesh(plotvar.lon,plotvar.lat,plotvar,transform=proj,vmin=vlms[0],vmax=vlms[1],cmap='cmo.dense')
cl      = ax.contour(plotvar.lon,plotvar.lat,plotvar,transform=proj,levels=cint_mld,colors="w")
ax.clabel(cl)


# Plot Mask
cl2      = ax.contour(mask.lon,mask.lat,mask_plot,
                     levels=[0,1,2],colors="cornflowerblue",linewidths=2,transform=proj,ls='dotted')
    
cb      = fig.colorbar(pcm,ax=ax,orientation='horizontal',pad=0.01,fraction=0.035)
cb.set_label(r"Detrain depth [m]",fontsize=fsz_title)

savename = "%sHdetrain_EnsAvg_SON.png" % (figpath)
plt.savefig(savename,bbox_inches='tight',dpi=150)

