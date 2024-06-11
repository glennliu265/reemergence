#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Visualize Mean states in CESM1

Uses data processed by:
    
    [calc_monmean_CESM1.py] : SST,SSS
    

Created on Tue Jun 11 12:25:53 2024

@author: gliu

"""

import xarray as xr
import numpy as np
import matplotlib as mpl

import matplotlib.pyplot as plt
import sys
import glob
import os

import tqdm
import time

import cartopy.crs as ccrs

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

#%% Plotting Information

bboxplot                    = [-80,0,20,65]
mpl.rcParams['font.family'] = 'Avenir'
mons3                       = proc.get_monstr(nletters=3)
fsz_tick                    = 18
fsz_axis                    = 14
fsz_title                   = 16

proj                        = ccrs.PlateCarree()

#%% Load necessary data

ds_uvel,ds_vvel = dl.load_current()
ds_bsf          = dl.load_bsf(ensavg=False)
ds_ssh          = dl.load_bsf(ensavg=False,ssh=True)

# Load data processed by [calc_monmean_CESM1.py]
ds_sss          = dl.load_monmean('SSS')
ds_sst          = dl.load_monmean('SST')


tlon  = ds_uvel.TLONG.mean('ens').values
tlat  = ds_uvel.TLAT.mean('ens').values

#%% Load Masks

# Load Land Ice Mask
icemask  = xr.open_dataset(input_path + "masks/CESM1LE_HTR_limask_pacificmask_enssum_lon-90to20_lat0to90.nc")


mask = icemask.MASK.squeeze()
mask_plot = xr.where(np.isnan(mask),0,mask)#mask.copy()

mask_apply = icemask.MASK.squeeze().values
#mask_plot[np.isnan(mask)] = 0


#%% Start Visualization (Ens Mean)

im          = 0
qint        = 1
contourvar  = "SSS"
cints_bsf   = np.arange(-50,55,5)
cints_sst   = np.arange(250,310,2)
cints_sss   = np.arange(33,39,.3)
cints_ssh   = np.arange(-200,200,10)

for im in range(12):
    
    # Initialize Plot and Map
    fig,ax,_    = viz.init_orthomap(1,1,bboxplot,figsize=(12,4))
    ax          = viz.add_coast_grid(ax,bbox=bboxplot,fill_color="lightgray")
    ax.set_title(mons3[im],fontsize=fsz_title)
    
    
    # Plot Currents
    plotu = ds_uvel.isel(month=im).UVEL.mean('ens').values
    plotv = ds_vvel.isel(month=im).VVEL.mean('ens').values
    ax.quiver(tlon[::qint,::qint],tlat[::qint,::qint],plotu[::qint,::qint],plotv[::qint,::qint],
              color='navy',transform=proj,alpha=0.75)
    
    if contourvar == "BSF":
        # Plot BSF
        plotbsf = ds_bsf.BSF.mean('ens').isel(mon=im).transpose('lat','lon')
        ax.contour(plotbsf.lon,plotbsf.lat,plotbsf,transform=proj,levels=cints_bsf,
                   linewidths=0.75,colors="k",)
    elif contourvar == "SSH":
        plotbsf = ds_ssh.SSH.mean('ens').isel(mon=im).transpose('lat','lon')
        cl = ax.contour(plotbsf.lon,plotbsf.lat,plotbsf,transform=proj,levels=cints_ssh,
                   linewidths=0.75,colors="k",)
        ax.clabel(cl)
    
    elif contourvar == "SST":
        # Plot mean SST
        plotvar = ds_sst.SST.mean('ens').isel(mon=im).transpose('lat','lon') * mask_apply
        cl = ax.contour(plotvar.lon,plotvar.lat,plotvar,transform=proj,
                    linewidths=1.5,colors="hotpink",levels=cints_sst)
        ax.clabel(cl)
    
    elif contourvar == 'SSS':
        # Plot mean SSS
        plotvar = ds_sss.SSS.mean('ens').isel(mon=im).transpose('lat','lon') * mask_apply
        cl = ax.contour(plotvar.lon,plotvar.lat,plotvar,transform=proj,
                    linewidths=1.5,colors="cornflowerblue",levels=cints_sss,linestyles='dashed')
        ax.clabel(cl)
    
    figname = "%s%s_Current_Comparison_mon%02i.png" % (figpath,contourvar,im+1)
    plt.savefig(figname,dpi=150,bbox_inches='tight')
    
    
    
#%% To Do
"""

- Make some regional visualizations
- Add contours of TEMP and SALT

"""
#%% Visualize Subpolar North Atlantic

# bboxice  = 
# fig,ax,_ = viz.init_orthomap(1,1,bboxice,figsize=(12,4))

# im = 1









