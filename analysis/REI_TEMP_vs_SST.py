#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Compare TEMP and SST REI

Created on Tue Jun 25 11:48:18 2024

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

#%% plotting Parameters (from viz_icefrac)

bboxice                     = [-70,-10,55,70]
bboxplot                    = [-80,0,10,65]
mpl.rcParams['font.family'] = 'Avenir'
mons3                       = proc.get_monstr(nletters=3)
fsz_tick                    = 18
fsz_axis                    = 14
fsz_title                   = 16
rhocrit                     = proc.ttest_rho(0.05,2,86)

proj                        = ccrs.PlateCarree()

#%% Load Masks

# Load Land Ice Mask
icemask  = xr.open_dataset(input_path + "masks/CESM1LE_HTR_limask_pacificmask_enssum_lon-90to20_lat0to90.nc")


mask = icemask.MASK.squeeze()
mask_plot = xr.where(np.isnan(mask),0,mask)#mask.copy()

mask_apply = icemask.MASK.squeeze().values

#%% Load the files

#nctemp="CESM1_1920to2005_TEMPACF_lag00to60_ALL_ensALL.nc"
# expname_temp = "TEMP_CESM"
# expname_sst  = "SST_CESM"


vnames       = ["SST","TEMP"]
ds_rei       = []
for vv in range(2):
    vname  = vnames[vv]
    ldpath = output_path + "%s_CESM/Metrics/REI_Pointwise.nc" % (vname)
    ds     = xr.open_dataset(ldpath).load()
    ds_rei.append(ds)

#%% Compare the Re-emergence Index  for a select onth and yr

im = 1
iy = 0
for im in range(12):
    
    levels  = np.arange(0,0.55,0.05)
    plevels = np.arange(0,0.6,0.1)
    cmapin  = 'cmo.dense'
    
    
    fig,axs,_ = viz.init_orthomap(1,2,bboxice,figsize=(20,5))
    
    for vv in range(2):
        
        ax      = axs[vv]
        
        ax           = viz.add_coast_grid(ax,bboxice,fill_color="lightgray",fontsize=20,grid_color="k")
        
        plotvar      = ds_rei[vv].rei.isel(mon=im,yr=iy).mean('ens')
        lon          = plotvar.lon
        lat          = plotvar.lat
        pcm          = ax.contourf(lon,lat,plotvar,cmap=cmapin,levels=levels,transform=proj,extend='both')
        cl           = ax.contour(lon,lat,plotvar,colors='darkslategray',linewidths=.5,linestyles='solid',levels=levels,transform=proj)
        
        ax.set_title(vnames[vv],fontsize=fsz_title+10)
        
        
        ax.contour(mask_plot.lon,mask_plot.lat,mask_plot,levels=[0,1],colors="cyan",linestyles="dotted",transform=proj,linewidths=2)
        
    cb = fig.colorbar(pcm,ax=axs.flatten(),fraction=0.0105,pad=0.01)
    cb.ax.tick_params(labelsize=fsz_tick)
    cb.set_label("Re-emergence Index",fontsize=fsz_axis+10)
    
    plt.suptitle("Year %i %s Re-emergence" % (iy+1,mons3[im]),fontsize=fsz_axis+15)
    
    savename = "%sTEMP_v_SST_Reemergence_mon%02i_year%i.png" % (figpath,im+1,iy+1)
    plt.savefig(savename,dpi=150,bbox_inches='tight')
    
    # ds_rei = [ds.rei for ds in ds)]
    # rei_temp,rei_sst = ds_rei

    


