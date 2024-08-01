#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Visualize the Skewness of SST/SSS Anomalies

Created on Wed Jul 24 11:04:55 2024

@author: gliu

"""


import xarray as xr
import numpy as np
import matplotlib as mpl
import scipy as sp

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
fsz_axis                    = 22
fsz_title                   = 28

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


# Load Mixed-Layer Depth
mldpath = input_path + "mld/"
mldnc   = "CESM1_HTR_FULL_HMXL_NAtl_EnsAvg.nc"
ds_mld  = xr.open_dataset(mldpath+mldnc).h.load()

#ds_h          = dl.load_monmean('HMXL')

#%% Load Masks

# Load Land Ice Mask
icemask     = xr.open_dataset(input_path + "masks/CESM1LE_HTR_limask_pacificmask_enssum_lon-90to20_lat0to90.nc")

mask        = icemask.MASK.squeeze()
mask_plot   = xr.where(np.isnan(mask),0,mask)#mask.copy()

mask_apply  = icemask.MASK.squeeze().values
#mask_plot[np.isnan(mask)] = 0

# Load Gulf Stream
ds_gs   = dl.load_gs()
ds_gs   = ds_gs.sel(lon=slice(-90,-50))
ds_gs2  = dl.load_gs(load_u2=True)


#%% Load SST/SSS from 

# Loading Information
regionset       = "SSSCSU"
comparename     = "SST_SSS_CESM1"
expnames        = ["SST_CESM","SSS_CESM",]
expnames_long   = ["CESM1 (SST)","CESM1 (SSS)",]
expnames_short  = ["CESM1_SST","CESM1_SSS",]
ecols           = ["firebrick","navy",]
els             = ["solid","solid",]
emarkers        = ["o","d",]

cesm_exps       = ["SST_CESM","SSS_CESM",
                  "SST_cesm1le_5degbilinear","SSS_cesm1le_5degbilinear",]

#%%

nexps = len(expnames)
ds_all = []
for e in tqdm.tqdm(range(nexps)):
    
    # Get Experiment information
    expname        = expnames[e]
    
    if "SSS" in expname:
        varname = "SSS"
    elif "SST" in expname:
        varname = "SST"
    
    # For stochastic model output
    ds = dl.load_smoutput(expname,output_path)
    
    if expname in cesm_exps:
        print("Detrending and deseasoning")
        ds = proc.xrdeseason(ds[varname])
        ds = ds - ds.mean('ens')
        ds = xr.where(np.isnan(ds),0,ds) # Sub with zeros for now
    else:
        ds = ds[varname]
        
    ds_all.append(ds)

#%% Use xr.reduce to apply skewness function along a dimension


ds = ds_all[0]

dsskew =  [ds.groupby('time.month').reduce(func=sp.stats.skew,dim='time') for ds in ds_all]


#%% Plot Skewness for a given month

im            = 1
for im in range(12):
    vlms      = [-1.5,1.5]
    
    fig,axs,_ = viz.init_orthomap(1,2,bboxplot,figsize=(20,8.00))
    
    for ii in range(2):
        ax      = axs.flatten()[ii]
        ax      = viz.add_coast_grid(ax,bboxplot,fill_color="lightgray",fontsize=20,
                                fix_lon=np.arange(-80,10,10),fix_lat=np.arange(0,70,10),grid_color="k")
        
        
        plotvar = dsskew[ii].isel(month=im).mean('ens')
        vname   = plotvar.name
        ax.set_title("%s Skewness" % (vname),fontsize=fsz_title)
        pcm     = ax.pcolormesh(plotvar.lon,plotvar.lat,plotvar,
                                transform=proj,
                                cmap='cmo.balance',
                                vmin=vlms[0],vmax=vlms[1],zorder=-1)
        
        cb      = viz.hcbar(pcm,ax=ax)
        cb.ax.tick_params(labelsize=fsz_tick)
        
        
        # Plot Currents
        qint=2
        plotu = ds_uvel.UVEL.mean('ens').mean('month').values
        plotv = ds_vvel.VVEL.mean('ens').mean('month').values
        ax.quiver(tlon[::qint,::qint],tlat[::qint,::qint],plotu[::qint,::qint],plotv[::qint,::qint],
                  color='navy',transform=proj,alpha=0.25)
        
        
        # Plot Gulf Stream Position
        #ax.plot(ds_gs.lon,ds_gs.lat.mean('ens'),transform=proj,lw=1.75,c="k",ls='dashed')
        ax.plot(ds_gs2.lon.mean('mon'),ds_gs2.lat.mean('mon'),transform=proj,lw=1.75,c='k',ls='dashdot')
    
        # Plot Ice Edge
        ax.contour(icemask.lon,icemask.lat,mask_plot,colors="cyan",linewidths=2.5,
                   transform=proj,levels=[0,1],zorder=-1)
    plt.suptitle("Skewness for %s" % (mons3[im]),fontsize=fsz_title+5)
    
    savename = "%sSST_Skewness_mon%02i.png" % (figpath,im+1)
    plt.savefig(savename,dpi=150,bbox_inches='tight')



