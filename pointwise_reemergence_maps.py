#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Compare Re-emergence "Maps" by base month Between Simulations

This includes:
~ Time Series ~
    - Monthly ACF (Correlation vs Lag)
    - Base Month vs. Lag

~ Spatial Maps ~
    - Mean Squared Error (I feel like i did this in another script, need to find this)
    - Difference over times of key re-emergence features...


Created on Thu Jun 27 14:10:35 2024

@author: gliu

"""


import xarray as xr
import numpy as np
import matplotlib as mpl
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

#%% Indicate the experiments/ load the files 

# Variable Information
vnames      = ["SST","SSS"]
vcolors     = ["hotpink","navy"]
vunits      = ["$\degree C$","$psu$"]
vmarkers    = ['o','x']

# CESM NetCDFs
ncs_cesm    = ["CESM1_1920to2005_SSTACF_lag00to60_ALL_ensALL.nc",
            "CESM1_1920to2005_SSSACF_lag00to60_ALL_ensALL.nc"]

# Note I might need to rerun all of this...
ncs_sm      = ["SM_SST_EOF_LbddCorr_Rerun_SST_autocorrelation_thresALL_lag00to60.nc",
            "SM_SSS_EOF_LbddCorr_Rerun_lbdE_neg_SSS_autocorrelation_thresALL_lag00to60.nc"]

compare_name = "CESMvSM_PaperOutline"

# Load Pointwise ACFs for each runs
st          = time.time()
ds_cesm     = []
ds_sm       = []
for vv in range(2):
    
    # Load CESM (ens, lon, lat, mons, thres, lags)
    ds = xr.open_dataset(procpath + ncs_cesm[vv]).acf.load()
    ds_cesm.append(ds.copy())
    
    # Load SM
    ds = xr.open_dataset(procpath + ncs_sm[vv])[vnames[vv]].load()
    ds_sm.append(ds.copy())
print("Loaded output in %.2fs" % (time.time()-st))


acf_lw      = 2.5

ds_all      = ds_cesm + ds_sm
lags        = ds_all[0].lags.data
expnames    = ["CESM SST","CESM SSS","SM SST","SM SSS"]
expcols     = ["firebrick","navy","lightsalmon","cornflowerblue"]
expls       = ["solid","solid","dashed","dashed"]
expmarkers  = ["o","d","+","x"]

#%% Standardize the output

#%% Plotting options

bboxplot                    = [-80,0,20,65]
mpl.rcParams['font.family'] = 'Avenir'
mons3                       = proc.get_monstr(nletters=3)
fsz_tick                    = 18
fsz_axis                    = 14
fsz_title                   = 16

proj                        = ccrs.PlateCarree()

# -----------------------------------------------------------------------------
#%% Part 1: Point Analysis (compare CESM1 and Stochastic Model ACFs at a point)
# -----------------------------------------------------------------------------

# Get Point Data
lonf                        = -30
latf                        = 50
locfn,loctitle              = proc.make_locstring(lonf,latf,lon360=True)

#%% Plot ACF for a month

im      = 1
xtks    = np.arange(0,63,3)

fig,ax  = plt.subplots(1,1,figsize=(12,4.5),constrained_layout=True)
ax,ax2  = viz.init_acplot(im,xtks,lags,ax=ax,title="")
ax.set_title("%s Anomaly Autocorrelation @ %s" % (mons3[im],loctitle),fontsize=24)


for ex in range(4):
    
    # Get the Dataset
    plotds = ds_all[ex].isel(mons=im).sel(lon=lonf,lat=latf,method='nearest').squeeze()
    
    # Plot each ensemble member for CESM1
    if ex < 2:
        nens   = len(plotds.ens)
        for e in range(nens):
            ax.plot(lags,plotds.isel(ens=e),alpha=0.05,label="",
                    c=expcols[ex],ls=expls[ex])
        ax.plot(lags,plotds.mean('ens'),lw=acf_lw,label=expnames[ex],
                c=expcols[ex],ls=expls[ex])
    # Just plot result for SM
    else:
        ax.plot(lags,plotds,lw=acf_lw,label=expnames[ex],
                c=expcols[ex],ls=expls[ex])

ax.legend()

savename = "%s%s_SST_SSS_ACF_%s_mon%02i.png" % (figpath,compare_name,locfn,im+1)
plt.savefig(savename,dpi=150,bbox_inches='tight')

#%% Plot the "Re-emergence Map"

vlms = [-1,1]
fig,axs = plt.subplots(4,1,figsize=(12,10),constrained_layout=True)

plotorder = [0,2,1,3]

for aa in range(4):
    
    ax  = axs[aa]
    ex  = plotorder[aa]
    
    # Get the Dataset
    plotds = ds_all[ex].sel(lon=lonf,lat=latf,method='nearest').squeeze()
    if ex < 2:
        plotds = plotds.mean('ens')
    
    
    ax.set_xticks(xtks)
    ax.set_ylabel(expnames[ex],color=expcols[ex],fontsize=22)
    pcm = ax.pcolormesh(lags,mons3,plotds,cmap='cmo.balance',vmin=vlms[0],vmax=vlms[1])
    ax.invert_yaxis()
    
    if aa == 3:
        ax.set_xlabel("Lags (Month)",fontsize=fsz_axis)
    

plt.suptitle("Monthly Lagged Autocorrelation Functions @ %s" % (loctitle),fontsize=fsz_title)
cb = viz.hcbar(pcm,ax=axs.flatten(),fraction=0.035)
cb.set_label("Correlation",fontsize=fsz_axis)

savename = "%s%s_SST_SSS_Correlation_Maps_%s.png" % (figpath,compare_name,locfn)
plt.savefig(savename,dpi=150,bbox_inches='tight')

#%% Plot Re-emergence Map Differences (SM - CESM)

vlms = [-.5,.5]
fig,axs = plt.subplots(2,1,figsize=(12,5.5),constrained_layout=True)

plotorder = [0,2,1,3]

for aa in range(2):
    
    ax  = axs[aa]
    if aa == 0:
        # Compare SST
        plotds_cesm = ds_all[0].sel(lon=lonf,lat=latf,method='nearest').squeeze().mean('ens')
        plotds_sm   = ds_all[2].sel(lon=lonf,lat=latf,method='nearest').squeeze()
        plotname    = "SST"#"SM - CESM (SST)"
        
    else:
        # Compare SSS
        plotds_cesm = ds_all[1].sel(lon=lonf,lat=latf,method='nearest').squeeze().mean('ens')
        plotds_sm   = ds_all[3].sel(lon=lonf,lat=latf,method='nearest').squeeze()
        plotname    = "SSS"#"SM - CESM (SSS)"
    
    
    plotds          = plotds_sm - plotds_cesm
    
    ax.set_xticks(xtks)
    ax.set_ylabel(plotname,color=expcols[ex],fontsize=22)
    pcm = ax.pcolormesh(lags,mons3,plotds,cmap='cmo.balance',vmin=vlms[0],vmax=vlms[1])
    ax.invert_yaxis()
    
    if aa == 3:
        ax.set_xlabel("Lags (Month)",fontsize=fsz_axis)
    

plt.suptitle("Monthly Lagged Autocorrelation Functions @ %s" % (loctitle),fontsize=fsz_title)
cb = viz.hcbar(pcm,ax=axs.flatten(),fraction=0.035)
cb.set_label("Correlation Difference (SM - CESM)",fontsize=fsz_axis)

savename = "%s%s_SST_SSS_Correlation_Differences_%s.png" % (figpath,compare_name,locfn)
plt.savefig(savename,dpi=150,bbox_inches='tight')



