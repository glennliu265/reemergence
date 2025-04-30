#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""


Compare some differences between the runs


Created on Thu Apr  3 17:44:28 2025

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


#%%

runids      = [0,1,2,3,4]
smids       = ["SST_Draft03_Rerun_QekCorr","SST_Revision_Qek_TauReg"]

smids       = ["SSS_Draft03_Rerun_QekCorr","SSS_Revision_Qek_TauReg"]

ds_all      = [dl.load_smoutput(smm,output_path,runids=runids) for smm in smids]




ds_stds     = [ds.std('time') for ds in ds_all]


#%% Compare Standard Deivation

ds_stds = [ds.std('time') for ds in ds_all]

#diff    = ds_stds[1].SST.mean('run') - ds_stds[0].SST.mean('run')
diff    = ds_stds[1].SSS.mean('run') - ds_stds[0].SSS.mean('run')

#%% Plot Difference in Sigma

proj   = ccrs.PlateCarree()
bbplot = [-80,0,20,65]
#vmax   = 0.025
vmax    = 0.025
fig,ax = viz.init_regplot(bboxin=bbplot)


plotvar = diff
pcm    = ax.pcolormesh(plotvar.lon,plotvar.lat,plotvar,transform=proj,
                       cmap='cmo.balance',vmin=-vmax,vmax=vmax)
cb = viz.hcbar(pcm,ax=ax)

#%%
proj   = ccrs.PlateCarree()
bbplot = [-80,0,20,65]
vmax   = 0.5
fig,ax = viz.init_regplot(bboxin=bbplot)


#plotvar = ds_stds[1].SST.mean('run')
plotvar = ds_stds[1].SSS.mean('run')
pcm     = ax.pcolormesh(plotvar.lon,plotvar.lat,plotvar,transform=proj,
                       cmap='cmo.thermal',vmin=0,vmax=vmax)
cb = viz.hcbar(pcm,ax=ax)


#%% Get metrics for the point

lonf = -30
latf = 50
dspts = [proc.selpt_ds(ds,lonf,latf) for ds in ds_all]


#ssts  = [ds.SST.data for ds in dspts]
ssts  = [ds.SSS.data for ds in dspts]
ssts  = [np.concatenate(s,0) for s in ssts]

tm    = scm.compute_sm_metrics(ssts)

#%%

lags   = np.arange(37)
kmonth = 1
xtks   = lags[::3]

fig,ax = plt.subplots(1,1,constrained_layout=True,figsize=(8,4))

ax,_   = viz.init_acplot(kmonth,xtks,lags,ax=ax)

acf = tm['acfs'][kmonth][1]
ax.plot(lags,acf,label="Updated")

acf = tm['acfs'][kmonth][0]
ax.plot(lags,acf,label="Old")
ax.legend()

ax.set_title("Persistence at SPG Point (50N, 30W)")
#%% Load the NAO for comparison

fpath = input_path + "forcing/"

ncuek        = "CESM1_HTR_FULL_Qek_SST_NAO_nomasklag1_nroll0_NAtl_concatEns.nc"
nctaunao     = "CESM1_HTR_FULL_Monthly_TAU_NAO_nomasklag1_nroll0_concatEns.nc"

ncqek_regtau = "CESM1_HTR_FULL_Qek_SSS_NAO_nomasklag1_nroll0_NAtl_concatEns.nc"
ncqek_dirreg = "CESM1_HTR_FULL_Qek_SST_NAO_DirReg_NAtl_concatEns_corrected_EnsAvgFirst.nc"

# Load everything and check
