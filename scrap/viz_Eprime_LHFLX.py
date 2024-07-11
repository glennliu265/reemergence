#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

To figure out the sign of things, lets visualize Eprime, Fprime, 
and other forcing inputs

Created on Wed Jun 12 09:11:42 2024

@author: gliu
"""



import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
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
sys.path.append(cwd+"/..")
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


#%% Import Custom Modules

# Import AMV Calculation
from amv import proc,viz
import amv.loaders as dl

# Import stochastic model scripts
import scm

#%%

fpath   = input_path + 'forcing/'
ncE     = "CESM1_HTR_FULL_Eprime_EOF_nomasklag1_nroll0_NAtl_corrected_EnsAvg.nc"

ncLH    = "CESM1_HTR_FULL_LHFLX_EOF_nomasklag1_nroll0_NAtl_corrected_EnsAvg.nc"




ds_names     = [ncE,ncLH]
vnames       = ["LHFLX","LHFLX"]
vnames_title = ["E'","LHFLX"]
ds_vars      = [xr.open_dataset(fpath+dsn).load() for dsn in ds_names]


#%%

imode = 0
im    = 0

bboxplot = [-80,0,20,65]
proj = ccrs.PlateCarree()

fig,axs,_ = viz.init_orthomap(1,2,bboxplot,figsize=(12,4))

for a in range(2):
    
    ax       = axs[a]
    ax       = viz.add_coast_grid(ax,bbox=bboxplot,fill_color='lightgray')
    ax.set_title(vnames_title[a])
    dsplot   = ds_vars[a].isel(mode=imode,mon=im)[vnames[a]]
    pcm      = ax.pcolormesh(dsplot.lon,dsplot.lat,dsplot,transform=proj)
    cb = viz.hcbar(pcm,ax=ax)

plt.suptitle("E' and LHFLX, Month %i, Mode %i" % (im+1,imode+1))




#%%








#%% Set Paths

Eprime     = True # Set to True to Compute E' instead of F'

stormtrack = 0

# Path to variables processed by prep_data_byvariable_monthly, Output will be saved to rawpath1
if stormtrack:
    rawpath1 = "/stormtrack/data3/glliu/01_Data/02_AMV_Project/03_reemergence/proc/CESM1/NATL_proc/"
    dpath    = "/stormtrack/data3/glliu/01_Data/02_AMV_Project/03_reemergence/proc/model_input/damping/"
    mldpath  = "/stormtrack/data3/glliu/01_Data/02_AMV_Project/03_reemergence/proc/model_input/mld/"
else:
    rawpath1 = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/proc/CESM1/NATL_proc/"
    mldpath  = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/proc/model_input/mld/"
    dpath    = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/proc/model_input/damping/"
