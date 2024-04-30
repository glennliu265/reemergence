#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Visualize output from run_SSS_pointmode_coupled

Created on Mon Apr 29 14:55:09 2024

@author: gliu
"""

import xarray as xr
import numpy as np
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
sys.path.append("../")
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

"""
SSS_EOF_Qek_Pilot

Note: The original run (2/14) had the incorrect Ekman Forcing and used ens01 detrainment damping with linear detrend
I reran this after fixing these issues (2/29)

"""

# Paths and Experiment
expname      = "SST_SSS_LHFLX" # Borrowed from "SST_EOF_LbddCorr_Rerun"
metrics_path = output_path + expname + "/Metrics/" 
exp_output   = output_path + expname + "/Output/" 

vnames       = ["SST","SSS"]

#%% Load the variables


# For some reason, 2 lat values are saved for SSS (50 and 50.42). 
# Need to fix this
ds_all = []
var_all = []
for vv in range(2):
    
    globstr = "%s%s_runid*.nc" % (exp_output,vnames[vv])
    nclist  = glob.glob(globstr)
    nclist.sort()
    ds      = xr.open_mfdataset(nclist,combine='nested',concat_dim="run").load()
    
    if len(ds.lat) > 1: # Related to SSS error...
        remake_ds = []
        for nr in range(len(ds.run)):
            invar = ds.isel(run=nr)[vnames[vv]]
            
            if np.all(np.isnan(invar.isel(lat=0))): 
                klat = 1
            if np.all(np.isnan(invar.isel(lat=1))):
                klat = 0
            print("Non-NaN Latitude Index was %i for run %i" % (klat,nr))
            invar = invar.isel(lat=klat)
            #invar['lat'] = 50.
            remake_ds.append(invar.values.copy())
        coords = dict(run=ds.run,time=ds.time)
        ds     = xr.DataArray(np.array(remake_ds).squeeze(),coords=coords,name=vnames[vv])
    else:
        ds = ds[vnames[vv]]
    
    #.sel(lat=50.42,method='nearest')
    ds_all.append(ds)
    var_all.append(ds.values.squeeze()) # [Run x Time]
    
var_flat = [v.flatten() for v in var_all]
   
#%% Chedk for NaNs

for vv in range(2):
    for rr in range(10):
        invar = var_all[vv][rr,:]
        if np.any(np.isnan(invar)):
            print("NaN Detected for v=%s, rr=%i" % (vnames[vv],rr))
    
#

#%% Compute the ACFs (auto and cross)


# Compute the Autocorrelation
lags     = np.arange(37)
acfs_all = [scm.calc_autocorr_mon(v,lags,verbose=False,return_da=False) for v in var_flat]#]scm.calc_autocorr(var_flat,lags,)

#%% Plot
xtks   = np.arange(0,37,1)
kmonth = 1
fig,ax = plt.subplots(1,1,constrained_layout=True,figsize=(12,4.5))
ax,_   = viz.init_acplot(kmonth,xtks,lags,ax=ax)

for vv in range(2):
    ax.plot(lags,acfs_all[vv][kmonth,:],label="%s" %  vnames[vv])  
ax.legend()

    










