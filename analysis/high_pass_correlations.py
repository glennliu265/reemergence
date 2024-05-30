#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""


Examine how applying high pass filters impacts the correlation of SST and SSS

Created on Wed May 29 15:11:59 2024

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
machine = "stormtrack"

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


#%% Load SST and SSS

vnames  = ["TEMP","SSS"]
ncnames = ["CESM1LE_TEMP_NAtl_19200101_20050101_NN.nc","CESM1LE_SSS_NAtl_19200101_20050101_bilinear.nc"]

ds_load = [xr.open_dataset(rawpath+ncnames[ii])[vnames[ii]].load() for ii in range(2)]


#%% Detrend and deseason
def preproc_ds(ds):
    if 'ensemble' in list(ds.dims):
        ds=ds.rename({'ensemble':'ens'})
    dsa = proc.xrdeseason(ds)#ds - ds.groupby('time.month').mean('time')
    dsa = dsa - dsa.mean('ens')
    return dsa


ds_anom = [preproc_ds(ds) for ds in ds_load]

#%% Design and apply pointwise high pass filters (all months, loop ver)
# Based on script in viz_SST_SSS_coupling

hicutoffs  = [3,6,9,12,15,18,24] # In Months
nthres     = len(hicutoffs)


#%% Single cutoff

hicutoff = 3

hipass    = lambda x: proc.lp_butter(x,hicutoff,6,btype='highpass') # Make Function
cesm_hipass = []
for vv in tqdm.tqdm(range(2)):
    hpout = xr.apply_ufunc(
        hipass,
        ds_anom[vv],
        input_core_dims=[['time']],
        output_core_dims=[['time']],
        vectorize=True, 
        )
    cesm_hipass.append(hpout)
# Takes 15 minutes per iteration ()

# Save outut
for vv in range(2):
    hipass_out = cesm_hipass[vv]
    outname    = rawpath + "/filtered/" + proc.addstrtoext(ncnames[vv],"_hpf%02imon" % hicutoff,adjust=-1)
    hipass_out.to_netcdf(outname,encoding={vnames[vv]:{'zlib':True}})


cesm_hipass[1]['ens'] = np.arange(1,43,1)

# Resize so they are the same size
cesm_rsz = proc.resize_ds(cesm_hipass)

# Get the cross correlation
crosscorr = lambda x,y: np.corrcoef(x,y)[0,1]
ccout = xr.apply_ufunc(
    crosscorr,
    cesm_rsz[0],
    cesm_rsz[1],
    input_core_dims=[['time'],['time']],
    output_core_dims=[[]],
    vectorize=True, 
    )



# Save output
outname    = rawpath + "/filtered/" + "CESM1_HTR_SST_SSS_NATL_crosscorr_hpf%02imon.nc" % hicutoff
ccout      =  ccout.rename("corr")
ccout.to_netcdf(outname,encoding={'corr':{'zlib':True}})

# Debug
lonf = -30
latf = 50

ccpt = proc.selpt_ds(ccout,lonf,latf)

varspt = [proc.selpt_ds(ds,lonf,latf) for ds in cesm_hipass]

testcc =  np.corrcoef(varspt[0].isel(month=1,ens=1).values,varspt[1].isel(month=1,ens=1).values)[0,1]

#%% loop ver)
hpvars = []
hpcorr = []
for th in range(nthres):
    
    # Apply High Pass Filter to each variable
    hicutoff = hicutoffs[th]
    hipass    = lambda x: proc.lp_butter(x,hicutoff,6,btype='highpass') # Make Function
    cesm_hipass = []
    for vv in tqdm.tqdm(range(2)):
        hpout = xr.apply_ufunc(
            hipass,
            ds_anom[vv],
            input_core_dims=[['time']],
            output_core_dims=[['time']],
            vectorize=True, 
            )
        cesm_hipass.append(hpout)
    #hpvars.append(cesm_hipass)
    # Takes 15 minutes per iteration ()
    

    
    
    cesm_hipass[1]['ens'] = np.arange(1,43,1)

    # Resize so they are the same size
    cesm_rsz = proc.resize_ds(cesm_hipass)
    
    # Save outut
    for vv in range(2):
        hipass_out = cesm_hipass[vv]
        outname    = rawpath + "/filtered/" + proc.addstrtoext(ncnames[vv],"_hpf%02imon" % hicutoff,adjust=-1)
        hipass_out.to_netcdf(outname,encoding={vnames[vv]:{'zlib':True}})

    # Get the cross correlation --------------------
    crosscorr = lambda x,y: np.corrcoef(x,y)[0,1]
    ccout = xr.apply_ufunc(
        crosscorr,
        cesm_rsz[0],
        cesm_rsz[1],
        input_core_dims=[['time'],['time']],
        output_core_dims=[[]],
        vectorize=True, 
        )
    
    # Save output
    outname    = rawpath + "/filtered/" + "CESM1_HTR_SST_SSS_NATL_crosscorr_hpf%02imon.nc" % hicutoff
    ccout      =  ccout.rename("corr")
    ccout.to_netcdf(outname,encoding={'corr':{'zlib':True}})
    
    #cesm_cc.append(ccout)
    #hpcorr.append(cesm_cc)
    
    











