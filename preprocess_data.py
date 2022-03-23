#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Preprocess data for re-emergence calculations 
- Runs on stormtrack server
- Currently works with regridded data by prep_MLD_PIC.py from "stochmod" module


Preprocessing Step
- Slice to region
- Take monthly anomalies
- Remove ensemble average (or detrend)


Created on Wed Mar 23 11:32:36 2022

@author: gliu
"""

import time
import numpy as np
import xarray as xr
import glob
import scipy
from tqdm import tqdm
import matplotlib.pyplot as plt
import sys

#%% User Edits

# Import module
sys.path.append("/home/glliu/00_Scripts/01_Projects/00_Commons/")
from amv import proc

# Data Information
varname       = "SSS" # "HMXL"
mconfig       = "FULL_HTR" # [FULL_PIC, SLAB_PIC, FULL_HTR]
method        = "bilinear" # regridding method
datpath       = "/stormtrack/data3/glliu/01_Data/02_AMV_Project/02_stochmod/%s/" % varname

# Set Bounding Box
bbox          = [-80,0,0,65] # Set Bounding Box
bboxfn        = "lon%ito%i_lat%ito%i" % (bbox[0],bbox[1],bbox[2],bbox[3])

# Preprocessing Option
detrend       = "linear" # Type of detrend (see scipy.signal.detrend)

if "HTR:" in mconfig:
    detrend = "EnsAvg"
    
# Output Name
savename       = "%s%s_%s_%s_DT%s.nc" % (datpath,varname,mconfig,bboxfn,detrend)


#%% Main Script

# Open /Load dataset
# ------------------

#% Get the filenames
# EX: SSS_FULL_HTR_bilinear_num00.nc
st = time.time()
globstr       = "%s%s_%s_%s_num*.nc" % (datpath,varname,mconfig,method)
nclist        = glob.glob(globstr)
nclist.sort()
print("Found %i files!" % (len(nclist)))

if "HTR" in mconfig: # Concatenate by ensemble
    # Ensemble x Time x z x Lat x Lon
    ds_all = xr.open_mfdataset(nclist,concat_dim='ensemble',combine="nested",parallel=True)
else: # Just open it
    # Time x z x Lat x Lon
    ds_all = xr.open_dataset(nclist)

# Slice to region
# ---------------
ds_reg = ds_all.sel(lon=slice(bbox[0],bbox[1]),lat=slice(bbox[2],bbox[3]))

# Remove monthly anomalies
# ------------------------
dsa_reg = proc.xrdeseason(ds_reg)

# Remove ensemble average
# -----------------------
if "HTR" in mconfig:
    dsa_ensavg = dsa_reg.mean("ensemble")
    dsa_reg_dt = dsa_reg - dsa_ensavg
else:
    # Detrend along time axis using specified method
    dsa_reg_dt = scipy.signal.detrend(dsa_reg_dt,axis=0,method=detrend)

# Save output
# -----------
dsa_reg_dt.to_netcdf(savename,
                     encoding={varname: {'zlib': True}})
print("Saved output to %s in %.2fs" % (savename,time.time()-st))