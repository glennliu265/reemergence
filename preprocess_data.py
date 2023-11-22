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

# Note: XR version is currently really slow, need to troubleshoot.

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
varname       = "HMXL" # "HMXL"
mconfig       = "FULL_HTR" # [FULL_PIC, SLAB_PIC, FULL_HTR]
method        = "bilinear" # regridding method
datpath       = "/stormtrack/data3/glliu/01_Data/02_AMV_Project/02_stochmod/%s/" % varname

# Set Bounding Box
bbox          = [-80,0,0,65] # Set Bounding Box
bboxfn        = "lon%ito%i_lat%ito%i" % (bbox[0],bbox[1],bbox[2],bbox[3])

# Preprocessing Option
detrend       = None # Type of detrend (see scipy.signal.detrend)
# Set None to not detrend the data!!

if "HTR" in mconfig:
    detrend = "EnsAvg"
    
# Output Name
savename       = "%s%s_%s_%s_DT%s.nc" % (datpath,varname,mconfig,bboxfn,detrend)

# Other toggles
use_xr         = False # Set to True to use xarray functions

#%% Main Script

# Open /Load dataset
# ------------------


st = time.time()
if "HTR" in mconfig: # Concatenate by ensemble
    
    #% Get the filenames
    # EX: SSS_FULL_HTR_bilinear_num00.nc
    globstr       = "%s%s_%s_%s_num*.nc" % (datpath,varname,mconfig,method)
    nclist        = glob.glob(globstr)
    nclist.sort()
    print("Found %i files!" % (len(nclist)))
    
    # Ensemble x Time x z x Lat x Lon
    ds_all = xr.open_mfdataset(nclist,concat_dim='ensemble',combine="nested",parallel=True)
    
else: # Just open it
    
    # Just open 1 file
    nc = "%s%s_%s_%s.nc" % (datpath,varname,mconfig,method)
    print("Opening %s" % (nc))

    # Time x z x Lat x Lon
    ds_all = xr.open_dataset(nc)

# Slice to region
# ---------------
ds_reg = ds_all.sel(lon=slice(bbox[0],bbox[1]),lat=slice(bbox[2],bbox[3]))


if use_xr: # Keep in Dataframe

    # Remove monthly anomalies
    # ------------------------
    dsa_reg = proc.xrdeseason(ds_reg)
    
    # Detrend Data
    # -----------------------
    if "HTR" in mconfig: # Remove ensemble average
        dsa_ensavg = dsa_reg.mean("ensemble")
        dsa_reg_dt = dsa_reg - dsa_ensavg
    else: # Detrend along time axis using specified method (NOT TESTED!!)
        dsa_reg_dt = scipy.signal.detrend(dsa_reg_dt,axis=0,method=detrend)
        
else: # Load to NumPy

    st_2 = time.time()
    invar = ds_reg[varname].values # [(ensemble) x time x lat x lon]
    lon   = ds_reg.lon.values
    lat   = ds_reg.lat.values
    times  = ds_reg.time.values
    print("Loaded data to np-arrays in %.2fs" % (time.time()-st_2))
    
    # Remove monthly anomalies
    # -------------------------
    if "HTR" in mconfig:
        time_axis = 1
    else:
        time_axis = 0
    climavg,tsyrmon = proc.calc_clim(invar,dim=time_axis,returnts=True)
    invar_anom      = tsyrmon - np.expand_dims(climavg,time_axis) # [yr x mon x lat x lon]
    
    # Detrend
    # -------
    if detrend is None: # Don't Detrend
        print("detrend set to None. No detrending will be performed!")
        invar_dt = invar_anom
    else:               # Detrend
        if "HTR" in mconfig: # Remove ensemble average
            invar_dt = invar_anom - np.mean(invar_anom,axis=0,keepdims=True)
        else: # Remove N-th order polynomial fit
            invar_dt,_,_,_ = proc.detrend_dim(invar_anom,time_axis)
    
    # Adjust dimensions [ens x yr x mon x z x lat x lon]
    # -----------------
    # Add z dimension
    if varname is "HMXL":
        # Add an extra "z" dimension
        invar_dt = np.expand_dims(invar_dt,time_axis+2) # [yr x mon x (z) x lat x lon]
        z_t = np.ones(1)
    else:
        z_t = ds_reg.z_t.values
    
    # Add ensemble dimension
    if "HTR" not in mconfig: # include additional ensemble axis at front...
        invar_dt = invar_dt[None,...] # [(ens) x yr x mon x (z) x lat x lon]
    
    # Make Coordinate dictionary
    nens,nyr,nmon,nz,nlat,nlon = invar_dt.shape
    newshape = (nens,nyr*nmon,nz,nlat,nlon)
    coords_dict = {"ensemble" : np.arange(1,nens+1,1),
                   "time" : times,
                   "z_t" : z_t,
                   "lat" : lat,
                   "lon" :lon
                   }
    
    # Reshape array to recombine monxyr to time
    invar_dt = invar_dt.reshape(newshape)
    
    # Place back in data array [ (ensemble) x time x lon x lat]
    dsa_reg_dt = xr.DataArray(invar_dt,coords=coords_dict,
                              dims=coords_dict,name=varname)
    
# Save output
# -----------
dsa_reg_dt.to_netcdf(savename,
                     encoding={varname: {'zlib': True}})
print("Saved output to %s in %.2fs" % (savename,time.time()-st))