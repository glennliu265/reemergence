#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Apply High Pass Filter to NATL Proc Data
(This is for the cross_correlation analysis)

Created on Tue Sep  3 10:48:53 2024

@author: gliu

"""

import xarray as xr
import numpy as np
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import cartopy.crs as ccrs
import scipy as sp

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
cwd             = os.getcwd()
sys.path.append(cwd + "/..")

# Paths and Load Modules
import reemergence_params as rparams
pathdict        = rparams.machine_paths[machine]

sys.path.append(pathdict['amvpath'])
sys.path.append(pathdict['scmpath'])

# Set needed paths
figpath         = pathdict['figpath']
input_path      = pathdict['input_path']
output_path     = pathdict['output_path']
procpath        = pathdict['procpath']
rawpath         = pathdict['raw_path']

# %% Import Custom Modules

from amv import proc, viz
import scm
import amv.xrfunc as xrf
import amv.loaders as dl

# Import stochastic model scripts
proc.makedir(figpath)

#%% Indicate Dataset to Filter

# # Dataset Parameters <CESM1 SST and SSS>
# # ---------------------------
nc_base      = "CESM1LE_SST_NAtl_19200101_20050101_bilinear.nc" # [ensemble x time x lat x lon 180]
vname_base   = "SST"
datpath      = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/proc/CESM1/NATL_proc/"
preprocess   = True # If True, demean (remove ens mean) and deseason (remove monthly climatology)
outpath      = datpath


# # Dataset Parameters <Stochastic Model SST and SSS>
# # ---------------------------
outname_data = "SM_SST_SSS_PaperDraft02"
vname_base   = "SSS"
nc_base      = "SSS_Draft01_Rerun_QekCorr" # [ensemble x time x lat x lon 180]
#datpath      = "/stormtrack/data3/glliu/01_Data/02_AMV_Project/03_reemergence/sm_experiments/"
datpath      = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/sm_experiments/"
preprocess   = True # If True, demean (remove ens mean) and deseason (remove monthly climatology)
outpath      = datpath

# Mask Loading Information
# ----------------------------
# Set to False to not apply a mask (otherwise specify path to mask)
loadmask    = False #"/stormtrack/data3/glliu/01_Data/02_AMV_Project/02_stochmod/Model_Data/model_input/limask180_FULL-HTR.npy"


#%% Function to load stochastic model output

def load_smoutput(expname,output_path,debug=True):
    # Load NC Files
    expdir       = output_path + expname + "/Output/"
    nclist       = glob.glob(expdir +"*.nc")
    nclist.sort()
    if debug:
        print(nclist)
        
    # Load DS, deseason and detrend to be sure
    ds_all   = xr.open_mfdataset(nclist,concat_dim="run",combine='nested').load()
    return ds_all

# ----------------
#%% Load the data (copied from pointwise_crosscorrelation)
# ----------------

# Uses output similar to preprocess_data_byens
# [ens x time x lat x lon]

st             = time.time()
    
# Load Variables
smflag = False
if "sm_experiments" in datpath: # Load Stochastic model output
    print("Loading Stochastic Model Output")
    ds_base        = load_smoutput(nc_base,datpath)
    ds_base = ds_base.rename({'run':'ens'})
    
    # Make a special outpath for hpf output
    smflag = True
    
    outpath = "%s%s/Output/hpf/" % (datpath,nc_base,) 
    proc.makedir(outpath)
    
else:
    ds_base        = xr.open_dataset(datpath+nc_base).load()
    outpath        = outpath

# Add Dummy Ensemble Dimension

# Get Lat/Lon
lon            = ds_base.lon.values
lat            = ds_base.lat.values
times          = ds_base.time.values
bbox_base      = proc.get_bbox(ds_base)
print("Loaded data in %.2fs"% (time.time()-st))

# --------------------------------
#%% Apply land/ice mask if needed
# --------------------------------
if loadmask:
    
    print("Applying mask loaded from %s!"%loadmask)
    
    # Load the mask
    msk  = xr.open_dataset(loadmask) # Lon x Lat (global)
    
    # Restrict to the same region
    dsin  = [ds_base,msk]
    dsout = proc.resize_ds(dsin) 
    _,msk = dsout
    
    # Apply to variables
    ds_base = ds_base * msk

# -----------------------------
#%% Preprocess, if option is set
# -----------------------------

def preprocess_ds(ds):
    
    if 'ensemble' in list(ds.dims):
        ds = ds.rename({'ensemble':'ens'})
    
    # Check for ensemble dimension
    lensflag=False
    if "ens" in list(ds.dims):
        lensflag=True
    
    # Remove mean seasonal cycle
    dsa = proc.xrdeseason(ds) # Remove the seasonal cycle
    if lensflag:
        print("Detrending by removing ensemble mean")
        dsa = dsa - dsa.mean('ens') # Remove the ensemble mean
        
    else: # Simple Linear Detrend, Pointwise
        print("Detrending by removing linear fit")
        dsa       = dsa.transpose('time','lat','lon')
        vname     = dsa.name
        
        # Simple Linear Detrend
        dt_dict   = proc.detrend_dim(dsa.values,0,return_dict=True)# ASSUME TIME in first axis
        
        # Put back into DataArray
        dsa = xr.DataArray(dt_dict['detrended_var'],dims=dsa.dims,coords=dsa.coords,name=vname)
        
    # Add dummy ensemble variable
    if lensflag is False:
        print("adding singleton ensemble dimension ")
        dsa  = dsa.expand_dims(dim={'ens':[1,]},axis=0) # Ensemble in first dimension
    
    return dsa

def chk_dimnames(ds,longname=False):
    if longname:
        if "ens" in ds.dims:
            ds = ds.rename({'ens':'ensemble'})
    else:
        if "ensemble" in ds.dims:
            ds = ds.rename({'ensemble':'ens'})
    return ds

try:
    ds_base = ds_base[varname]
except:
    print("[ds_base] is already a DataArray")    
    
if preprocess:
    st     = time.time()
    
    dsvar  = ds_base
    dsvar  = chk_dimnames(dsvar)
    dsanom = preprocess_ds(dsvar)
    ds_base = dsanom
    print("Preprocessed data in %.2fs"% (time.time()-st))


#%% Apply High Pass Filter (from viz_SST_SSS_coupling)

hicutoff  = 12 # In Months
cutoffstr = "hpf%03imons" % (hicutoff)
hipass    = lambda x: proc.lp_butter(x,hicutoff,6,btype='highpass')

st = time.time()
hpout = xr.apply_ufunc(
    hipass,
    ds_base,
    input_core_dims=[['time']],
    output_core_dims=[['time']],
    vectorize=True, 
    )
print("Applied High Pass Filter in %.2fs" % (time.time()-st))

#%% Save the Output


edict = proc.make_encoding_dict(hpout)


if smflag:
    st = time.time()
    # Save for each ensemble member
    nens    = len(hpout.ens)
    for e in range(nens):
        dsout   = hpout.isel(ens=e)
        outname = "%s%s_runidrun%02i.nc" % (outpath,vname_base,e)
        dsout.to_netcdf(outname,encoding=edict)
    print("Saved output in %.2fs\n\tSaved to %s" % (time.time()-st,outname))


else:
    st = time.time()
    outname = "%s%s" % (outpath,nc_base)
    outname = proc.addstrtoext(outname,"_%s" % cutoffstr,adjust=-1)
    
    
    hpout.to_netcdf(outname,encoding=edict)
    print("Saved output in %.2fs\n\tSaved to %s" % (time.time()-st,outname))

#%% Save the output
  







