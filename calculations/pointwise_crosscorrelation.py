#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Compute pointwise crosscorrelation for 2 variables
For a given model output.

[Copied from pointwise_autocorrelation_lens.py]

Support separate calculation for positive and negative anomalies, based on the base variable.

Based on postprocess_autocorrelation.py

Created on Thu Mar 17 17:09:18 2022
@author: gliu
"""

import sys
import time
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from tqdm import tqdm

#%% User Edits


stormtrack      = False

# Autocorrelation parameters
# --------------------------
lags            = np.arange(0,61)
lagname         = "lag%02ito%02i" % (lags[0],lags[-1]) 
thresholds      = None#[-1,1] # Standard Deviations, Set to None if you don't want to apply thresholds
thresholds_name = "ALL" # Manually name this
conf            = 0.95
tails           = 2

# Dataset Parameters
# ---------------------------
outname_data = "CESM1_1920to2005_SSSACF"
vname_base   = "SSS"
vname_lag    = "SSS"
nc_base      = "CESM1LE_SSS_NAtl_19200101_20050101_bilinear.nc" # [ensemble x time x lat x lon 180]
nc_lag       = "CESM1LE_SSS_NAtl_19200101_20050101_bilinear.nc" # [ensemble x time x lat x lon 180]
datpath      = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/proc/CESM1/NATL_proc/"
preprocess   = True # If True, demean (remove ens mean) and deseason (remove monthly climatology)

# Output Information
# -----------------------------
outpath     = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/proc/"
figpath     = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/02_Figures/20230929/"

# Mask Loading Information
# ----------------------------
# Set to False to not apply a mask (otherwise specify path to mask)
loadmask    = False #"/stormtrack/data3/glliu/01_Data/02_AMV_Project/02_stochmod/Model_Data/model_input/limask180_FULL-HTR.npy"

# Load another variable to compare thresholds (might need to manually correct)
# ----------------------------------------------------------------------------
# CAUTION: This has not been updated from original script...
thresvar      = False #
thresvar_name = "HMXL"
thresvar_path = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/proc/thresholdvar/HMXL_FULL_PIC_lon-80to0_lat0to65_DTNone.nc"
if thresvar is True:
    print("WARNING NOT IMPLEMENTED. See Old Script...")
    # loadvar = xr.open_dataset(thresvar_path)
    # loadvar = loadvar[thresvar_name].values.squeeze() # [ensemble x time x lat x lon]
    
    # # Adjust dimensions to [lon x lat x time x (otherdims)]
    # loadvar = loadvar.transpose(2,1,0)#[...,None]

# Other Information
# ----------------------------
colors      = ['b','r','k']
bboxplot    = [-80,0,0,60]
bboxlim     = [-80,0,0,65]
debug       = False
saveens_sep = False

#%% Set Paths for Input (need to update to generalize for variable name)

if stormtrack:
    # Module Paths
    sys.path.append("/home/glliu/00_Scripts/01_Projects/00_Commons/")
    sys.path.append("/home/glliu/00_Scripts/01_Projects/01_AMV/02_stochmod/stochmod/model/")
else:
    # Module Paths
    sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/03_Scripts/")
    sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/03_Scripts/stochmod/model/")

# Import modules
from amv import proc,viz
import scm

# ----------------
#%% Load the data
# ----------------

# Uses output similar to preprocess_data_byens
# [ens x time x lat x lon]

st             = time.time()
    
# Load Variables
ds_base        = xr.open_dataset(datpath+nc_base).load()
if nc_base == nc_lag:
    ds_lag         = ds_base # Just reassign to speed things up
else:
    ds_lag         = xr.open_dataset(datpath+nc_lag).load()

# Make sure they are the same size
ncs_raw        = [ds_base,ds_lag]
ncs_resize     = proc.resize_ds(ncs_raw)
ds_base,ds_lag = ncs_resize

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
    ds_lag  = ds_lag * msk
    
# -----------------------------
#%% Preprocess, if option is set
# -----------------------------

def preprocess_ds(ds):
    # Remove mean seasonal cycle
    dsa = proc.xrdeseason(ds) # Remove the seasonal cycle
    dsa = dsa - dsa.mean('ensemble') # Remove the ensemble mean
    return dsa

if preprocess:
    st     = time.time()
    dsin   = [ds_base,ds_lag]
    dsanom = [preprocess_ds(ds) for ds in dsin]
    ds_base,ds_lag = dsanom
    print("Preprocessed data in %.2fs"% (time.time()-st))

# -------------------
#%% Prepare for input
# -------------------

def make_mask(var_in,sumdims=[0,]):
    vsum                  = np.sum(var_in,sumdims)
    vsum[~np.isnan(vsum)] = 1
    return vsum

# Indicate inputs
dsin                 = [ds_base,ds_lag]
vnames_in            = [vname_base,vname_lag] 

# Transpose and read out files  (Make into [lon x lat x ens x time])
varsin               = [dsin[vv][vnames_in[vv]].transpose('lon','lat','ensemble','time').values for vv in range(len(dsin))]

# Get Dimensions
nlon,nlat,nens,ntime = varsin[0].shape
npts                 = nlon*nlat
varsin               = [v.reshape(npts,nens,ntime) for v in varsin]

# Make sure they have consistent masks
vmasks               = [make_mask(vin,sumdims=(2,)) for vin in varsin]
maskfin              = np.prod(np.array(vmasks),0) # Make Product of masks (if nan in one, nan in all...) # [np.nan]
varsin               = [v * maskfin[:,:,None] for v in varsin]

# Get Threshold Informaton
if thresholds is None:
    print("No Threshold detected, doing calculations for all values...")
    nthres = 1
else:
    nthres               = len(thresholds) + 1  + 1 # less than, between, greater, and ALL
nlags                = len(lags)



#var_base,var_lag     = varsin


# Repeat for thresholding variable, if option is set
if thresvar is True:
    print("WARNING NOT IMPLEMENTED. See Old Script...")
    
    
# ----------------------
#%% Perform calculations
# ----------------------

"""
Inputs are:
    1) variable [ens x time x lat x lon]
    2) lon      [lon]
    3) lat      [lat]
    4) thresholds [Numeric] (Standard Deviations)
    5) savename [str] Full path to output file
    6) loadvar(optional) [lon x lat x time x otherdims] (thresholding variable)
    
"""

ds_all = []
for e in tqdm(range(nens)):
    
    
    
    # Remove NaN Points
    ensvars   = [invar[:,e,:] for invar in varsin] # npts x time
    nandicts  = [proc.find_nan(ensv,1,return_dict=True,verbose=False) for ensv in ensvars]
    validdata = [nd['cleaned_data'] for nd in nandicts] # [pts x yr x mon]
    
    nptsvalid = [vd.shape[0] for vd in validdata]
    if np.all([i == nptsvalid[0] for i in nptsvalid]):
        npts_valid = nptsvalid[0]
    else:
        print("WARNING, NaN points are not the same across variables. Aborting loop...")
        npts_valid = np.nan
        break
    
    # Split to year and month
    nyr        = int(ntime/12)
    validdata  = [vd.reshape(npts_valid,nyr,12) for vd in validdata]
    
    # Preallocate
    class_count = np.zeros((npts_valid,12,nthres)) # [pt x eventmonth x threshold]
    sst_acs     = np.zeros((npts_valid,12,nthres,nlags))  # [pt x eventmonth x threshold x lag]
    #sst_cfs     = np.zeros((npts_valid,12,nthres+2,nlags,2))  # [pt x eventmonth x threshold x lag x bounds]
    
    for im in range(12):
        
        # For that month, determine which years fall into which thresholds [pts,years]
        data_mon = [vd[:,:,im] for vd in validdata] # [pts x yr]
        
        if thresholds is None:
            data_mon_classes = np.zeros(data_mon[0].shape) # Dummy Variable
        else:
            if thresvar:
                print("WARNING NOT IMPLEMENTED. See Old Script...")
            else:
                data_mon_classes = proc.make_classes_nd(data_mon[0],thresholds,dim=1,debug=False) # Use the Base Variable
        
        for th in range(nthres): # Loop for each threshold
            
            if th < nthres - 1: # Calculate/Loop for all points 
                
                for pt in tqdm(range(npts_valid)): 
                    
                    # Get years which fulfill criteria
                    yr_mask          = np.where(data_mon_classes[pt,:] == th)[0] # Indices of valid years
                    if len(yr_mask) < 2:
                        print("Only 1 point found for pt=%i, th=%i" % (pt,th))
                        continue
                    
                    # Compute the lagcovariance (with detrending)
                    datain_base      = validdata[0][pt,:,:].T # transpose to [month x year]
                    datain_lag       = validdata[1][pt,:,:].T # transpose to [month x year]
                    
                    ac,yr_count      = proc.calc_lagcovar(datain_base,datain_lag,lags,im+1,0,yr_mask=yr_mask,debug=False)
                    #cf = proc.calc_conflag(ac,conf,tails,len(yr_mask)) # [lags, cf]
                    
                    # Save to larger variable
                    class_count[pt,im,th] = yr_count
                    sst_acs[pt,im,th,:]   = ac.copy()
                    #sst_cfs[pt,im,th,:,:]  = cf.copy()
                    # End Loop Point -----------------------------
            
            else: # Use all Data
                #print("Now computing for all data on loop %i"%th)
                # Reshape to [month x yr x npts]
                datain_base    = validdata[0].transpose(2,1,0)
                datain_lag     = validdata[1].transpose(2,1,0)
                acs            = proc.calc_lagcovar_nd(datain_base,datain_lag,lags,im+1,0) # [lag, npts]
                #cfs = proc.calc_conflag(acs,conf,tails,nyr) # [lag x conf x npts]
                
                # Save to larger variable
                sst_acs[:,im,th,:] = acs.T.copy()
                #sst_cfs[:,im,th,:,:]  = cfs.transpose(2,0,1).copy()
                class_count[:,im,th]   = nyr
            # End Loop Threshold -----------------------------
        # End Loop Event Month -----------------------------
    
    
    #% Now Replace into original matrices
    # Preallocate
    
    count_final = np.zeros((npts,12,nthres)) * np.nan
    acs_final   = np.zeros((npts,12,nthres,nlags)) * np.nan
    #cfs_final   = np.zeros((npts,12,nthres+2,nlags,2)) * np.nan
    
    
    # Replace
    okpts_var                  = nandicts[0]['ok_indices'] # Basevar
    count_final[okpts_var,...] = class_count
    acs_final[okpts_var,...]   = sst_acs
    #cfs_final[okpts,...]  = sst_cfs
    
    # Reshape
    count_final = count_final.reshape(nlon,nlat,12,nthres)
    acs_final   = acs_final.reshape(nlon,nlat,12,nthres,nlags)
    
    
    # Get Threshold Labels
    threslabs   = []
    if thresholds is None:
        threslabs.append("ALL")
    else:
        if nthres == 1:
            threslabs.append("$T'$ <= %i"% thresholds[0])
            threslabs.append("$T'$ > %i" % thresholds[0])
        elif nthres == 4: # Positive, Neutral, Negative, All
            for th in range(nthres-1):
                if th == 0:
                    tstr = "x < %s" % thresholds[th]
                elif th == 1:
                    tstr = "%s < x =< %s" % (thresholds[0],thresholds[1])
                elif th == 2:
                    tstr = "x > %s" % (thresholds[1])
                threslabs.append(tstr)
        else:
            threslabs = [th for th in range(nthres-1)]
        threslabs.append("ALL")
    
    
    # Make into Dataset
    coords_count = {'lon':lon,
                    'lat':lat,
                    'mons':np.arange(1,13,1),
                    'thres':threslabs}
    
    coords_acf  = {'lon'    :lon,
                    'lat'   :lat,
                    'mons'  :np.arange(1,13,1),
                    'thres' :threslabs,
                    'lags'  :lags}
    
    da_count   = xr.DataArray(count_final,coords=coords_count,dims=coords_count,name="class_count")
    da_acf     = xr.DataArray(acs_final,coords=coords_acf,dims=coords_acf,name="acf")
    ds_out     = xr.merge([da_count,da_acf])
    encodedict = proc.make_encoding_dict(ds_out)
    
    # Save Output
    if saveens_sep:
        savename = "%s%s_%s_%s_ens%02i.nc" % (outpath,outname_data,lagname,thresholds_name,e+1)
        ds_out.to_netcdf(savename,encoding=encodedict)
    ds_all.append(ds_out)

#%% Run this final point to merge this output

if saveens_sep:
    # Load everything again JTBS
    ds_all = []
    for e in range(nens):
        savename = "%s%s_%s_%s_ens%02i.nc" % (outpath,outname_data,lagname,thresholds_name,e+1)
        
        ds = xr.open_dataset(savename).load()
        ds_all.append(ds)
    
ds_all       = xr.concat(ds_all,dim='ens')
savename_out = "%s%s_%s_%s_ensALL.nc" % (outpath,outname_data,lagname,thresholds_name)
ds_all.to_netcdf(savename_out,encoding=encodedict)

print("Output saved to %s in %.2fs" % (savename_out,time.time()-st))
#%%

#%% Do the calculations

#print("Script ran in %.2fs!"%(time.time()-st))
#print("Output saved to %s."% (savename))