#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Compute pointwise autocorrelation for CMIP6 lens output

[Copied from pointwise_autocorrelation.py]
Support separate calculation for warm and cold anomalies

Based on postprocess_autocorrelation.py
Uses data preprocessed by reemergence/preprocess_data_byens.py

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

stormtrack = False

# Autocorrelation parameters
# --------------------------
lags        = np.arange(0,61)
lagname     = "lag%02ito%02i" % (lags[0],lags[-1]) 
thresholds  = [0,] # Standard Deviations
conf        = 0.95
tails       = 2

# Dataset Parameters
# ---------------------------
dataset     = "IPSL-CM6A-LR"
varname     = "SST"
yearstr     = "1850to2014"
datpath     = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/proc/CMIP6/"

# Output Information
# -----------------------------
outpath     = datpath + "proc/"
figpath     = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/02_Figures/20230929/"

# Mask Loading Information
# ----------------------------
# Set to False to not apply a mask (otherwise specify path to mask)
loadmask    = False #"/stormtrack/data3/glliu/01_Data/02_AMV_Project/02_stochmod/Model_Data/model_input/limask180_FULL-HTR.npy"
glonpath    = "/stormtrack/data3/glliu/01_Data/02_AMV_Project/02_stochmod/Model_Data/model_input/CESM1_lon180.npy"
glatpath    = "/stormtrack/data3/glliu/01_Data/02_AMV_Project/02_stochmod/Model_Data/model_input/CESM1_lat.npy"

# Load another variable to compare thresholds (might need to manually correct)
# ----------------------------------------------------------------------------
# CAUTION: This has not been updated from original script...
thresvar      = False #
thresvar_name = "HMXL"
thresvar_path = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/proc/thresholdvar/HMXL_FULL_PIC_lon-80to0_lat0to65_DTNone.nc"
if thresvar is True:
    print("WARNING NOT IMPLEMENTED. See Old Script...")
    loadvar = xr.open_dataset(thresvar_path)
    loadvar = loadvar[thresvar_name].values.squeeze() # [time x lat x lon]
    
    # Adjust dimensions to [lon x lat x time x (otherdims)]
    loadvar = loadvar.transpose(2,1,0)#[...,None]

# Other Information
# ----------------------------
colors   = ['b','r','k']
bboxplot = [-80,0,0,60]
bboxlim  = [-80,0,0,65]
debug    = False

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

#%% Load the data

st     = time.time()
# Uses output from preprocess_data_byens
# [ens x time x lat x lon]
# <datpath>/<dataset>_<varnane>_<yearstr>.nc
ncname = "%s%s_%s_%s.nc" % (datpath,dataset,varname,yearstr)
ds     = xr.open_dataset(ncname).load()
lon    = ds.lon.values
lat    = ds.lat.values
print("Loaded data in %.2fs"% (time.time()-st))

#%% Apply land/ice mask if needed
if loadmask:
    print("Applying mask loaded from %s!"%loadmask)
    # Load the mask
    msk  = np.load(loadmask) # Lon x Lat (global)
    glon = np.load(glonpath)
    glat = np.load(glatpath)
    
    # Restrict to Region
    bbox = [lon[0],lon[-1],lat[0],lat[-1]]
    rmsk,_,_ = proc.sel_region(msk,glon,glat,bbox)
    
    # Apply to variable # Assume ask is [lat x lon]
    ds = ds * rmsk[None,None,:,:]

#%% Perform calculations

"""
Inputs are:
    1) variable [ens x time x lat x lon]
    2) lon      [lon]
    3) lat      [lat]
    4) thresholds [Numeric] (Standard Deviations)
    5) savename [str] Full path to output file
    6) loadvar(optional) [lon x lat x time x otherdims] (thresholding variable)
    
"""


# Make into [lon x lat x ens x time]
da        = ds[varname].transpose('lon','lat','ens','time')
invar     = da.values
nlon,nlat,nens,ntime = invar.shape
npts      = nlon*nlat
invar     = invar.reshape(npts,nens,ntime)

# Repeat for thresholding variable, if option is set
if thresvar is True:
    print("WARNING NOT IMPLEMENTED. See Old Script...")

#for e in range(nens):
for e in range(nens):
    
    if e < 69:
        continue
    # Remove NaN Points
    ensvar     = invar[:,e,:] # [npts,ntime]
    nandict    = proc.find_nan(ensvar,1,return_dict=True,verbose=False)
    validdata  = nandict['cleaned_data']
    npts_valid = validdata.shape[0]
    
    # Split to year and month
    nyr        = int(ntime/12)
    validdata  = validdata.reshape(npts_valid,nyr,12)
    
    # Preallocate
    nthres      = len(thresholds)
    nlags       = len(lags)
    class_count = np.zeros((npts_valid,12,nthres+2)) # [pt x eventmonth x threshold]
    sst_acs     = np.zeros((npts_valid,12,nthres+2,nlags))  # [pt x eventmonth x threshold x lag]
    #sst_cfs     = np.zeros((npts_valid,12,nthres+2,nlags,2))  # [pt x eventmonth x threshold x lag x bounds]
    
    
    for im in range(12):
        
        # For that month, determine which years fall into which thresholds [pts,years]
        data_mon = validdata[:,:,im] # [pts x yr]
        
        if thresvar:
            print("WARNING NOT IMPLEMENTED. See Old Script...")
        else:
            data_mon_classes = proc.make_classes_nd(data_mon,thresholds,dim=1,debug=False)
        
        for th in range(nthres+2): # Loop for each threshold
            
            if th < nthres + 1: # Calculate/Loop for all points
                #print(th)
                for pt in tqdm(range(npts_valid)): 
                    
                    # Get years which fulfill criteria
                    yr_mask     = np.where(data_mon_classes[pt,:] == th)[0] # Indices of valid years
                    
                    # Compute the lagcovariance (with detrending)
                    datain = validdata[pt,:,:].T # transpose to [month x year]
                    ac,yr_count = proc.calc_lagcovar(datain,datain,lags,im+1,0,yr_mask=yr_mask,debug=False)
                    #cf = proc.calc_conflag(ac,conf,tails,len(yr_mask)) # [lags, cf]
                    
                    # Save to larger variable
                    class_count[pt,im,th] = yr_count
                    sst_acs[pt,im,th,:] = ac.copy()
                    #sst_cfs[pt,im,th,:,:]  = cf.copy()
                    # End Loop Point -----------------------------
            
            else: # Use all Data
                #print("Now computing for all data on loop %i"%th)
                # Reshape to [month x yr x npts]
                datain    = validdata.transpose(2,1,0)
                acs = proc.calc_lagcovar_nd(datain,datain,lags,im+1,1) # [lag, npts]
                #cfs = proc.calc_conflag(acs,conf,tails,nyr) # [lag x conf x npts]
                
                # Save to larger variable
                sst_acs[:,im,th,:] = acs.T.copy()
                #sst_cfs[:,im,th,:,:]  = cfs.transpose(2,0,1).copy()
                class_count[:,im,th]   = nyr
            # End Loop Threshold -----------------------------
        # End Loop Event Month -----------------------------
    
    
    #% Now Replace into original matrices
    # Preallocate
    count_final = np.zeros((npts,12,nthres+2)) * np.nan
    acs_final   = np.zeros((npts,12,nthres+2,nlags)) * np.nan
    #cfs_final   = np.zeros((npts,12,nthres+2,nlags,2)) * np.nan
    
    
    # Replace
    okpts                  = nandict['ok_indices']
    count_final[okpts,...] = class_count
    acs_final[okpts,...]   = sst_acs
    #cfs_final[okpts,...]  = sst_cfs
    
    # Reshape
    count_final = count_final.reshape(nlon,nlat,12,nthres+2)
    acs_final   = acs_final.reshape(nlon,nlat,12,nthres+2,nlags)
    
    
    # Get Threshold Labels
    threslabs   = []
    if nthres == 1:
        threslabs.append("$T'$ <= %i"% thresholds[0])
        threslabs.append("$T'$ > %i" % thresholds[0])
    else:
        for th in range(nthres):
            thval= thresholds[th]
            
            if thval != 0:
                sig = ""
            else:
                sig = "$\sigma$"
            
            if th == 0:
                tstr = "$T'$ <= %i %s" % (thval,sig)
            elif th == nthres:
                tstr = "$T'$ > %i %s" % (thval,sig)
            else:
                tstr = "%i < $T'$ =< %i %s" % (thresholds[th-1],thval,sig)
            threslabs.append(th)
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
    da_acf     = xr.DataArray(acs_final,coords=coords_acf,dims=coords_acf,name=varname)
    ds_out     = xr.merge([da_count,da_acf])
    encodedict = proc.make_encoding_dict(ds_out)
    
    # Save Output
    savename = "%s%s_%s_ACF_%s_%s_ens%02i.nc" % (outpath,dataset,varname,yearstr,lagname,e+1)
    ds_out.to_netcdf(savename,encoding=encodedict)


#%%

#%% Do the calculations

print("Script ran in %.2fs!"%(time.time()-st))
print("Output saved to %s."% (savename))


# #%% Debugging Corner

# if debug:
    
#     """
#     Section one, examine subsetting at a point...
#     """
#     nlon = len(lon)
#     nlat = len(lat)
#     kpt  = np.ravel_multi_index(np.array(([40],[53])),(nlon,nlat))
    
    
#     # Get Point Variable and reshape to yr x mon
#     sstpt  = sstrs[kpt,:]
#     mldpt  = loadvarrs[kpt,:]
#     sst_in = sstpt.reshape(int(sstpt.shape[1]/12),12) # []
#     mld_in = mldpt.reshape(sst_in.shape)
    
#     # Calculate autocorrelation (no mask)
#     acs    = proc.calc_lagcovar(sst_in.T,sst_in.T,lags,im+1,0,yr_mask=None,debug=False)
#     plt.plot(acs)
    
#     # Calculate autocorrelation with mask
#     loadvar_mon = loadvar_valid[:,:,im]
#     sst_mon     = sst_valid[:,:,im]
    
#     sst_class = proc.make_classes_nd(sst_mon,thresholds,dim=1,debug=False)[kpt,:]
#     mld_class = proc.make_classes_nd(loadvar_mon,thresholds,dim=1,debug=False)[kpt,:]
    
#     mask_sst     = np.where(sst_class.squeeze() == th)[0] # Indices of valid years
#     mask_mld     = np.where(mld_class.squeeze() == th)[0] # Indices of valid years
    
#     acs_sst,yr_count_sst = proc.calc_lagcovar(sst_in.T,sst_in.T,lags,im+1,0,yr_mask=mask_sst,debug=False)
#     acs_mld,yr_count_mld = proc.calc_lagcovar(sst_in.T,sst_in.T,lags,im+1,0,yr_mask=mask_mld,debug=False)
    
#     fig,ax=plt.subplots(1,1)
#     ax.plot(lags,acs_sst,label="SST Threshold, count=%i" % yr_count_sst,color='k')
#     ax.plot(lags,acs_mld,label="MLD Threshold, count=%i" % yr_count_mld,color='b')
#     ax.legend()
    
#     fig,ax = plt.subplots(1,1)
#     ax.plot(sst_in_actual[0,:],label="actual SST")
#     ax.plot(sst_in[:,0])
#     ax.legend()