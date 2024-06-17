

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Copied pointwise autocorrelation for SSS Basinwide Output

Created on Wed Feb  7 19:19:33 2024

@author: gliu

"""

import sys
import time
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import glob

#%% Select dataset to postprocess

# Set Machine
# -----------
stormtrack  = 1 # Set to True to run on stormtrack, False for local run

# Autocorrelation parameters
# --------------------------
lags        = np.arange(0,61)
lagname     = "lag%02ito%02i" % (lags[0],lags[-1]) 
thresholds  = [0,] # Standard Deviations
conf        = 0.95
tails       = 2

# For Stochastic Model Output indicate the experiment name
# -------------------------
expname    = "SST_EOF_LHFLX"
varname    = "SST" # ["TS","SSS","SST]
thresholds = None
if thresholds is None:
    thresname   = "thresALL"
else:
    thresname  = "thres" + "to".join(["%i" % i for i in thresholds])

if stormtrack:
    output_path = "/stormtrack/data3/glliu/01_Data/02_AMV_Project/03_reemergence/sm_experiments/"
else:
    output_path = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/sm_experiments/"
exppath    = output_path + expname + "/Output/"

# Set to False to not apply a mask (otherwise specify path to mask)
loadmask   = False #"/stormtrack/data3/glliu/01_Data/02_AMV_Project/02_stochmod/Model_Data/model_input/limask180_FULL-HTR.npy"
glonpath   = "/stormtrack/data3/glliu/01_Data/02_AMV_Project/02_stochmod/Model_Data/model_input/CESM1_lon180.npy"
glatpath   = "/stormtrack/data3/glliu/01_Data/02_AMV_Project/02_stochmod/Model_Data/model_input/CESM1_lat.npy"

# Load another variable to compare thresholds (might need to manually correct)
thresvar      = False #
thresvar_name = "HMXL"
if stormtrack:
    thresvar_path = "/stormtrack/data3/glliu/01_Data/02_AMV_Project/03_reemergence/proc/thresholdvar/HMXL_FULL_PIC_lon-80to0_lat0to65_DTNone.nc"
else:
    thresvar_path = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/proc/thresholdvar/HMXL_FULL_PIC_lon-80to0_lat0to65_DTNone.nc"

if thresvar is True:
    loadvar = xr.open_dataset(thresvar_path)
    loadvar = loadvar[thresvar_name].values.squeeze() # [time x lat x lon]
    
    # Adjust dimensions to [lon x lat x time x (otherdims)]
    loadvar = loadvar.transpose(2,1,0)#[...,None]

# Plotting Params
# ---------------
colors   = ['b','r','k']
bboxplot = [-80,0,0,60]
bboxlim  = [-80,0,0,65]

debug = False # Debug section below script (set to True to run)
#%% Set Paths for Input (need to update to generalize for variable name)

if stormtrack:
    # Module Paths
    sys.path.append("/home/glliu/00_Scripts/01_Projects/00_Commons/")
    sys.path.append("/home/glliu/00_Scripts/01_Projects/01_AMV/02_stochmod/stochmod/model/")
    
    # Output Paths
    figpath = "/stormtrack/data3/glliu/02_Figures/20220622/"
    outpath = "/stormtrack/data3/glliu/01_Data/02_AMV_Project/03_reemergence/proc/"
    
else:
    # Module Paths
    sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/03_Scripts/")
    sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/03_Scripts/stochmod/model/")
    
    # Output Paths
    figpath     = '/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/02_Figures/20220930/'
    outpath     = '/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/proc/'

cwd = os.getcwd()
sys.path.append(cwd+"/../")

# Import modules
from amv import proc,viz
import scm

#%% Set Output Directory
# --------------------
proc.makedir(figpath)
savename   = "%sSM_%s_%s_autocorrelation_%s_%s.nc" %  (outpath,expname,varname,thresname,lagname)
if thresvar is True:
    savename = proc.addstrtoext(savename,"_thresvar%s" % (thresvar_name))

print("Output will save to %s" % savename)

#%% Read in the data 
# ----------------------------
st = time.time()

# Load NC Files
expdir       = output_path + expname + "/Output/"
nclist       = glob.glob(expdir +"*.nc")
nclist.sort()
print(nclist)

# Load DS, deseason and detrend to be sure
ds_all   = xr.open_mfdataset(nclist,concat_dim="run",combine='nested').load()

ds_sm  = proc.xrdeseason(ds_all[varname])
ds_sm  = ds_sm - ds_sm.mean('run')
ds_sm  = ds_sm.rename(dict(run='ens'))

# Load Param Dictionary
dictpath   = output_path + expname + "/Input/expparams.npz"
expdict  = np.load(dictpath,allow_pickle=True)

# Move variable
sst = ds_sm.transpose('lon','lat','ens','time').values

# Merge Run with time
nlon,nlat,nens,ntime=sst.shape
sst = sst.reshape(nlon,nlat,nens*ntime)

# Load out Lat.Lon for Later
lon = ds_sm.lon.values
lat = ds_sm.lat.values

#%% Do the calculations
"""
Inputs are:
    1) variable [lon x lat x time x otherdims]
    2) lon      [lon]
    3) lat      [lat]
    4) thresholds [Numeric] (Standard Deviations)
    5) savename [str] Full path to output file
    6) loadvar(optional) [lon x lat x time x otherdims] (thresholding variable)
    
"""
# First things first, combine lat/lon/otherdims, remove nan points

# Get Dimensions
if len(sst.shape) > 3:
    
    print("%s has more than 3 dimensions. Combining." % varname)
    nlon,nlat,ntime,notherdims = sst.shape
    sst = sst.transpose(0,1,3,2) # [nlon,nlat,otherdims,time]
    npts = nlon*nlat*notherdims # combine ensemble and points
    
else:
    notherdims      = 0
    nlon,nlat,ntime = sst.shape
    npts            = nlon*nlat

nyr             = int(ntime/12)
nlags           = len(lags)
if thresholds is None:
    nthres = 1
else:
    nthres          = len(thresholds) + 2 # Above, Below, All

# Combine space, remove NaN points
sstrs                = sst.reshape(npts,ntime) 
if varname == "SSS":
    sstrs[:,219]     = 0 # There is something wrong with this timestep?
    
if thresvar: # Only analyze where both threshold variable and target var are non-NaN
    ntimeldvar    = loadvar.shape[2]
    loadvarrs     = loadvar.reshape(nlat*nlon,ntimeldvar)
    _,knan,okpts  = proc.find_nan(sstrs*loadvarrs,1) # [finepoints,time]
    sst_valid     = sstrs[okpts,:]
    loadvar_valid = loadvarrs[okpts,:]
    
else:
    sst_valid,knan,okpts = proc.find_nan(sstrs,1) # [finepoints,time]
npts_valid           = sst_valid.shape[0] 


# Split to Year x Month
sst_valid = sst_valid.reshape(npts_valid,nyr,12)
if thresvar: # Select non-NaN points for thresholding variable
    loadvar_valid = loadvar_valid.reshape(npts_valid,nyr,12)

# Preallocate (nthres + 1 (for all thresholds), and last is all data)
class_count = np.zeros((npts_valid,12,nthres)) # [pt x eventmonth x threshold]
sst_acs     = np.zeros((npts_valid,12,nthres,nlags))  # [pt x eventmonth x threshold x lag]
sst_cfs     = np.zeros((npts_valid,12,nthres,nlags,2))  # [pt x eventmonth x threshold x lag x bounds]


# A pretty ugly loop....
# Now loop for each month
for im in range(12):
    #print(im)
    
    # For that month, determine which years fall into which thresholds [pts,years]
    sst_mon = sst_valid[:,:,im] # [pts x yr]
    
    if thresholds is not None:
        
        if thresvar:
            loadvar_mon = loadvar_valid[:,:,im]
            sst_mon_classes = proc.make_classes_nd(loadvar_mon,thresholds,dim=1,debug=False)
        else:
            sst_mon_classes = proc.make_classes_nd(sst_mon,thresholds,dim=1,debug=False)
        
    for th in range(nthres): # Loop for each threshold
        
        if th > 0: 
            for pt in tqdm(range(npts_valid)): 
                
                # Get years which fulfill criteria
                yr_mask     = np.where(sst_mon_classes[pt,:] == (th-1))[0] # Indices of valid years
                
                # Compute the lagcovariance (with detrending)
                sst_in = sst_valid[pt,:,:].T # transpose to [month x year]
                ac,yr_count = proc.calc_lagcovar(sst_in,sst_in,lags,im+1,0,yr_mask=yr_mask,debug=False)
                cf = proc.calc_conflag(ac,conf,tails,len(yr_mask)) # [lags, cf]
                
                # Save to larger variable
                class_count[pt,im,th] = yr_count
                sst_acs[pt,im,th,:] = ac.copy()
                sst_cfs[pt,im,th,:,:]  = cf.copy()
                # End Loop Point -----------------------------
        
        else: # Use all Data
            print("Now computing for all data on loop %i"%th)
            # Reshape to [month x yr x npts]
            sst_in    = sst_valid.transpose(2,1,0)
            acs = proc.calc_lagcovar_nd(sst_in,sst_in,lags,im+1,1) # [lag, npts]
            cfs = proc.calc_conflag(acs,conf,tails,nyr) # [lag x conf x npts]
            
            # Save to larger variable
            sst_acs[:,im,th,:] = acs.T.copy()
            sst_cfs[:,im,th,:,:]  = cfs.transpose(2,0,1).copy()
            class_count[:,im,th]   = nyr
        # End Loop Threshold -----------------------------
    
    # End Loop Event Month -----------------------------

#% Now Replace into original matrices
# Preallocate
count_final = np.zeros((npts,12,nthres)) * np.nan
acs_final   = np.zeros((npts,12,nthres,nlags)) * np.nan
cfs_final   = np.zeros((npts,12,nthres,nlags,2)) * np.nan

# Replace
count_final[okpts,...] = class_count
acs_final[okpts,...]   = sst_acs
cfs_final[okpts,...]   = sst_cfs

# Reshape output
if notherdims == 0:
    count_final = count_final.reshape(nlon,nlat,12,nthres)
    acs_final   = acs_final.reshape(nlon,nlat,12,nthres,nlags)
    cfs_final   = cfs_final.reshape(nlon,nlat,12,nthres,nlags,2)
else:
    count_final = count_final.reshape(nlon,nlat,notherdims,12,nthres)
    acs_final   = acs_final.reshape(nlon,nlat,notherdims,12,nthres,nlags)
    cfs_final   = cfs_final.reshape(nlon,nlat,notherdims,12,nthres,nlags,2)

# Get Threshold Labels
threslabs   = []
if thresholds is None:
    threslabs.append("ALL")

else:
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
#savename = "%s%s_%s_ACF_%s_%s_ens%02i.nc" % (outpath,dataset,varname,yearstr,lagname,e+1)
ds_out.to_netcdf(savename,encoding=encodedict)


print("Script ran in %.2fs!"%(time.time()-st))
print("Output saved to %s."% (savename))

