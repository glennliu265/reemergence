#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Compute Significance Pointwise
Copied upper section of pointwise crosscorrelation

Still in Progress, but the idea is

(1) Get Std and R(1) of timeseries
(2) Solve for Variance of White Noise
(3) Simulate [N] pairs of red noise timeseries
(4) Compute the significance thresholds...


Created on Tue Sep  3 07:57:17 2024

@author: gliu

"""

import sys
import time
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import tqdm
import glob

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


# # Dataset Parameters <CESM1 SST and SSS>
# # ---------------------------
outname_data = "CESM1_1920to2005_SST_SSS_crosscorrelation_nomasklag1_nroll0"
vname_base   = "SST"
vname_lag    = "SSS"
nc_base      = "CESM1LE_SST_NAtl_19200101_20050101_bilinear.nc" # [ensemble x time x lat x lon 180]
nc_lag       = "CESM1LE_SSS_NAtl_19200101_20050101_bilinear.nc" # [ensemble x time x lat x lon 180]
datpath      = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/proc/CESM1/NATL_proc/"
#datpath      = "/stormtrack/data3/glliu/01_Data/02_AMV_Project/03_reemergence/proc/CESM1/NATL_proc/"
preprocess   = True # If True, demean (remove ens mean) and deseason (remove monthly climatology)

# # # Dataset Parameters <Stochastic Model SST and SSS>
# # # ---------------------------
# outname_data = "SM_SST_SSS_PaperDraft02"
# vname_base   = "SST"
# vname_lag    = "SSS"
# nc_base      = "SST_Draft01_Rerun_QekCorr" # [ensemble x time x lat x lon 180]
# nc_lag       = "SSS_Draft01_Rerun_QekCorr" # [ensemble x time x lat x lon 180]
# datpath      = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/sm_experiments/"
# #datpath      = "/stormtrack/data3/glliu/01_Data/02_AMV_Project/03_reemergence/sm_experiments/"
# preprocess   = True # If True, demean (remove ens mean) and deseason (remove monthly climatology)
# hpf = False

# # Dataset Parameters <Stochastic Model SST and SSS (High Pass Filter)>
# # ---------------------------
outname_data = "SM_SST_SSS_PaperDraft02_hpf012mons"
vname_base   = "SST"
vname_lag    = "SSS"
nc_base      = "SST_Draft01_Rerun_QekCorr" # [ensemble x time x lat x lon 180]
nc_lag       = "SSS_Draft01_Rerun_QekCorr" # [ensemble x time x lat x lon 180]
datpath      = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/sm_experiments/"#"/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/proc/CESM1/NATL_proc/"
#datpath      = output_path#"/stormtrack/data3/glliu/01_Data/02_AMV_Project/03_reemergence/sm_experiments/"
preprocess   = True # If True, demean (remove ens mean) and deseason (remove monthly climatology)
hpf          = True

# # Dataset Parameters <CESM1 SST and SSS (High Pass Filter)>
# # ---------------------------
outname_data = "CESM1_1920to2005_SST_SSS_crosscorrelation_nomasklag1_nroll0_hpf012mons"
vname_base   = "SST"
vname_lag    = "SSS"
nc_base      = "CESM1LE_SST_NAtl_19200101_20050101_bilinear_hpf012mons.nc" # [ensemble x time x lat x lon 180]
nc_lag       = "CESM1LE_SSS_NAtl_19200101_20050101_bilinear_hpf012mons.nc" # [ensemble x time x lat x lon 180]
datpath      = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/proc/CESM1/NATL_proc/"
#datpath      = "/stormtrack/data3/glliu/01_Data/02_AMV_Project/03_reemergence/proc/CESM1/NATL_proc/"
preprocess   = True # If True, demean (remove ens mean) and deseason (remove monthly climatology)
hpf          = False


# # Dataset Parameters <Stochastic Model SST and SSS>, Draft03
# # ---------------------------
outname_data = "SM_SST_SSS_PaperDraft03"
vname_base   = "SST"
vname_lag    = "SSS"
nc_base      = "SST_Draft03_Rerun_QekCorr" # [ensemble x time x lat x lon 180]
nc_lag       = "SSS_Draft03_Rerun_QekCorr" # [ensemble x time x lat x lon 180]
#datpath      = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/proc/CESM1/NATL_proc/"
datpath      = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/sm_experiments/"#"/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/proc/CESM1/NATL_proc/"

preprocess   = True # If True, demean (remove ens mean) and deseason (remove monthly climatology)
hpf=False

# Output Information
# -----------------------------
outpath     = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/proc/"
#outpath      = "/stormtrack/data3/glliu/01_Data/02_AMV_Project/03_reemergence/proc/"
#figpath      = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/02_Figures/20230929/"

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


#%% Function to load stochastic model output

def load_smoutput(expname,output_path,debug=True,hpf=False):
    # Load NC Files
    expdir       = output_path + expname + "/Output/"
    if hpf:
        expdir = expdir + "hpf/"
    nclist       = glob.glob(expdir +"*.nc")
    nclist.sort()
    if debug:
        print(nclist)
        
    # Load DS, deseason and detrend to be sure
    ds_all   = xr.open_mfdataset(nclist,concat_dim="run",combine='nested').load()
    return ds_all

# ----------------
#%% Load the data
# ----------------

# Uses output similar to preprocess_data_byens
# [ens x time x lat x lon]

st             = time.time()
    
# Load Variables
if "sm_experiments" in datpath: # Load Stochastic model output
    print("Loading Stochastic Model Output")
    ds_base        = load_smoutput(nc_base,datpath,hpf=hpf)
    if nc_base == nc_lag:
        ds_lag         = ds_base # Just reassign to speed things up
    else:
        ds_lag         = load_smoutput(nc_lag,datpath)
    
    ds_base = ds_base.rename({'run':'ens'})
    ds_lag = ds_lag.rename({'run':'ens'})
else:
    ds_base        = xr.open_dataset(datpath+nc_base).load()
    if nc_base == nc_lag:
        ds_lag         = ds_base # Just reassign to speed things up
    else:
        ds_lag         = xr.open_dataset(datpath+nc_lag).load()

# Make sure they are the same size
ncs_raw        = [ds_base,ds_lag]
ncs_resize     = proc.resize_ds(ncs_raw)
ds_base,ds_lag = ncs_resize

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
    ds_lag  = ds_lag * msk
    
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

if preprocess:
    st     = time.time()
    dsin   = [ds_base[vname_base],ds_lag[vname_lag]]
    dsin   = [chk_dimnames(ds,longname=True) for ds in dsin]
    dsanom = [preprocess_ds(ds) for ds in dsin]
    
    ds_base,ds_lag = dsanom
    print("Preprocessed data in %.2fs"% (time.time()-st))
else:
    ds_base = ds_base[vname_base]
    ds_lag  = ds_lag[vname_lag]


#%% End Copy of pointwise_crosscorrelation

# Calc Needed Metrics

# Stdev
std_base = ds_base.std('time') # [Ens x Lat x Lon]
std_lag  = ds_lag.std('time')  # [Ens x Lat x Lon]

# R1 
def calc_r1(ts):
    if np.isnan(np.any(ts)):
        return np.nan
    else:
        return np.corrcoef(ts[:-1],ts[1:])[0,1]

r1_base = xr.apply_ufunc(
    calc_r1,
    ds_base,
    input_core_dims=[['time'],],
    output_core_dims=[[],],
    vectorize=True,
    )

st = time.time()
r1_lag = xr.apply_ufunc(
    calc_r1,
    ds_lag,
    input_core_dims=[['time'],],
    output_core_dims=[[],],
    vectorize=True,
    )
print("Computed R1 in %.2fs" % (time.time()-st))
    
# Reduce func doesn't work because you need an axis argument built in...
# r1_base = ds_base.reduce(func=calc_r1,axis=('lat','lon','ens'))

# For simplicity, just take ensemble mean of all the quantities
r1_in   = [r1_base.mean('ens'),r1_lag.mean('ens')]
std_in  = [std_base.mean('ens'),std_lag.mean('ens')]

# Compute variance of white noise
"""
From 12.S992 PSET 01 (or general derivation of variance from AR(1) model)
Assuming variance of the variable [x] is stationary, its variance is given by:
Var(x) = eta^2 / (1-r1^2) where eta is the white noise variance.
We can then solve for eta:
    eta = sqrt( Var(x) * (1-r1^2) )
"""

# Get the white noise amplitude
eta_in  = [np.sqrt(std_in[ii]**2 * (1-r1_in[ii]**2)) for ii in range(2)]

#%% Looping for each point...

# User selections
mciter      = 1000 # Number of iterations
use_monthly = True  # If Use Monthly = True, assume computation was for monthly ACF (dof = time/12 instead of time)
thres       = [.1,0.05,0.01] # Significance Levels
tails       = 2 # One sided or two sided

# Get Dimensions
ntime       = len(ds_base.time)
nlon        = len(ds_base.lon)
nlat        = len(ds_base.lat)
nthres      = len(thres)
if use_monthly:
    simlen = int(ntime/12)
else:
    simlen = ntime

# Preallocate and create arrays
if tails == 1:
    outthres    = np.zeros((nthres,nlat,nlon)) * np.nan
    coords = dict(thres=thres,lat=ds_base.lat,lon=ds_base.lon)
else: # Preallocate for two tails
    outthres    = np.zeros((2,nthres,nlat,nlon)) * np.nan
    coords = dict(tails=['lower','upper'],thres=thres,lat=ds_base.lat,lon=ds_base.lon)
    #coords['tails'] = ['lower','upper']

for o in tqdm.tqdm(range(nlon)):
    
    for a in range(nlat):
        
        # Get Parameters
        eta_pt = [ds.isel(lon=o,lat=a).item() for ds in eta_in]
        r1_pt  = [ds.isel(lon=o,lat=a).item() for ds in r1_in]
        
        # Skip point if either parameter is NaN
        if np.any(np.isnan(eta_pt)) or np.any(np.isnan(r1_pt)):
            continue
        
        # Compute correlations for red noise pairs
        corrs_rn = []
        for mc in range(mciter):
            rns     = [proc.make_ar1(r1_pt[ii],eta_pt[ii],simlen) for ii in range(2)]
            cc      = np.corrcoef(rns[0],rns[1])[0,1]
            corrs_rn.append(cc)
        corrs_rn = np.array(corrs_rn)
        
        # Get Percentiles
        if tails == 1:
            pcts = 1-np.array(thres)
        else:
            pcts = [np.array(thres)/2,1-np.array(thres)/2] # first 3 are lower, second 3 are upper
        thresholds_point = np.quantile(corrs_rn,pcts) # Interesting so this apparently takes lists and does not mess things up
        
        # Save to Output
        if tails == 1:
            outthres[:,a,o] = thresholds_point.copy()
        else:
            outthres[0,:,a,o] = thresholds_point[0,:].copy()
            outthres[1,:,a,o] = thresholds_point[1,:].copy()

# Make into dataarray
da_thres = xr.DataArray(outthres,coords=coords,dims=coords,name='thresholds')
savename = "%s%s_Significance_mciter%i_usemon%i_tails%i.nc" % (outpath,outname_data,mciter,use_monthly,tails)
edict    = proc.make_encoding_dict(da_thres)
da_thres.to_netcdf(savename,encoding=edict)
