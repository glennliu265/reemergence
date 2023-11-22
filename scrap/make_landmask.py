#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Compute pointwise autocorrelation for CESM or stochastic model outputs
Support separate calculation for warm and cold anomalies

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

#%% Select dataset to postprocess

# Set Machine
# -----------
stormtrack = True # Set to True to run on stormtrack, False for local run

# Autocorrelation parameters
# --------------------------
lags        = np.arange(0,37)
lagname     = "lag%02ito%02i" % (lags[0],lags[-1]) 
thresholds  = [0,] # Standard Deviations
conf        = 0.95
tails       = 2

mconfig    = "HTR-FULL" # #"PIC-FULL"

thresholds = [0,]
thresname  = "thres" + "to".join(["%i" % i for i in thresholds])
varname    = "SSS"



# Plotting Params
# ---------------
colors   = ['b','r','k']
bboxplot = [-80,0,0,60]

#%% Set Paths for Input (need to update to generalize for variable name)


if stormtrack:
    # Module Paths
    sys.path.append("/home/glliu/00_Scripts/01_Projects/00_Commons/")
    sys.path.append("/home/glliu/00_Scripts/01_Projects/01_AMV/02_stochmod/stochmod/model/")
    
    # Input Paths 
    if mconfig == "SM":
        datpath     = "/stormtrack/data3/glliu/01_Data/02_AMV_Project/02_stochmod/Model_Data/model_output/"
    else:
        datpath     = "/stormtrack/data3/glliu/01_Data/02_AMV_Project/02_stochmod/%s/" % varname
        
    # Output Paths
    figpath = "/stormtrack/data3/glliu/02_Figures/20220324/"
    outpath = "/stormtrack/data3/glliu/01_Data/02_AMV_Project/03_reemergence/proc/"
    
else:
    # Module Paths
    sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/03_Scripts/")
    sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/03_Scripts/stochmod/model/")

    # Input Paths 
    if mconfig == "SM":
        datpath     = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/01_Data/model_output/"
    elif "PIC" in mconfig:
        datpath    = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/01_Data/"
    elif "HTR" in mconfig:
        datpath    = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/01_Data/CESM_proc/"
    # Output Paths
    figpath     = '/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/02_Figures/20220325/'
    outpath     = '/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/proc/'
    
# Import modules
from amv import proc,viz
import scm

# Set Input Names
# ---------------
if mconfig == "SM": # Stochastic model
    # Postprocess Continuous SM  Run
    # ------------------------------
    print("WARNING! Not set up for stormtrack yet.")
    fnames      = ["forcingflxeof_090pct_SLAB-PIC_eofcorr2_Fprime_rolln0_1000yr_run2%02d_ampq0_method5_dmp0"%i for i in range(10)]
    mnames      = ["constant h","vary h","entraining"] 
elif "PIC" in mconfig:
    # Postproess Continuous CESM Run
    # ------------------------------
    print("WARNING! Not set up for stormtrack yet.")
    fnames     = ["CESM1_FULL_postprocessed_NAtl.nc","CESM1_SLAB_postprocessed_NAtl.nc"]
    mnames     = ["FULL","SLAB"] 
elif "HTR" in mconfig:
    # CESM1LE Historical Runs
    # ------------------------------
    fnames     = ["%s_FULL_HTR_lon-80to0_lat0to65_DTEnsAvg.nc" % varname,]
    mnames     = ["FULL",]

# Set Output Directory
# --------------------
proc.makedir(figpath)
savename   = "%sCESM1_%s_%s_autocorrelation_%s.npz" %  (outpath,mconfig,varname,thresname)
print("Output will save to %s" % savename)

#%% Read in the data (Need to update for variable name)
st = time.time()

if mconfig == "PIC_FULL":
    sst_fn = fnames[0]
elif mconfig == "PIC_SLAB":
    sst_fn = fnames[1]
else:
    sst_fn = fnames[0]
print("Processing: " + sst_fn)

if ("PIC" in mconfig) or ("SM" in mconfig):
    # Load in SST [model x lon x lat x time] Depending on the file format
    if 'npy' in sst_fn:
        print("Loading .npy")
        sst = np.load(datpath+sst_fn)
        # NOTE: Need to write lat/lon loader
    elif 'npz' in sst_fn:
        print("Loading .npz")
        ld  = np.load(datpath+sst_fn,allow_pickle=True)
        lon = ld['lon']
        lat = ld['lat']
        sst = ld['sst'] # [model x lon x lat x time]
    elif 'nc' in sst_fn:
        print("Loading netCDF")
        ds  = xr.open_dataset(datpath+sst_fn)
        
        ds  = ds.sel(lon=slice(-80,0),lat=slice(0,65))
        
        
        lon = ds.lon.values
        lat = ds.lat.values
        sst = ds[varname].values # [lon x lat x time]
        
elif "HTR" in mconfig:
    
    ds  = xr.open_dataset(datpath+fnames[0])
    ds  = ds.sel(lon=slice(-80,0),lat=slice(0,65))
    lon = ds.lon.values
    lat = ds.lat.values
    sst = ds[varname].values # [ENS x Time x Z x LAT x LON]
    sst = sst[:,840:,...].squeeze() # Select 1920 onwards
    sst = sst.transpose(3,2,1,0) # [LON x LAT x Time x ENS]

print("Loaded data in %.2fs"% (time.time()-st))
#%% Make the ice mask


sst[:,:,219,:] = 0

mask = sst.sum(2)
mask = ~np.isnan(mask)

maskfin = np.ones(mask.shape)
maskfin[~mask]=np.nan

