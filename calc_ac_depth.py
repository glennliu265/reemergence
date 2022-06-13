#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Get UOHC Data for a single point


Created on Fri Jun 10 13:35:47 2022

@author: gliu
"""

import sys
import time
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from tqdm import tqdm

import glob

#%% Select dataset to postprocess

# Set Machine
# -----------
stormtrack = 1 # Set to True to run on stormtrack, False for local run

# Data Preprocessing
# ------------------
startyr     = 1920
endyr       = 2006

lonf        = -30+360
latf        = 50

# Autocorrelation parameters
# --------------------------
lags        = np.arange(0,61)
lagname     = "lag%02ito%02i" % (lags[0],lags[-1]) 
thresholds  = [0,] # Standard Deviations
conf        = 0.95
tails       = 2

mconfig    = "CESM" #"HadISST" #["PIC-FULL","HTR-FULL","PIC_SLAB","HadISST","ERSST"]
thresholds = [0,]
thresname  = "thres" + "to".join(["%i" % i for i in thresholds])
varname    = "TEMP" # ["TS","SSS","SST]


# MLD DAta
outpath = "/stormtrack/data3/glliu/01_Data/02_AMV_Project/03_reemergence/proc/CESM1/"
mldname = "CESM1_PiC_HMXL_Clim_Stdev.nc" # Made with viz_mldvar.py (stochmod/analysis)


# Set to False to not apply a mask (otherwise specify path to mask)
loadmask   = False #"/stormtrack/data3/glliu/01_Data/02_AMV_Project/02_stochmod/Model_Data/model_input/limask180_FULL-HTR.npy"
glonpath   = "/stormtrack/data3/glliu/01_Data/02_AMV_Project/02_stochmod/Model_Data/model_input/CESM1_lon180.npy"
glatpath   = "/stormtrack/data3/glliu/01_Data/02_AMV_Project/02_stochmod/Model_Data/model_input/CESM1_lat.npy"

# Plotting Params
# ---------------
colors   = ['b','r','k']
bboxplot = [-80,0,0,60]
bboxlim  = [-80,0,0,65]
#%% Set Paths for Input (need to update to generalize for variable name)

if stormtrack:
    # Module Paths
    sys.path.append("/home/glliu/00_Scripts/01_Projects/00_Commons/")
    sys.path.append("/home/glliu/00_Scripts/01_Projects/01_AMV/02_stochmod/stochmod/model/")
    
    # Input Paths 
    datpath = "/stormtrack/data4/share/deep_learning/data_yuchiaol/cesm_le/TEMP/"
    
    # Output Paths
    figpath = "/stormtrack/data3/glliu/02_Figures/20220622/"
    outpath = "/stormtrack/data3/glliu/01_Data/02_AMV_Project/03_reemergence/proc/"
    
else:
    # Module Paths
    sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/03_Scripts/")
    sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/03_Scripts/stochmod/model/")

    # Input Paths 
    datpath = ""
    
    # Output Paths
    figpath     = '/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/02_Figures/20220325/'
    outpath     = '/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/proc/'
    
# Import modules
from amv import proc,viz
import scm


def find_tlatlon(ds,lonf,latf):

    # Get minimum index of flattened array
    kmin      = np.argmin( (np.abs(ds.TLONG-lonf) + np.abs(ds.TLAT-latf)).values)
    klat,klon = np.unravel_index(kmin,ds.TLAT.shape)
    
    # Print found coordinates
    foundlat = ds.TLAT.isel(nlat=klat,nlon=klon).values
    foundlon = ds.TLONG.isel(nlat=klat,nlon=klon).values
    print("Closest lon to %.2f was %.2f" % (lonf,foundlon))
    print("Closest lat to %.2f was %.2f" % (latf,foundlat))
    
    return ds.isel(nlon=klon,nlat=klat)
    
# -----------------------
#%% Get MLD Data
# -----------------------

# Get Mean Climatological Cycle
# Get Stdev

dsmld = xr.open_dataset(outpath+mldname)

if lonf > 180:
    lonfw = lonf-360
else:
    lonfw = lonf
    
# Get Mean and Stdev, take maximum of that
hbar     = dsmld.clim_mean.sel(lon=lonfw,lat=latf,method='nearest').max().values
mmax     = dsmld.clim_mean.sel(lon=lonfw,lat=latf,method='nearest').values.argmax()
hstd     = dsmld.stdev.sel(lon=lonfw,lat=latf,method='nearest').isel(month=mmax).values

# Convert to CM
hmax_sel =(hbar+hstd)*100

# -----------------------
#%% Load Data
# -----------------------
# Get the list of files
ncsearch = "b.e11.B20TRC5CNBDRD.f09_g16.*.pop.h.TEMP.*.nc"
nclist   = glob.glob(datpath+ncsearch)
nclist   = [nc for nc in nclist if "OIC" not in nc]
nclist.sort()
nens     = len(nclist)
print("Found %i files!"%nens)


# Open and Slice
dsall = [] # This is just a remnant, remove it at some point...
for n in tqdm(range(nens)):
    
    # Open Dataset
    nc = nclist[n]
    ds = xr.open_dataset(nc)
    
    # Drop unwanted variables
    varkeep = [varname,"TLONG","TLAT","z_t","time"]
    dsvars  = list(ds.variables)
    remvar  = [i for i in dsvars if i not in varkeep]
    ds      = ds.drop(remvar)
    
    # Slice to Time
    ds      = ds.sel(time=slice("%s-02-01"%(startyr),"%s-01-01"%(endyr+1)))
    
    # Select a Point
    ds      = find_tlatlon(ds,lonf,latf)
    
    # Select a Depth
    ds      = ds.sel(z_t=slice(0,hmax_sel))
    
    # Save into an array
    if n == 0:
        v_all = np.zeros((nens,)+ds.TEMP.shape) * np.nan # [ens x time x depth]
        z_t   = ds.z_t.values
        times = ds.time.values
    v_all[n,:,:] = ds.TEMP.values
    
    # Append and move on...
    dsall.append(ds)


tlon = ds.TLONG.values
tlat = ds.TLAT.values

# Resave the variable as an array
dims = {"ensemble":np.arange(1,43,1),
          "time":times,
          "z_t":z_t
          }

attr_dict = {'lon':tlon,
             'lat':tlat,
             'hbar':hbar,
             'hstd':hstd,
             'hmax':hmax_sel,
             }

da = xr.DataArray(v_all,
    dims=dims,
    coords=dims,
    name = varname,
    attrs=attr_dict
    )

savename = "%sCESM1LE_UOTEMP_lon%i_lat%i.nc" % (outpath,tlon,tlat)
#% Save as netCDF
# ---------------
st = time.time()
encoding_dict = {varname : {'zlib': True}} 
print("Saving as " + savename)exi
da.to_netcdf(savename,
         encoding=encoding_dict)
print("Saved in %.2fs" % (time.time()-st))


#%% Read each file in

# Restrict to Point
# Cut to time period
# Cut to depth


# Remove Seasonal Cycle (separately for each ensemble member?)
# Calculate Ens Avg (save it, then remove)


#%%

#

