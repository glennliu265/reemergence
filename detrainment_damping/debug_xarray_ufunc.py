#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Debugging cases of xarray ufunc


Created on Thu Mar 28 13:45:12 2024

@author: gliu
"""

#%% ===========================================================================
#%% Setup

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import sys
from tqdm import tqdm
import scipy as sp
import time

# %% Import Custom Modules

amvpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/03_Scripts/"  # amv module
scmpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/03_Scripts/stochmod/model/"

sys.path.append(amvpath)
sys.path.append(scmpath)

from amv import proc, viz
import amv.loaders as dl
import scm

# %% Set data paths

# Select Point
lonf   = 330
latf   = 50
locfn, loctitle = proc.make_locstring(lonf, latf)

# Calculation Settings
lags   = np.arange(0,37,1)
lagmax = 3 # Number of lags to fit for exponential function 

# Indicate Paths
datpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/ptdata/lon%s_lat%s/" % (
    lonf, latf)
figpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/02_Figures/20240322/"
proc.makedir(figpath)
outpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/ptdata/%s/" % locfn

# Other toggles
debug = True # True to make debugging plots


# --------------------------------------------------------------------
#%% Load CESM1 variables (see repair_file_SALT_CESM1.py)
# --------------------------------------------------------------------

# Load SALT or TEMP at a point ----------------------------------------------------------------
# Paths and Names
vname = "TEMP"
if vname == "TEMP":
    ncname = "CESM1_htr_TEMP_repaired.nc"
    outvar = "T"
else:
    ncname  = "CESM1_htr_SALT_repaired.nc"
    outvar = "S"
ncsalt  = outpath + ncname
ds_salt = xr.open_dataset(ncsalt)

# Load
z       = ds_salt.z_t.values  # /100 NOTE cm --> meter conversion done in repair code
times   = ds_salt.time.values
salt    = ds_salt[vname].values  # [Ens x Time x Depth ]
nens, ntime, nz = salt.shape

if "repaired" not in ncname:
    print("Repairing File")
    # Repair File if needed
    # Set depths to zero
    salt_sumtime = salt.sum(1)[0,:]
    idnanz       = np.where(np.isnan(salt_sumtime))[0][0]
    salt = salt[:,:,:idnanz]
    z    = z[:idnanz] / 100
    nz    = len(z)
    
    for t in range(len(times)):
        for e in range(42):
            if np.all(np.isnan(salt[e,t,:])):
                print("ALL is NaN at t=%i, e =%i" % (t,e))
                salt[:,t,:] = 0

# Get strings for time
timesstr = ["%04i-%02i" % (t.year, t.month) for t in times]

# Get Ensemble Numbers
ens     = np.arange(nens)+1

# Load HBLT ----------------------------------------------------------------
# Paths and Names
mldname = "HMXL"
if mldname == "HBLT":
    
    mldpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/01_Data/CESM_proc/"
    mldnc   = "HBLT_FULL_HTR_lon-80to0_lat0to65_DTFalse.nc"
    
    # Load and select point
    dsh         = xr.open_dataset(mldpath+mldnc)
    hbltpt      = dsh.sel(lon=lonf-360, lat=latf,
                     method='nearest').load()  # [Ens x time x z_t]
    
    # Compute Mean Climatology [ens x mon]
    hclim       = hbltpt.groupby('time.month').mean('time').squeeze().HBLT.values/100  # Ens x month, convert cm --> m
    
    # Compute Detrainment month
    kprev, _    = scm.find_kprev(hclim.mean(1)) # Detrainment Months #[12,]
    hmax        = hclim.mean(1).max() # Maximum MLD of seasonal cycle # [1,]
elif mldname == "HMXL":
    mldpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/proc/model_input/mld/"
    mldnc   = "CESM1_HTR_FULL_HMXL_NAtl.nc"
    
    # Load and select point
    dsh       = xr.open_dataset(mldpath+mldnc)
    hbltpt      = dsh.sel(lon=lonf-360, lat=latf,
                     method='nearest').load()  # [Ens x time x z_t]
    
    # Compute Mean Climatology [ens x mon]
    hclim       = hbltpt.h.values
    
    # Compute Detrainment month
    kprev, _    = scm.find_kprev(hclim.mean(-1)) # Detrainment Months #[12,]
    hmax        = hclim.max()#hclim.mean(1).max() # Maximum MLD of seasonal cycle # [1,]

#%% Try Applying Unfunc to calculate detrianment ahead of time

mldin  = dsh.h.mean('ens')#.sel(lon=lonf-360,lat=latf,method='nearest')
infunc = lambda x: scm.find_kprev(x,debug=False,returnh=False)

st = time.time()
kprevall = xr.apply_ufunc(
    infunc, # Pass the function
    mldin, # The inputs in order that is expected
    input_core_dims =[['mon'],], # Which dimensions to operate over for each argument... 
    output_core_dims=[['mon'],], # Output Dimension
    vectorize=True, # True to loop over non-core dims
    )
print("Completed in %.2fs" % (time.time()-st))

kprevufunc = kprevall.sel(lon=lonf-360,lat=latf,method='nearest')#.mean('ens')
#kprev      = kprevufunc.values

diff       = kprev - kprevufunc.values

fig,ax = plt.subplots(1,1)
ax.plot(kprev,label="Ori")
ax.plot(kprevufunc,label="Ufunc")
ax.legend()

ax2 = ax.twinx()
ax.bar(np.arange(1,13,1),diff,alpha=0.2)

print(diff)

#%% Try applying detrainment depth computation