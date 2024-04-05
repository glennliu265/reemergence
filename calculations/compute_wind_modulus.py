#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Compute the wind modulus for U and V CESM1 LENs
as processed by [prep_data_byvariable_monthly.py]

Copied upper section from the same script

Created on Tue Apr  2 09:46:55 2024

@author: gliu
"""

import numpy as np
import xarray as xr
import glob
import time
from tqdm import tqdm
import sys

# -----------------------------------------------------------------------------
#%% User Edits
# -----------------------------------------------------------------------------

stall         = time.time()
machine       = "stormtrack"

# Dataset Information
varnames      = ["U","V"]
outvar        = "Umod"

mconfig       = "FULL_HTR"
method        = "bilinear" # regridding method for POP ocean data

# Processing Options
regrid        = None  # Set to desired resolution. Set None for no regridding.
regrid_step   = True  # Set to true if regrid indicates the stepsize rather than total dimension size..
save_concat   = True  # Set to true to save the concatenated dataset (!! before annual anomaly calculation)
save_ensavg   = False # Set to true to save ensemble average (!! before annual anomaly calculation)
load_concat   = False # Set to true to load concatenated data

# Cropping Options
bbox          = None # Crop Selection, defaults to value indicated in predict_amv_params
ystart        = 1920 # Start year
yend          = 2005 # End year

# Data Path
datpath_manual = None # Manually set datpath
outpath        = "/stormtrack/data3/glliu/01_Data/02_AMV_Project/03_reemergence/proc/CESM1/NATL_proc/"

#outpath        = "../../CESM_data/Predictors/"

# Other Toggles
debug         = True # Set to True for debugging flat

# -----------------------------------------------------------------------------
#%% Import Packages + Paths based on machine
# -----------------------------------------------------------------------------

# Get Project parameters
sys.path.append("../")
import predict_amv_params as pparams

# Get paths based on machine
machine_paths = pparams.machine_paths[machine]

# Import custom modules
sys.path.append(machine_paths['amv_path'])
from amv import loaders,proc

# Get experiment bounding box for preprocessing
if bbox is None:
    bbox  = pparams.bbox_crop
nvars = len(varnames)


#%% Load U and V data processed by other script

st = time.time()
ds_all = []
for varname in varnames:
    ncname = "%sCESM1LE_%s_NAtl_%s0101_%s0101_%s.nc" % (outpath,varname,ystart,yend,method)
    ds = xr.open_dataset(ncname)[varname].load()
    ds_all.append(ds)
print("Data loaded in %.2fs" % (time.time()-st))

#%% Compute the modulus

dsmod = np.sqrt(ds_all[0]**2 + ds_all[1]**2)
dsmod = dsmod.rename(outvar)


#%% Do a quick check

lonf    = -30
latf    = 50
e       = 0
t       = 33
ds_chk  = [dsmod,ds_all[0],ds_all[1]]
dspt    = [proc.selpt_ds(ds,lonf,latf).isel(time=t,ensemble=e).values.item() for ds in ds_chk]


print("Check {:.2f} == {:.2f} = sqrt( {:.2f}^2 + {:.2f}^2 )".format(dspt[0],
                                                                    np.sqrt(dspt[1]**2+dspt[2]**2),
                                                                    dspt[1],
                                                                    dspt[2]))



#%% Save output

outname = "%sCESM1LE_%s_NAtl_%s0101_%s0101_%s.nc" % (outpath,outvar,ystart,yend,method)
edict   = {outvar : {'zlib':True}}
dsmod.to_netcdf(outname,encoding=edict)

