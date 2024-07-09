#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Do a simple coarsen operation on a stochastic model input
(as an alternative to re-estimating everything)

Created on Mon Jul  8 17:20:24 2024

@author: gliu

"""

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import sys
import glob

import tqdm
import time

# ----------------------------------
#%% Import custom modules and paths
# ----------------------------------

# Indicate the Machine!
machine = "Astraeus"

# First Load the Parameter File
sys.path.append("../")
import reemergence_params as rparams

# Paths and Load Modules
pathdict   = rparams.machine_paths[machine]

sys.path.append(pathdict['amvpath'])
sys.path.append(pathdict['scmpath'])
from amv import proc
import scm

# Get needed paths
figpath     = pathdict['figpath']
proc.makedir(figpath)
input_path  = pathdict['input_path']
output_path = pathdict['output_path']
rawpath     = pathdict['raw_path']
rawpath_3d  = rawpath + "ocn_var_3d/"

# Set input parameter paths
mpath     = input_path + "mld/"
dpath     = input_path + "damping/"
fpath     = input_path + "forcing/"
maskpath  = input_path + "masks/" 

vnames      = ["SALT","TEMP"]

#%% Indicate the file name

# Load the Target Dataset
target_nc   = "CESM1_HTR_FULL_corr_d_SALT_detrendensmean_lagmax3_interp1_imshift1_dtdepth1_EnsAvg.nc"
varname     = "lbd_d"
target_path = dpath
dstarg      = xr.open_dataset(target_path + target_nc)[varname].load()
target_var  = dstarg.transpose('mon','lat','lon').data
latold      = dstarg.lat.data
lonold      = dstarg.lon.data


# Load and get the reference grid
ref_nc      = "cesm1le_htr_5degbilinear_Fprime_EOF_corrected_cesm1le5degqnet_nroll0_perc090_NAtl_EnsAvg.nc"
ref_path    = fpath
dsref       = xr.open_dataset(ref_path + ref_nc).load()

# Make it to the same bounding box
#dsref,dstarg = proc.resize_ds([dsref,dstarg])

lonnew      = dsref.lon.data
latnew      = dsref.lat.data

outnc       = dpath + "CESM1_HTR_FULL_corr_d_SALT_detrendensmean_lagmax3_interp1_imshift1_dtdepth1_EnsAvg_coarsen5deg.nc"

print("Coarsening: %s" % target_nc)
print("\tTarget Grid from \t: %s" % ref_nc)
print("\tOutput saved to \t\t: %s" % outnc)

#%% Start to regrid (using coarsen bavg)

help(proc.coarsen_byavg)

dx = (lonnew[1:] - lonnew[:-1])[0] # Use this
#dy = (latnew[1:] - latnew[:-1])[0]

# Coarsen
outvar,latn,lonn=proc.coarsen_byavg(target_var,latold,lonold,deg=5,tol=dx/2,newlatlon=[lonnew,latnew])

# Save output
coords  = dict(mon=dstarg.mon,lat=latn,lon=lonn)
daout   = xr.DataArray(outvar,coords=coords,dims=coords,name=varname)
edict   = proc.make_encoding_dict(dstarg)
daout.to_netcdf(outnc,encoding=edict)
print("Saved output to %s" % outnc)

#%% For debugging, compare the two

im = 11







