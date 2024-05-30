#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Compute Surface Variance for SALT and TEMP
For normalizing vertical gradients.

Created on Fri May 24 14:50:54 2024

@author: gliu
"""

import xarray as xr
import numpy as np
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import cartopy.crs as ccrs

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
cwd = os.getcwd()
sys.path.append(cwd+ "/..")
import reemergence_params as rparams

# Paths and Load Modules
pathdict = rparams.machine_paths[machine]

sys.path.append(pathdict['amvpath'])
sys.path.append(pathdict['scmpath'])

# Set needed paths
figpath     = pathdict['figpath']
input_path  = pathdict['input_path']
output_path = pathdict['output_path']
procpath    = pathdict['procpath']
rawpath     = pathdict['raw_path']

#%% Import Custom Modules

# Import AMV Calculation
from amv import proc,viz
import amv.loaders as dl

# Import stochastic model scripts
import scm

#%% Set information and load (surface) variables

ncnames = ["CESM1LE_SST_NAtl_19200101_20050101_bilinear.nc",
           "CESM1LE_SSS_NAtl_19200101_20050101_bilinear.nc",
           "CESM1LE_TEMP_NAtl_19200101_20050101_NN.nc",]

vnames  = ["SST","SSS","TEMP"]

dsall   = [xr.open_dataset(rawpath+nc).load() for nc in ncnames]

nvars = len(vnames)

#dsall = []
for vv in range(nvars):
    ds = dsall[vv]
    if "ensemble" in list(ds.dims):
        dsall[vv] = ds.rename({'ensemble':'ens'})
    if "month" in list(ds.dims):
        dsall[vv] = ds.rename({'month':'mon'})
    
#%% Load, Deseason, Detrend


dsanom = [proc.xrdeseason(dsall[ii][vnames[ii]]) for ii in range(nvars)]
dsanom = [ds-ds.mean('ens') for ds in dsanom]


#%% Compute the variance

dsvars      = [ds.groupby('time.month').std('time') for ds in dsanom]
dsvars      = [ds.rename({'month':'mon'}) for ds in dsvars]

outnames    = [rawpath + proc.addstrtoext(nc,"_stdev",adjust=-1) for nc in ncnames]
edicts      = [{vn : dict(zlib=True)} for vn in vnames]

[dsvars[ii].to_netcdf(outnames[ii],encoding=edicts[ii]) for ii in range(3)]






