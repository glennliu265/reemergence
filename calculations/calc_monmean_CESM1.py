#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Calc Monmean CESM1

- Computes monthly mean variables for CESM1
- Uses output processed by several scripts...
- Prepared for intake for viz_CESM1_HTR_meanstates script and amv.dl functions

Format:
    
    [ens x mon x lat x lon (degW)]
    

Created on Tue Jun 11 12:30:28 2024

@author: gliu
"""

import xarray as xr
import numpy as np
import matplotlib as mpl

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

#%% Reformatting script

def format_ds(ds):
    # Adapted from func. in high_pass_correlations.py
    if 'ensemble' in list(ds.dims):
        ds=ds.rename({'ensemble':'ens'})
        print("Renaming 'ensemble' to 'ens'")
    if 'month' in list(ds.dims):
        ds=ds.rename({'month':'mon'})
        print("Renaming 'month' to 'mon'")
    return ds


#%%

outpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/01_Data/CESM1_LE/proc/NATL/"

#%% Load SST

st     = time.time()

ncname = "CESM1LE_SST_NAtl_19200101_20050101_bilinear.nc"
ds_sst = xr.open_dataset(rawpath+ncname).SST.load()
ds_sst = proc.fix_febstart(ds_sst)

# Take Monthly Average
ds_sst = ds_sst.groupby('time.month').mean('time')
ds_sst = format_ds(ds_sst)

print("Formatted in %.2fs" % (time.time()-st))

outname = "%sCESM1_HTR_SST_NATL_scycle.nc" % outpath
edict   = dict(SST=dict(zlib=True))#proc.make_encoding_dict(ds_sst)
ds_sst.to_netcdf(outname,encoding=edict)

#%% Load SSS

st     = time.time()

vname  = "SSS"

ncname = "CESM1LE_%s_NAtl_19200101_20050101_bilinear.nc" % vname
ds_sst = xr.open_dataset(rawpath+ncname)[vname].load()
ds_sst = proc.fix_febstart(ds_sst)

# Take Monthly Average
ds_sst = ds_sst.groupby('time.month').mean('time')
ds_sst = format_ds(ds_sst)

print("Formatted in %.2fs" % (time.time()-st))

outname = "%sCESM1_HTR_%s_NATL_scycle.nc" % (outpath,vname)
edict   = {vname:dict(zlib=True)}#proc.make_encoding_dict(ds_sst)
ds_sst.to_netcdf(outname,encoding=edict)

#%%

