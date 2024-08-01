#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Calculate mean seasonal cycle for data in ocn_var_3d

Created on Wed May 22 18:51:25 2024

@author: gliu
"""


import xarray as xr
import numpy as np
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import cartopy.crs as ccrs
from scipy import signal

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
machine = "stormtrack"

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

#%% Indicate variables and path

vnames  = ["UVEL","VVEL"]
outpath = rawpath + "ocn_var_3d/"

nens    = 42

#%%

dsens = []
for e in tqdm.tqdm(range(nens)):
    
    ncname = "%s/%s/%s_NATL_ens%02i.nc" % (outpath,vnames[vv],vnames[vv],e+1)
    # if e == 32:
    #     ncname = proc.addstrtoext(ncname,"_repaired",adjust=-1)
    ds = xr.open_dataset(ncname).load()[vnames[vv]]
    dssavg = ds.groupby('time.month').mean('time')
    dsens.append(dssavg.copy())

dsens   = xr.concat(dsens,dim='ens')

outname = "%s/%s/%s_NATL_ensALL_scycle.nc" % (outpath,vnames[vv],vnames[vv])
edict   = {vnames[vv]:{'zlib':True}}
dsens.to_netcdf(outname,encoding=edict)
