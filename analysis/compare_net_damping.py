#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 21:36:32 2024

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



#%%

dpath = input_path + "damping/"

vnames = ["SST","SSS"]
est    = []
for vv in range(2):
    ncname = "%sCESM1_HTR_FULL_Expfit_%s_damping_lagsfit123_EnsAvg.nc" % (dpath,vnames[vv])
    ds = xr.open_dataset(ncname).load().damping
    est.append(ds)
    
#%%


timescales = [1/ds for ds in est]
timescales_ann_mean = [ds.mean('mon') for ds in timescales]

damping_ann_mean  = [ds.mean('mon') for ds in est]

# SST vs SSS
timescale_ratio = timescales_ann_mean[1] / timescales_ann_mean[0]


damping_ratio   = damping_ann_mean[1] / damping_ann_mean[0]

#%% Plot Timescale

cints   = np.arange(0,12,0.5)
fig,ax  = plt.subplots(1,1,constrained_layout=True)
plotvar = timescale_ratio
cf      = ax.contourf(plotvar.lon,plotvar.lat,plotvar,levels=cints)

viz.hcbar(cf,ax=ax)

#%% Plot Damping

cints = np.arange(0,12,0.5)
cints = np.arange(0,1.1,0.1)

fig,ax = plt.subplots(1,1,constrained_layout=True)

plotvar = damping_ratio
cf = ax.contourf(plotvar.lon,plotvar.lat,plotvar,levels=cints)

cl = ax.contour(plotvar.lon,plotvar.lat,plotvar,levels=cints,colors="k")
ax.clabel(cl)

viz.hcbar(cf,ax=ax)

#%% 
