#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Check CESM1 5deg Output


Created on Wed Jul  3 13:54:39 2024

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


#%% Check Fprime

nc       = "cesm1le_htr_5degbilinear_Fprime_timeseries_cesm1le5degqnet_nroll0_NAtl.nc"
dsfprime = xr.open_dataset(rawpath+nc).Fprime
dsfp     = dsfprime.isel(time=323,ens=0)

#%% Check Damping

fig,ax   = plt.subplots(1,1,subplot_kw={'projection':ccrs.PlateCarree()})
ax.coastlines()
pcm      = ax.pcolormesh(dsfp.lon,dsfp.lat,dsfp)
fig.colorbar(pcm)
ax.set_title("Fprime at t=%s, ens %i" % (dsfp.time.data,dsfp.ens.data))






