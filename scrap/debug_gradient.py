#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Compare Gradient Functions (Scrap from visualize_dz)

Created on Tue May 21 08:19:13 2024


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

#%% Work with data postprocessed by [viz_icefrac.py]

# Load the data
outpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/ptdata/profile_analysis/"
ncname  = "IrmingerAllEns_SALT_TEMP.nc"
ds      = xr.open_dataset(outpath+ncname).load()

# Load the mixed layer depth
mldpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/proc/model_input/mld/"
mldnc   = "CESM1_HTR_FULL_HMXL_NAtl.nc"
ds_mld  = xr.open_dataset(mldpath + mldnc).load()

# Load detrainment depth computed from [calc_detrainment_depth]
hdtnc = "CESM1_HTR_FULL_hdetrain_NAtl.nc"
ds_hdetrain = xr.open_dataset(mldpath+hdtnc).load()

# Load estimates of SALT and TEMP


vnames = ["TEMP","SALT"]
#%% Load the depth

e = 0
d = 0 

# Input: ds (time x z_t), lonf, latf, mld

dspt = ds.isel(dir=d,ensemble=e) # (time, z_t)  # Time x Depth

lonf = dspt.TLONG.values - 360 #  convert to degrees west
latf = dspt.TLAT.values

mldpt = ds_mld.sel(lon=lonf,lat=latf,method='nearest').isel(ens=e)
kprev,_ = scm.find_kprev(mldpt.h.values)

mons3 = proc.get_monstr()



#%% Double check this by looking at the actual profile

vv = 0


plotvar     = dspt[vnames[vv]].groupby('time.month').mean('time').T.values # Plot the actual mean seasonal cycle
plotdetrain = ds_hdetrain.sel(lon=lonf,lat=latf,method='nearest').isel(ens=e)

z_t = plotvar.z_t

plot_dz      = (np.roll(plotvar,1,axis=0) - np.roll(plotvar,-1,axis=0)) / (np.roll(z_t,1) - np.roll(z_t,-1))[:,None]

auto_dz      = np.gradient(plotvar,z_t,axis=0)

fig,ax  = plt.subplots(1,1)

# Plot the actual variable
pcm     = ax.pcolormesh(mons3,z_t,plotvar)
plt.gca().invert_yaxis()
fig.colorbar(pcm,ax=ax,fraction=0.045,pad=0.01)

# Contour vertical gradient (centered difference)
cl      = ax.contour(mons3,z_t[1:-1],plot_dz)

# Plot the detrainment depths
ax.plot(mons3,plotdetrain.h,marker="x",c="red",label="detrain depths")


ax.legend(loc='lower center')

#%% Compare Gradient Functions (Debug)


plot_dz      = (np.roll(plotvar,1,axis=0) - np.roll(plotvar,-1,axis=0)) / (np.roll(z_t,1) - np.roll(z_t,-1))[:,None]


fig,axs = plt.subplots(1,2)

ax = axs[0]
pcm = ax.pcolormesh(auto_dz)
fig.colorbar(pcm,ax=ax)
ax.set_title("Gradient Function")

ax = axs[1]
pcm = ax.pcolormesh(plot_dz)
fig.colorbar(pcm,ax=ax)
ax.set_title("Roll Gradient")
