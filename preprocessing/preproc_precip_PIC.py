#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Preprocess Precipitation from PiControl

Remove Seasonal Cycle, Detrend

Works with output from combine_precip.py

Created on Thu Jan  4 17:07:15 2024

@author: gliu
"""




import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import sys
from tqdm import tqdm
import copy


import scipy as sp

#%% Import Custom Modules
amvpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/03_Scripts/" # amv module
scmpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/03_Scripts/stochmod/model/"

sys.path.append(amvpath)
sys.path.append(scmpath)

from amv import proc,viz
import scm
import amv.loaders as dl
import yo_box as ybx

#%% User Inputs
ncpath="/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/raw/"
ncname="PRECTOT_PIC_FULL.nc"

lonf = -30
latf = 50


#%% Visualize and load the data

# Load Data
ds  = xr.open_dataset(ncpath+ncname).load()


# Remove the seasonal cycle
dsa = proc.xrdeseason(ds)

# Check Plots
fig,axs = viz.geosubplots(1,2)

pcm = axs[0].pcolormesh(ds.lon,ds.lat,ds.PRECTOT.mean('time'))
axs[0].coastlines(color='w')
fig.colorbar(pcm,ax=axs[0],orientation='horizontal')

pcm = axs[1].pcolormesh(ds.lon,ds.lat,dsa.PRECTOT.std('time'),cmap='cmo.dense')
axs[1].coastlines(color='w')
fig.colorbar(pcm,ax=axs[1],orientation='horizontal')




#%% Check plot at a point

fig,ax= plt.subplots(1,1)
ax.plot(dsa.sel(lon=lonf,lat=latf,method='nearest').PRECTOT)
ax.set_title("Precipitation Anomaly")

#%% Part II, Compute Stochastic Evaporation Forcing


# Load T
# Load Q_LH
# Load HFF_LH












