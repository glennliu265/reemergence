#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Investigate MLD Variability

Created on Mon Jun 24 15:55:08 2024

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

#%% Indicate Plotting Parameters (taken from visualize_rem_cmip6)


bboxplot                    = [-80,0,10,65]
mpl.rcParams['font.family'] = 'Avenir'
mons3                       = proc.get_monstr(nletters=3)
fsz_tick                    = 18
fsz_axis                    = 14
fsz_title                   = 16
rhocrit                     = proc.ttest_rho(0.05,2,86)

proj                        = ccrs.PlateCarree()



#%% Load HMXL File

ncname      = rawpath + "CESM1LE_HMXL_NAtl_19200101_20050101_bilinear.nc"
ds          = xr.open_dataset(ncname).load()
h           = ds.HMXL

h_scycle    = h.groupby('time.month').mean('time')
h_stdev     = h.groupby('time.month').std('time')
h_ratio     = h_stdev/h_scycle

#%% Load Ice Edge and other variables


#%% Plot some pointwise views

im = 1
pcolor=False
cints= np.arange(0,0.85,0.05)
#from im in range(12):

hratiomon   = h_ratio.isel(month=im)

fig,ax,_    = viz.init_orthomap(1,1,bboxplot,figsize=(16,4))

ax          = viz.add_coast_grid(ax,bbox=bboxplot,fill_color='lightgray')

pv          = hratiomon.mean('ensemble')
if pcolor:
    pcm         = ax.pcolormesh(pv.lon,pv.lat,pv,transform=proj,cmap='cmo.deep',vmin=cints[0],vmax=cints[-1])
else:
    pcm         = ax.contourf(pv.lon,pv.lat,pv,transform=proj,cmap='cmo.deep',levels=cints)
    
ax.set_title(mons3[im])

cb          = viz.hcbar(pcm,ax=ax,fraction=0.045)
cb.set_label(u"$h_{stdev}/h_{mean}$")
#cb.set_label(u"$\frac{h_{mean}}{h_{st.dev}}$")


#%%

lonf       = -30
latf       = 50
proc.selpt_ds(h_stdev,lonf,latf).mean('ensemble')



correction_pos = 1/(h_scycle + h_stdev**2)
correction_neg = 1/(h_scycle - h_stdev**2)


proc.selpt_ds(correction_pos,lonf,latf).mean('ensemble')
proc.selpt_ds(correction_neg,lonf,latf).mean('ensemble')




