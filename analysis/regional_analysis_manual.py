#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Manually do some regional analysis

Created on Thu Jul  4 15:22:32 2024

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

bboxplot        = [-80,0,10,65]
mpl.rcParams['font.family'] = 'Avenir'
mons3           = proc.get_monstr(nletters=3)

fsz_tick        = 18
fsz_axis        = 20
fsz_title       = 16

rhocrit = proc.ttest_rho(0.05,2,86)

proj= ccrs.PlateCarree()



#%% Indicate netcdfs:
    
ncname = "cesm2_pic_0200to2000_TS_ACF_lag00to60_ALL_ensALL.nc"
vname  = "TS"
ncpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/proc/"
ds1    = xr.open_dataset(ncpath+ncname).acf.load().squeeze()


ncname = "SM_SST_cesm2_pic_noQek_SST_autocorrelation_thresALL_lag00to60.nc"
vname  = "SST"
ncpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/proc/"
ds2    = xr.open_dataset(ncpath+ncname)[vname].load()#.acf.load().squeeze()

ds_in  = [ds1,ds2]
ds_in  = proc.resize_ds(ds_in)

#%% Compare wintertime ACF over bounding box
bbsel   = [-45,-30,50,65]
selmons = [1,]

ds_sel  = [proc.sel_region_xr(ds.isel(mons=selmons).mean('mons'),bbsel).mean('lat').mean('lon') for ds in ds_in]

#%% Plot Mean ACF
expnames = ["CESM2 PIC","Stochastic Model"]

kmonth    = selmons[0]
lags      = ds_sel[0].lags.data
xtks      = lags[::3]


fig       = plt.figure(figsize=(18,6.5))
gs        = gridspec.GridSpec(4,4)


# --------------------------------- # Locator
ax1       = fig.add_subplot(gs[0:3,0],projection=ccrs.PlateCarree())
ax1       = viz.add_coast_grid(ax1,bbox=bboxplot,fill_color="lightgray")
ax1.set_title(bbsel)
ax1 = viz.plot_box(bbsel)



ax2       = fig.add_subplot(gs[1:3,1:])
ax2,_     = viz.init_acplot(kmonth,xtks,lags,title="",)


for ii in range(2):
    ax2.plot(lags,ds_sel[ii].squeeze().data,label=expnames[ii])

ax2.legend()

