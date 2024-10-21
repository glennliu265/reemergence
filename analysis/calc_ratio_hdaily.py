#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Compute the Ratio of Daily vs Monthly Mixed-Layer Depth Variability
For the text in Paper Draft 04

Created on Fri Oct 18 15:41:08 2024

@author: gliu

"""

import numpy as np
import xarray as xr
import sys
import time
import matplotlib.pyplot as plt

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
from amv import proc,viz
import scm
import amv.loaders as dl
import yo_box as ybx

# Set needed paths
figpath     = pathdict['figpath']
proc.makedir(figpath)
input_path  = pathdict['input_path']
output_path = pathdict['output_path']
procpath    = pathdict['procpath']


#%% Indicate Paths

#figpath   = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/02_Figures/20240411/"
datpath   = pathdict['raw_path']
outpath   = pathdict['input_path']+"forcing/"
rawpath   = pathdict['raw_path']

#%% Load Point Information

# Get Point Info
pointset        = "PaperDraft02"
ptdict          = rparams.point_sets[pointset]
ptcoords        = ptdict['bboxes']
ptnames         = ptdict['regions']
ptnames_long    = ptdict['regions_long']
ptcols          = ptdict['rcols']
ptsty           = ptdict['rsty']

#%% Load data for each point

dsdaily_pt = []
for pp in range(3):
    lonf,latf = ptcoords[pp][0],ptcoords[pp][1]
    if lonf < 0:
        lonf = lonf + 360
    opath    = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/proc/CESM1/NATL_proc/daily/"
    savename = "%sDaily_HMXL_lon%03i,lat%03i.nc" % (opath,lonf,latf)
    ds  = xr.open_dataset(savename).load()
    dsdaily_pt.append(ds.copy())


#%% Load monthly variability

ncname           = rawpath + "CESM1LE_HMXL_NAtl_19200101_20050101_bilinear.nc"
ds_hprime        = xr.open_dataset(ncname).load().HMXL


ds_hprime_monvar = ds_hprime.groupby('time.month').std('time')


ds_hprime_monvar_pt = []
for pp in range(3):
    lonf,latf = ptcoords[pp][0],ptcoords[pp][1]
    ds = proc.selpt_ds(ds_hprime_monvar,lonf,latf)
    ds_hprime_monvar_pt.append(ds.copy())
    
    
#%% Load daily variability

ds_daily_monvar = []
for pp in range(3):
    
    ds = dsdaily_pt[pp].HMXL_2.groupby('time.month').std('time')
    ds_daily_monvar.append(ds)
    
#%% Make the plot at each location

mons3   = proc.get_monstr()
fig,axs = viz.init_monplot(1,3,figsize=(18,4.5))

for pp in range(3):
    
    
    ax      = axs[pp]
    hmon    = ds_hprime_monvar_pt[pp].mean('ensemble')/100
    ax.plot(mons3,hmon,label="$\sigma$(h') (Monthly)")
    
    hdaily  = ds_daily_monvar[pp]/100
    ax.plot(mons3,hdaily,label="$\sigma$(h') (Daily)")
    
    ax.legend()
    if pp ==0:
        ax.set_ylabel("Mixed Layer Depth Variability (m)")
    
    
    ax2 = ax.twinx()
    ax2.scatter(mons3,hdaily/hmon*100,c='gray',lw=0.75,ls='dotted')
    ax2.set_ylim([0,400])
    if pp == 2:
        ax2.set_ylabel("Ratio of Daily/Monthy Variability (%)",c='gray')
    
    ax.set_title(ptnames[])
    #ax.grid(True)
    
    








