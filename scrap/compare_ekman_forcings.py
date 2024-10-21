#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Compare Ekman Forcings (with correction factors)
Check Before and After implementation of correction, using NAO direct regression and TAU regressions...

Created on Tue Aug 27 12:14:03 2024

@author: gliu

"""


import xarray as xr
import numpy as np
import matplotlib as mpl
import scipy as sp

import matplotlib.pyplot as plt
import sys
import glob
import os

import tqdm
import time

import cartopy.crs as ccrs

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
pathdict    = rparams.machine_paths[machine]

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

#%% Plotting Information

bboxplot                    = [-80,0,20,65]
mpl.rcParams['font.family'] = 'Avenir'
mons3                       = proc.get_monstr(nletters=3)
fsz_tick                    = 18
fsz_axis                    = 22
fsz_title                   = 28
proj                        = ccrs.PlateCarree()

#%% Other User Edits

# Set Constants
omega = 7.2921e-5           # rad/sec
rho   = 1026                # kg/m3
cp0   = 3996                # [J/(kg*C)]
mons3 = proc.get_monstr()   # ('Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec')

dtmon  = 3600*24*30

# Load mixed layer depth climatological cycle, already converted to meters
#mldpath   = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/proc/model_input/mld/" # Take from model input file, processed by prep_SSS_inputs
mldpath   = input_path + "mld/"
mldnc     = "CESM1_HTR_FULL_HMXL_NAtl_EnsAvg.nc" #NOTE: this file is in cm
hclim     = xr.open_dataset(mldpath + mldnc).h.load() # [mon x ens x lat x lon]
#hclim     = preproc_dimname(hclim) # ('ens', 'mon', 'lat', 'lon')



vname = "SSS"


#%% Try for 1 variable first (SST)

if vname == "SST":
    # 
    vlms        =[ -.1,.1]
    
    ncname      = input_path + "forcing/CESM1_HTR_FULL_Qek_SST_NAO_DirReg_NAtl_corrected_EnsAvgFirst.nc"
    ds_qekreg   = xr.open_dataset(ncname).load()
    qek_reg     = ds_qekreg.Qek
    qek_reg_conv = qek_reg * dtmon
    
    #
    ncname1     = input_path + "forcing/CESM1_HTR_FULL_Qek_SST_NAO_nomasklag1_nroll0_NAtl_EnsAvg.nc"
    ds_taureg   = xr.open_dataset(ncname1).load()
    qek_taureg  = ds_taureg.Qek
    # Perform conversion for qek_taureg
    qek_taureg_conv = qek_taureg * dtmon / (rho * cp0 * hclim)
else:
    # 
    vlms        =[ -.01,.01]
    
    ncname      = input_path + "forcing/CESM1_HTR_FULL_Qek_SSS_NAO_DirReg_NAtl_corrected_EnsAvgFirst.nc"
    ds_qekreg   = xr.open_dataset(ncname).load()
    qek_reg     = ds_qekreg.Qek
    qek_reg_conv = qek_reg * dtmon
    
    #
    ncname1     = input_path + "forcing/CESM1_HTR_FULL_Qek_SSS_NAO_nomasklag1_nroll0_NAtl_EnsAvg.nc"
    ds_taureg   = xr.open_dataset(ncname1).load()
    qek_taureg  = ds_taureg.Qek
    # Perform conversion for qek_taureg
    qek_taureg_conv = qek_taureg * dtmon 

plotvars = [qek_reg_conv,qek_taureg_conv]
plotnames = ["Direct Regression","Wind Stress Regression"]
#%% Investigate Differences between the two


im          = 0
imode       = 0
fig,axs,_   = viz.init_orthomap(1,2,bboxplot,figsize=(16,12),centlon=-40)


for a,ax in enumerate(axs):
    
    ax          = viz.add_coast_grid(ax,bbox=bboxplot,fill_color='lightgray')
    plotvar     = plotvars[a].isel(mon=im,mode=imode)
    pcm         = ax.pcolormesh(plotvar.lon,plotvar.lat,plotvar,
                                cmap='cmo.balance',vmin=vlms[0],vmax=vlms[1],
                                transform=proj)
    cb          = viz.hcbar(pcm,ax=ax)
    ax.set_title(plotnames[a])
    
#%% Plot ssq of outpout

im          = 0
imode       = 0
fig,axs,_   = viz.init_orthomap(1,2,bboxplot,figsize=(16,12),centlon=-40)
vlms        = [ -.1,.1]

for a,ax in enumerate(axs):
    
    ax          = viz.add_coast_grid(ax,bbox=bboxplot,fill_color='lightgray')
    plotvar     = np.sqrt((plotvars[a].isel(mon=im)**2).sum('mode'))
    if a == 0:
        plotvar = plotvar# * 100
    pcm         = ax.pcolormesh(plotvar.lon,plotvar.lat,plotvar,
                                cmap='cmo.balance',vmin=vlms[0],vmax=vlms[1],
                                transform=proj)
    cb          = viz.hcbar(pcm,ax=ax)
    ax.set_title(plotnames[a] + '2')

#%% Check differences

im          = 0
imode       = 0
fig,ax,_    = viz.init_orthomap(1,1,bboxplot,figsize=(8.5,4),centlon=-40)
vlms        = [ -.0008,.0008]

ax          = viz.add_coast_grid(ax,bbox=bboxplot,fill_color='lightgray')
plotvar_qek = np.sqrt((plotvars[0].isel(mon=im)**2).sum('mode'))
plotvar_tau = np.sqrt((plotvars[1].isel(mon=im)**2).sum('mode'))
plotvar     = plotvar_qek-plotvar_tau
if a == 0:
    plotvar = plotvar# * 100
pcm         = ax.pcolormesh(plotvar.lon,plotvar.lat,plotvar,
                            cmap='cmo.balance',vmin=vlms[0],vmax=vlms[1],
                            transform=proj)
cb          = viz.hcbar(pcm,ax=ax)
cb.set_label("Direct Regression - Qek")
ax.set_title('Qek Total Forcing Differences for Mon %i (SST)' % (im+1))
    
    
#%% Perform Conversion




