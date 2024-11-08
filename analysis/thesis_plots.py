#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Thesis Plots

Copied Sections from compare_AMV_HadISST

Created on Fri Nov  8 09:27:51 2024

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

#%% Import Custom Modules

amvpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/03_Scripts/" # amv module
scmpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/03_Scripts/stochmod/model/"

sys.path.append(amvpath)
sys.path.append(scmpath)

from amv import proc,viz
import scm
import amv.loaders as dl
import cvd_utils as cvd

#%% User Edits

datpath     = "/Users/gliu/Downloads/02_Research/01_Projects/04_Predict_AMV/03_Scripts/CESM_data/"#"../../CESM_data/"
outpath     = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/02_Figures/Thesis/"

# Set File Names
vnames      = ['sst','sss','psl']
vnamelong   = ["Sea Surface Temperature (degC)","Sea Surface Salinity (psu)","Sea Level Pressure (hPa)"]
fn1         = "CESM1LE_sst_NAtl_19200101_20051201_Regridded2deg.nc"
fn2         = "CESM1LE_sss_NAtl_19200101_20051201_Regridded2deg.nc"
fn3         = "CESM1LE_psl_NAtl_19200101_20051201_Regridded2deg.nc"
fns         = [fn1,fn2,fn3]

# Plotting Box
bbox        = [-80,0,0,65] # North Atlantic [lonW, lonE, latS, latN]

#%% Load and compute AMV


dtr = True

# 
fn  = "hadisst.1870-01-01_2018-12-01.nc"
dsh = xr.open_dataset(datpath+fn)
ssth = dsh.sst.values # [time x lat x lon]
lath = dsh.lat.values
lonh = dsh.lon.values
times = dsh.time.values
timesmon = np.datetime_as_string(times,unit="M")
#timesmon = timesmon.astype('str')
timesyr  = np.datetime_as_string(times,unit="Y")[:]

# Calculate Monthly Anomalies
ssts = ssth.transpose(2,1,0)
nlon,nlat,nmon = ssts.shape
ssts = ssts.reshape(nlon,nlat,int(nmon/12),12)
ssta = ssts - ssts.mean(2)[:,:,None,:]
ssta = ssta.reshape(nlon,nlat,nmon)



# Transpose to [time lat lon]
ssta   = ssta.transpose(2,1,0)

# Calculate AMV and AMV Pattern
#amvid,amvpattern=proc.calc_AMVquick(ssta,lonh,lath,bbox,runmean=False,)


amvid       = calc_AMV_index('NAT',ssta[None,:,:,:],lath,lonh,lp=True,dtr=dtr)
amvidstd    = amvid/amvid.std(1)[:,None] # Standardize
amvid       = amvid.squeeze()
amvidraw    = calc_AMV_index('NAT',ssta[None,:,:,:],lath,lonh,lp=False,dtr=dtr)
amvidraw    = amvidraw.squeeze()

# Calculate undetrended version
amvid_undtr     = calc_AMV_index('NAT',ssta[None,:,:,:],lath,lonh,lp=True,dtr=False)
amvidraw_undtr  = calc_AMV_index('NAT',ssta[None,:,:,:],lath,lonh,lp=False,dtr=False)

# Regress back to sstanomalies to obtain AMV pattern
#ssta   = ssta.transpose(1,0,2,3) # [time x ens x lon x lat]
sstar         = ssta.reshape(nmon,nlat*nlon) 
beta,_        = regress_2d(amvidstd.squeeze(),sstar)
beta_undtr,_  = regress_2d((amvid_undtr/amvid_undtr.std(1)[:,None]).squeeze(),sstar)
amvpath       = beta
amvpath       = amvpath.reshape(nlat,nlon)








