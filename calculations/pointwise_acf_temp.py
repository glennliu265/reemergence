#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Compute Pointwise ACF for TEMP (rather than SST)

(1) Compile TEMP data and Save



Created on Thu May 23 16:18:22 2024

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



#%%

# bounding box for final output
bbox    = [-80,0,20,65]


#%% Pull Surface Data

datpath = "/stormtrack/data3/glliu/01_Data/02_AMV_Project/03_reemergence/proc/CESM1/NATL_proc/ocn_var_3d/"
dsall   = []
nens    = 42

for e in tqdm.tqdm(range(nens)):
    ncname = "%sTEMP_NATL_ens%02i.nc" % (datpath,e+1)
    if e == 32:
        ncname = proc.addstrtoext(ncname,"_repaired",adjust=-1)
        
        
    ds = xr.open_dataset(ncname).isel(z_t=0).load()
    dsall.append(ds)
    
dsall   = xr.concat(dsall,dim='ens')
outname = "%s/CESM1_HTR_FULL_TEMP_SURF_EnsALl.nc" % datpath
edict = proc.make_encoding_dict(dsall)

    

#%% Load MLD for reference grid

mldpath = input_path + "mld/"
mldnc   = "CESM1_HTR_FULL_HMXL_NAtl.nc"
dsh     = xr.open_dataset(mldpath+mldnc)

dshreg  = dsh.sel(lon=slice(bbox[0],bbox[1]),lat=slice(bbox[2],bbox[3]))

outlat  = dshreg.lat.values
outlon  = dshreg.lon.values

nmon,nens,nlat,nlon = dshreg.h.shape
ntime = dsall.TEMP.shape[1]
#%% Fix TLONG and TLAT


tlon    = dsall.isel(ens=0).TLONG.values
tlat    = dsall.isel(ens=0).TLAT.values

tlon_ds     = xr.DataArray(tlon,coords=dict(nlat=dsall.nlat,nlon=dsall.nlon),name="TLONG")
tlat_ds     = xr.DataArray(tlat,coords=dict(nlat=dsall.nlat,nlon=dsall.nlon),name="TLAT")

dstemp_new = xr.merge([dsall.TEMP.drop(['TLAT','TLONG']),tlon_ds,tlat_ds])

#%%

var_regrid   = np.zeros((nens,ntime,nlat,nlon)) * np.nan          # Estimated Detrainment Damping
for o in tqdm.tqdm(range(nlon)): # Took (1h 11 min if you don't load, 2 sec if you load, T-T)
    
    lonf = outlon[o]
    
    if lonf < 0:
        lonf += 360
        
    for a in range(nlat):
        latf   = outlat[a]
        
        # Get the nearest point
        outids = proc.get_pt_nearest(dstemp_new,lonf,latf,tlon_name="TLONG",tlat_name="TLAT",returnid=True,debug=False)
        dspt   = dstemp_new.isel(nlat=outids[0],nlon=outids[1])
        
        var_regrid[:,:,a,o] = dspt.TEMP.values
        
        
        
#%%

coords = dict(ens=np.arange(1,43,1),time=dsall.time,lat=outlat,lon=outlon)
da_out = xr.DataArray(var_regrid,coords=coords,dims=coords,name="TEMP")
edict  = {'TEMP':{'zlib':True}}
outname = "%sCESM1LE_TEMP_NAtl_19200101_20050101_NN.nc" % rawpath
da_out.to_netcdf(outname,encoding=edict)


#dsl         = ds.assign(lon=(['nlat','nlon'],tlon),lat=(['nlat','nlon'],tlat))
    
    