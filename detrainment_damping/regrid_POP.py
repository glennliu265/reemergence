#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Regrid POP data, copied from [regrid_detrainment_damping]

Regrid Detrainment Damping calculated by preproc_detrainment_data

Inputs:
------------------------
    
    varname : dims                              - units                 - processing script
    lbd_d   :  (mon, nlat, nlon)                [-1/mon]                preproc_detrainment_data
    tau     :  (mon, z_t, nlat, nlon)           [-1/mon]
    acf_est :  (mon, lag, z_t, nlat, nlon)      [corr]
    acf_mon :  (mon, lag, z_t, nlat, nlon)      [corr]

Outputs: 
------------------------

    varname : dims                              - units 
    lbd_d   :  (mon, lat, lon)                  [-1/mon]         
    tau     :  (mon, z_t, lat, lon)             [-1/mon]
    acf_est :  (mon, lag, z_t, lat, lon)        [corr]
    acf_mon :  (mon, lag, z_t, lat, lon)        [corr]


Output File Name:

What does this script do?
------------------------
(1) Load in data processed by 
Script History
------------------------


Created on Wed Feb 14 11:11:20 2024

@author: gliu
"""

import time
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import xarray as xr
import sys
from tqdm import tqdm
import scipy as sp
import cartopy.crs as ccrs

# ----------------------------------
#%% Import custom modules and paths
# ----------------------------------

# Indicate the Machine!
machine = "stormtrack"

# First Load the Parameter File
sys.path.append("../")
import reemergence_params as rparams

# Paths and Load Modules
pathdict    = rparams.machine_paths[machine]

sys.path.append(pathdict['amvpath'])
sys.path.append(pathdict['scmpath'])
from amv import proc

# Set needed paths
figpath     = pathdict['figpath']
proc.makedir(figpath)
input_path  = pathdict['input_path']
rawpath     = pathdict['raw_path']

mldpath     = input_path + "mld/"
outpath     = rawpath + "ocn_var_3d/"

vnames      = ["TEMP",]#"SALT",]
nens        = 42
loopens     = [32,]

#%% Indicate which files to process

fns = []
for v in range(len(vnames)):
    for e in loopens:
        fn = "CESM1_HTR_FULL_lbd_d_params_%s_detrendensmean_lagmax3_ens%02i.nc" % (vnames[v],e+1)
        fns.append(fn)

# fns     = ["CESM1_HTR_FULL_lbd_d_params_SALT_detrendensmean_lagmax3_ens01.nc",
#           "CESM1_HTR_FULL_lbd_d_params_TEMP_detrendensmean_lagmax3_ens01.nc"]

# bounding box for final output
bbox    = [-80,0,20,65]

#%% Retrieve TLAT/TLON from a file in outpath
# Note this part should change as I modified preproc_detrainment_data to include tlat and tlon as a coord
fnlat   = "SALT_NATL_ens01.nc" # Name of file to take tlat/tlon information from

dstl    = xr.open_dataset(outpath+fnlat)
tlat    = dstl.TLAT.values
tlon    = dstl.TLONG.values

#%% Retrieve dimensions of CESM1 from another file

mldnc   = "CESM1_HTR_FULL_HMXL_NAtl.nc"
dsh     = xr.open_dataset(mldpath+mldnc)

dshreg  = dsh.sel(lon=slice(bbox[0],bbox[1]),lat=slice(bbox[2],bbox[3]))

outlat  = dshreg.lat.values
outlon  = dshreg.lon.values

#%% Seems the easier way might just be to do this loopwise (silly but I guess...)

v       = 1
for v in range(len(fns)):
    
    ds          = xr.open_dataset(outpath+fns[v]).load()
    dsl         = ds.assign(lon=(['nlat','nlon'],tlon),lat=(['nlat','nlon'],tlat))
    dsvars      = ["lbd_d","tau","acf_est","acf_mon"]
    
    nlon,nlat   = len(outlon),len(outlat)
    
    # Preallocate
    nz          =  len(dsl.z_t)
    nlags       =  len(dsl.lag)
    
    lbd_d_all   = np.zeros((12,nlat,nlon)) * np.nan          # Estimated Detrainment Damping
    tau_est_all = np.zeros((12,nz,nlat,nlon))  * np.nan      # Fitted Timescales
    acf_est_all = np.zeros((12,nlags,nz,nlat,nlon)) * np.nan # Fitted ACF
    acf_mon_all = np.zeros((12,nlags,nz,nlat,nlon)) * np.nan # Actual ACF
    
    for o in tqdm(range(nlon)): # Took (1h 11 min if you don't load, 2 sec if you load, T-T)
        
        lonf = outlon[o]
        
        if lonf < 0:
            lonf += 360
            
        for a in range(nlat):
            latf   = outlat[a]
            
            # Get the nearest point
            outids = proc.get_pt_nearest(dsl,lonf,latf,tlon_name="lon",tlat_name="lat",returnid=True,debug=False)
            dspt   = dsl.isel(nlat=outids[0],nlon=outids[1])
            
            # Reassign variables
            lbd_d_all[:,a,o]        = dspt['lbd_d'].values
            tau_est_all[:,:,a,o]    = dspt['tau'].values
            acf_est_all[:,:,:,a,o]  = dspt['acf_est'].values
            acf_mon_all[:,:,:,a,o]  = dspt['acf_mon'].values
    
    #%% Apply mask based on h
    
    mask         = np.sum(dshreg.h.values,(0,1))
    mask[~np.isnan(mask)] = 1
    
    da_mask      = xr.DataArray(mask,coords=dict(lat=outlat,lon=outlon))
    
    #%% Remap and save variables
    
    nlat         = outlat
    nlon         = outlon
    mons         = np.arange(1,13,1)
    z            = dsl.z_t.values
    lags         = np.arange(0,37,1)
    
    # Make data arrays
    lcoords      = dict(mon=mons,lat=nlat,lon=nlon)
    da_lbdd      = xr.DataArray(lbd_d_all * mask[None,:,:],coords=lcoords,dims=lcoords,name="lbd_d") #* da_mask
    
    taucoords    = dict(mon=mons,z_t=z,lat=nlat,lon=nlon)
    da_tau       = xr.DataArray(tau_est_all * mask[None,None,:,:],coords=taucoords,dims=taucoords,name="tau")
    
    acfcoords    = dict(mon=mons,lag=lags,z_t=z,lat=nlat,lon=nlon)
    da_acf_est   = xr.DataArray(acf_est_all * mask[None,None,None,:,:],coords=acfcoords,dims=acfcoords,name="acf_est")
    da_acf_mon   = xr.DataArray(acf_mon_all* mask[None,None,None,:,:],coords=acfcoords,dims=acfcoords,name="acf_mon")
    
    ds_out       = xr.merge([da_lbdd,da_tau,da_acf_est,da_acf_mon,])
    edict        = proc.make_encoding_dict(ds_out)
    
    savename     = outpath+fns[v]
    savename_new = proc.addstrtoext(savename,"_regridNN",adjust=-1)
    
    ds_out.to_netcdf(savename_new,encoding=edict)
    
