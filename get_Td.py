#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

This script retrieves Td and Sd (temperature and salinity at the base of the mixed layer)
Using the maximum of the climatological MLD cycle.


Created on Tue Oct 10 15:07:48 2023

Methodology

(1) Load or calculate maximum climatological wintertime MLD
(2) Open 3D TEMP and SALINITY
(3) Select level below min level and get timeseries of T' or S'. Save the level. 
(4) Save to file

(5) At each point...
    (Compute monthly autocorrelation function)
    (Fit estimated damping timescale to get timescale of anomalies)


@author: gliu
"""

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import glob
import sys
import tqdm

#%%

amvpath = "/home/glliu/00_Scripts/01_Projects/00_Commons/" # amv module
scmpath = "/home/glliu/00_Scripts/01_Projects/01_AMV/02_stochmod/stochmod/model/" # scm module

sys.path.append(amvpath)
sys.path.append(scmpath)


from amv import proc
#%%


# MLD Path: HMXL_FULL_HTR_bilinear_num00.nc (Counts start from 0)
mldpath = "/stormtrack/data3/glliu/01_Data/02_AMV_Project/02_stochmod/HMXL/"

# TEMP and Salinity Path
ncvars  = ["TEMP","SALT"]
ncpaths = [
    "/stormtrack/data4/share/deep_learning/data_yuchiaol/cesm_le/TEMP/",
    "/stormtrack/data4/glliu/01_Data/CESM1_LE/SALT/"
    ]

# Path to Output
outpath = "/stormtrack/data3/glliu/01_Data/02_AMV_Project/02_stochmod/Td/"


# Other EDits
bbox = [-80,0,0,65]

#%% Get the data paths

ncsalts = glob.glob(ncpaths[1] + "*.nc")
ncsalts = [nc for nc in ncsalts if "OIC" not in nc]
ncsalts.sort()


nctemps = glob.glob(ncpaths[0] + "*.nc")
nctemps = [nc for nc in nctemps if "OIC" not in nc]
nctemps.sort()


ncmlds = []
for e in range(40):
    ncmlds.append("%sHMXL_FULL_HTR_bilinear_num%02i.nc" % (mldpath,e))
#%% Loop for each ensemble member

# Part (1) Load and calculate maximum Climatological Wintertime MLD

e = 0

# Get the mixed layer depth, subset region and load
ds_mld = xr.open_dataset(ncmlds[e])
ds_mld = proc.format_ds(ds_mld)
ds_mld = proc.sel_region_xr(ds_mld,bbox)
mld    = ds_mld['HMXL'].values
lon    = ds_mld.lon.values
lat    = ds_mld.lat.values
times  = ds_mld.time.values
ntime,nlat,nlon = mld.shape

# Compute the wintertime mean
hclim     = proc.calc_clim(mld,0)
hmax      = np.nanmax(hclim,0)/100 # <-- SAVE THIS
hmax_mon  = np.argmax(hclim,0) # <-- SAVE THIS

# Load views of temp and sal
dst       = xr.open_dataset(nctemps[e])
dss       = xr.open_dataset(ncsalts[e])
z_bot     = dss.z_w_bot/100 # Depth to bottom of layer

#%% PREALLOCATE

debug = False
# o = 64, a =0
Td_map = np.zeros((ntime,nlat,nlon)) * np.nan
Sd_map = Td_map.copy()
z_map  = np.zeros((nlat,nlon)) * np.nan
for o in range(nlon):
    for a in tqdm.tqdm(range(nlat)):
        hmax_pt = hmax[a,o]
        
        if np.isnan(hmax_pt):
            continue
        
        # Selec the point
        lonf = lon[o]
        if lonf < 0:
            lonf += 360
        latf = lat[a]
        tpt = proc.find_tlatlon(dst,lonf,latf,verbose=False)
        spt = proc.find_tlatlon(dss,lonf,latf,verbose=False)
        
        # Find the index corresponding to hmax
        id_z = np.argmin(np.abs(z_bot.values - hmax_pt))
        if debug:
            print("Nearest MLD to %.2f was %.2f" % (hmax_pt,z_bot.values[id_z]))
        
        tsel = tpt.isel(z_t=id_z)#.load()
        ssel = spt.isel(z_t=id_z)#.load() # [Time x Z]
        Td_map[:,a,o] = tsel['TEMP'].values.copy()
        Sd_map[:,a,o] = ssel['SALT'].values.copy()
        z_map[a,o] = tsel.z_t.values/100
        
        #temp_sel = dst['TEMP'].isel(z_t=id_z)
        
#%% SAVE OUTPUT

# Save Td
savename = "%sCESM1_HTR_Td_ens%02i.nc" % (outpath,e+1)
proc.numpy_to_da(Td_map,times,lat,lon,"Td",savenetcdf=savename)

# Save Sd
savename = "%sCESM1_HTR_Sd_ens%02i.nc" % (outpath,e+1)
proc.numpy_to_da(Sd_map,times,lat,lon,"Sd",savenetcdf=savename)

# Save MLDs
savename = "%sCESM1_HTR_hmax_ens%02i.npz" % (outpath,e+1)
np.savez(savename,**{
    'hmax':hmax,
    'hmax_monid':hmax_mon,
    'z_t':z_map
    })







