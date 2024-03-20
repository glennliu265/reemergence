#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Created on Thu Jan 25 01:45:25 2024


# Heres the idea:

Script 1
--------
# For a single point
# 1. Load in all the ensemble members
# 2. Repair the ensemble member where there is an issue
# 3. Save repaired version

Script 2
--------
# 4. Deseason and Detrend with Ensemble Average
# 5. Compute ACF
# 6. Do Exponential Fit



Copied from 

@author: gliu
"""

import time
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import sys
from tqdm import tqdm
import scipy as sp

#%%
# stormtrack
amvpath = "/home/glliu/00_Scripts/01_Projects/00_Commons/" # amv module
scmpath = "/home/glliu/00_Scripts/01_Projects/01_AMV/02_stochmod/stochmod/model/" # scm module

sys.path.append(amvpath)
sys.path.append(scmpath)

from amv import proc,viz
import scm
import amv.loaders as dl

#%% Set longitude and latitude ranges

# Set filepaths
vname    = "TEMP"
keepvars = [vname,"TLONG","TLAT","time","z_t"] 
outdir   = '/stormtrack/data3/glliu/01_Data/02_AMV_Project/03_reemergence/proc/CESM1/%s/hlim/' % vname
mconfig  = "HTR_FULL"

# Set Bounding Boxes
#latlonpath = "/stormtrack/data3/glliu/01_Data/02_AMV_Project/02_stochmod/Model_Data/model_input/ensavg_nhflxdamping_monwin3_sig020_dof082_mode4.mat"
bbox    = [-80,0,20,65]
#lon,lat = scm.load_latlon(datpath=latlonpath)

llpath    = "/stormtrack/data3/glliu/01_Data/02_AMV_Project/02_stochmod/Model_Data/"
lon       = np.load(llpath + "lon.npy")
lat       = np.load(llpath + "lat.npy")
lonr      = lon[(lon >= bbox[0]) * (lon <= bbox[1])]
latr      = lat[(lat >= bbox[2]) * (lat <= bbox[3])]
nlon,nlat = len(lonr),len(latr)

mnum      = dl.get_mnum()

# Search Options
searchdeg = 0.2
atm       = False

#%% Load mixed layer depth to speed up indexing

# Set MLDs
mldpath = "/stormtrack/data3/glliu/01_Data/02_AMV_Project/03_reemergence/proc/model_input/mld/"
mldnc   = mldpath + "CESM1_HTR_FULL_HMXL_NAtl.nc"
dsmld   = xr.open_dataset(mldnc)

# Find maximum of each point for the ensemble member, convert to cm
hmax_bypt = dsmld.h.max(('mon','ens')) * 100 # [ens x lat x lon]

#%% Load TLat/TLon


tllpath = "/home/glliu/01_Data/tlat_tlon.npz"
ldtl    = np.load(tllpath,allow_pickle=True)
tlon    = ldtl['tlon']
tlat    = ldtl['tlat']


#nlon is 384
#nlat is 320
# TLONG is 384 x 320 (lon x lat)
# TLAT is n

# Moved this to Proc
def get_pt_nearest(ds,lonf,latf,tlon_name="TLONG",tlat_name="TLAT",debug=True):

    tlon_name = "TLONG"
    tlat_name = "TLAT"
    x1name    = "nlat"[]
    x2name    = "nlon"
    tlon      = ds[tlon_name].values
    tlat      = ds[tlat_name].values

    # Find minimum in tlon
    # Based on https://stackoverflow.com/questions/58758480/xarray-select-nearest-lat-lon-with-multi-dimension-coordinates
    londiff   =  np.abs(tlon - lonf)
    latdiff   =  np.abs(tlat - latf)
    locdiff   =  londiff+latdiff

    # Get Point Indexes
    ([x1], [x2]) = np.where(locdiff == np.min(locdiff))
    #plt.pcolormesh(np.maximum(londiff,latdiff)),plt.colorbar(),plt.show()
    #plt.pcolormesh(locdiff),plt.colorbar(),plt.show()

    if debug:
        print("Nearest point to (%.2f,%.2f) is (%.2f,%.2f) at index (%i,%i)" % (lonf,latf,tlon[x1,x2],tlat[x1,x2],x1,x2))

    return ds.isel(**{x1name : x1, x2name : x2})

# st = time.time()
# saltpt = dspt.SALT.values
# print("Loaded data in %.2fs" % (time.time()-st))


# st = time.time()
# dspt  = proc.getpt_pop(lonf,latf,ds,returnarray=False,searchdeg=searchdeg)
# print("Loaded data in %.2fs" % (time.time()-st))

# tlonmin      =  np.unravel_index(tlonmin_flat,tlon.shape)
# tlatmin_flat = np.argmin(np.abs(tlat - lonf))

#%% Extract Point Data (Copied from get_point_data_stormtrack) --------

if vname == "SSS":
    nens = 42
    datpath = "/vortex/jetstream/climate/data1/yokwon/CESM1_LE/processed/ocn/proc/tseries/monthly/"
elif vname == "SALT":
    nens = 42
    datpath = "/stormtrack/data4/glliu/01_Data/CESM1_LE/"
elif vname == "TEMP":
    nens = 42
    datpath = "/stormtrack/data4/share/deep_learning/data_yuchiaol/cesm_le/"
else:
    nens = 40
    datpath = None


if "HTR" in mconfig:
    dsloader = dl.load_htr
elif "RCP85" in mconfig:
    dsloader = dl.load_rcp85

#% Point Loop 
# Set up Point Loop
debug=True

for o in range(nlon):
    for a in range(nlat):
        # Get the Point
        lonf = lonr[o]
        if lonf < 0:
            lonf += 360
        latf = latr[a]
        locfn,loctitle      = proc.make_locstring(lonf,latf)
        
        # Get the maximum MLD to set to
        hmax = hmax_bypt.sel(lon=lonf-360,lat=latf,method='nearest').values.item()
        if hmax == 0:
            # No Mixed Layer Depth at point, skip
            continue
        
        # Start LOOP HERE 
        nanflag = False
        dsptall = []
        for e in tqdm(range(nens)):
            if nanflag:
                continue
            
            # Looping by Ensemble member
            ensid  = mnum[e]
            ds     = dl.load_htr(vname,ensid,atm=atm,datpath=datpath,return_da=False)
            dsdrop = proc.ds_dropvars(ds,keepvars)
            
            # Restrict to mixed layer depth
            dsdrop = dsdrop.sel(z_t=slice(0,hmax))
            
            # Get point
            st     = time.time()
            dspt   = proc.get_pt_nearest(dsdrop,lonf,latf,debug=False) 
            #dspt  = proc.getpt_pop(lonf,latf,dsdrop,returnarray=False,searchdeg=searchdeg)
            if debug:
                print("Subset data in %.2fs" % (time.time()-st))
            
            st   = time.time()
            dspt = dspt.load() # [Time x Z]
            if debug:
                print("Load data in %.2fs" % (time.time()-st))
            
            # st = time.time()
            # dspt  = proc.getpt_pop(lonf,latf,ds,returnarray=False,searchdeg=searchdeg) # << This seems like a bottleneck, might be faster to start from TLAT/.TLONG Mesh and pull from there 
            # print("Loaded data in %.2fs" % (time.time()-st))
            
            # If all points are NaN, exit
            if np.all(np.isnan(dspt[vname].values)):
                nanflag=True
                continue
            
            # If some points are NaN, repair... (actually will move this to a different script)
            # if np.any(np.isnan(dspt[vname].values)):
            #     (knan_t,knan_z) = np.where(np.isnan(dspt[vname]))
            
            dsptall.append(dspt)
            # <End ensemble loop>
            
        if nanflag:
            continue
        
        # Concatenate
        dsptall       = xr.concat(dsptall,dim="ens") # [Ens x Time]
        encoding_dict = {vname:{'zlib':True}}
        
        # Save output
        locfn_update = "lon%03i_lat%02i" % (lonf,latf) 
        savename     = "%sCESM_%s_%s_%s.nc" % (outdir,mconfig,vname,locfn_update,)
        dsptall.to_netcdf(savename,encoding=encoding_dict)
        # < End Lat Loop >
    # < End Lon Loop >


