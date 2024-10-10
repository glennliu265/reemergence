#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Scrap script to crop the UET/VNT Budget computed from ___
to the North Atlantic

Copied [process_bylevel_ens]

Created on Mon Sep 30 14:22:45 2024

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
#vname    = "VNT" # "VVEL"
#keepvars = [vname,"TLONG","TLAT","time","z_t"] 
#mconfig  = "HTR_FULL"


dpath   = "/stormtrack/data4/glliu/01_Data/CESM1_LE/proc/"
ncname  = "CESM1_VNT_UET_Budget.nc"




outpath  = "/stormtrack/data3/glliu/01_Data/02_AMV_Project/03_reemergence/proc/CESM1/NATL_proc/ocn_var_3d/"

# Set Bounding Boxes
#latlonpath = "/stormtrack/data3/glliu/01_Data/02_AMV_Project/02_stochmod/Model_Data/model_input/ensavg_nhflxdamping_monwin3_sig020_dof082_mode4.mat"
bbox    = [-80,0,20,65]
cropname = "NAtl"
#lon,lat = scm.load_latlon(datpath=latlonpath)

llpath    = "/stormtrack/data3/glliu/01_Data/02_AMV_Project/02_stochmod/Model_Data/"
lon       = np.load(llpath + "lon.npy")
lat       = np.load(llpath + "lat.npy")
lonr      = lon[(lon >= bbox[0]) * (lon <= bbox[1])]
latr      = lat[(lat >= bbox[2]) * (lat <= bbox[3])]
nlon,nlat = len(lonr),len(latr)

mnum      = [2,]#dl.get_mnum()

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
hmax_abs  = hmax_bypt.max(('lat','lon'))

# Check longitudinal extent, converting back to meters
hmax_zonal = hmax_bypt.max('lon')/100


#%% Load Tlon/Tlat, and the vertical coordinate?

tllpath = "/home/glliu/01_Data/tlat_tlon.npz"
ldtl    = np.load(tllpath,allow_pickle=True)
tlon    = ldtl['tlon']
tlat    = ldtl['tlat']

#%% Include analysis of certain ensemb
#%% Function (copied from get_mld_max working script)


def sel_region_xr_cv(ds2,bbox,debug=False):
    
    # Get mesh
    tlat = ds2.TLAT.values
    tlon = ds2.TLONG.values
    
    # Make Bool Mask
    latmask = (tlat >= bbox[2]) * (tlat <= bbox[3])
    
    # Three Cases
    # Case 1. Both are degrees west
    # Case 2. Crossing prime meridian (0,360)
    # Case 3. Crossing international date line (180,-180)
    # Case 4. Both are degrees east
    if np.any(np.array(bbox)[:2] < 0):
        print("Degrees West Detected")
        
        if np.all(np.array(bbox[:2])) < 0: # Case 1 Both are degrees west
            print("Both are degrees west")
            lonmask = (tlon >= bbox[0]+360) * (tlon <= bbox[1]+360)
            
        elif (bbox[0] < 0) and (bbox[1] >= 0): # Case 2 (crossing prime meridian)
            print("Crossing Prime Meridian")
            lonmaskE = (tlon >= bbox[0]+360) * (tlon <= 360) # [lonW to 360]
            if bbox[1] ==0:
                lonmaskW = 1
            else:
                lonmaskW = (tlon >= 0)           * (tlon <= bbox[1])       # [0 to lonE]
            
            lonmask = lonmaskE * lonmaskW
        elif (bbox[0] > 0) and (bbox[1] < 0): # Case 3 (crossing dateline)
            print("Crossing Dateline")
            lonmaskE = (tlon >= bbox[0]) * (tlon <= 180) # [lonW to 180]
            lonmaskW = (tlon >= 180)     * (tlon <= bbox[1]+360) # [lonW to 180]
            lonmask = lonmaskE * lonmaskW
    else:
        print("Everything is degrees east")
        lonmask = (tlon >= bbox[0]) * (tlon <= bbox[1])


    regmask = lonmask*latmask

    # Select the box
    if debug:
        plt.pcolormesh(lonmask*latmask),plt.colorbar(),plt.show()


    # Make a mask
    #ds2 = ds2[vname]#.isel(z_t=1)
    
    ds2.coords['mask'] = (('nlat', 'nlon'), regmask)
    
    st = time.time()
    ds2 = ds2.where(ds2.mask,drop=True)
    print("Loaded in %.2fs" % (time.time()-st))
    return ds2

#%% Extract file depending on the loader



ds    = xr.open_dataset(dpath + ncname,chunks={"time": 12})

# Fix February start
ds    = proc.fix_febstart(ds)
ds    = ds.rename(dict(nlat_t='nlat',nlon_t='nlon'))

dsreg  = sel_region_xr_cv(ds,bbox)#(dsz,bbox,vname)


# Save output to cropped name
savename = proc.addstrtoext(dpath+ncname,"_%s" % cropname,adjust=-1)
edict    = proc.make_encoding_dict(dsreg)
dsreg.to_netcdf(savename,encoding=edict)

#%%

e = 0
for e in np.arange(0,43):
    
    if vname == "SSS":
        nens = 42
        datpath = "/vortex/jetstream/climate/data1/yokwon/CESM1_LE/processed/ocn/proc/tseries/monthly/"
    elif vname in ["SALT","VNT","UET"]:
        nens = 42
        datpath = "/stormtrack/data4/glliu/01_Data/CESM1_LE/"
    elif vname == "TEMP":
        nens = 42
        datpath = "/stormtrack/data4/share/deep_learning/data_yuchiaol/cesm_le/"
    elif vname in ["UVEL","VVEL"]:
        nens = 42
        datpath = "/stormtrack/data4/glliu/01_Data/CESM1_LE/ocn/"
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
    
    
    # Looping by Ensemble member
    ensid  = mnum[e]
    ds     = dl.load_htr(vname,ensid,atm=atm,datpath=datpath,return_da=False)
    
    # Drop Variables
    dsdrop = proc.ds_dropvars(ds,keepvars)
    
    # Crop Level (To Maximum Clim MLD) (took 1867.38s for 44 levels)
    dsz    = dsdrop.sel(z_t=slice(0,hmax_abs.values))
    
    # Crop Time
    dsz    = proc.fix_febstart(dsz)
    dsz    = dsz.sel(time=slice("1920-01-01","2005-12-31"))
    
    # Crop Region (takes  1237.98s )
    dsreg  = sel_region_xr_cv(dsz,bbox,vname)
    
    
    # Save Output
    edict    = {vname:{'zlib':True}}
    savename = "%s%s_NATL_ens%02i.nc" % (outpath,vname,e+1)
    st = time.time()
    dsreg.to_netcdf(savename,encoding=edict)
    print("saved output in %.2fs to %s" % (time.time()-st,savename))






