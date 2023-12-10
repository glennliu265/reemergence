#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Preprocess glorys12v1 data
- Merge Daily Data
- Crop to region and depth
- Drop all other variables
- Rename Lat/Lon dimensions and flip longitude if needed

Created on Wed Nov 22 13:58:08 2023

@author: gliu

"""

import numpy as np
import glob
import time
import xarray as xr
import sys
import tqdm

#%% Add Custom Paths (copied from Obsidian/DATA_CODE_CENTRAL)

# Stormtrack
amvpath = "/home/glliu/00_Scripts/01_Projects/00_Commons/" # amv module
scmpath = "/home/glliu/00_Scripts/01_Projects/01_AMV/02_stochmod/stochmod/model/" # scm module

sys.path.append(amvpath)
sys.path.append(scmpath)

from amv import proc,viz
import scm
import amv.loaders as dl

#%% Function
def crop_ds(ds,keepvars):
    
    ds         = proc.ds_dropvars(ds,keepvars)  # Drop Unnecessary variables
    id_surface = np.nanargmin(ds.depth)         # Select surface quantity
    ds         = ds.isel(depth=id_surface)
    ds         = proc.format_ds(ds,lonname='longitude',latname='latitude',verbose=False) # Rename (and flip) Lat x Lon
    ds         = proc.sel_region_xr(ds,bbox)    # Crop region
    
    return ds

#%% User Edits

datpath  = "/mnt/CMIP6/data/ocean_reanalysis/glorys12v1/"
outpath  = "/stormtrack/data3/glliu/01_Data/02_AMV_Project/03_reemergence/proc/glorys12v1/"
years    = np.arange(1993,2020)
mons     = np.arange(1,13,1)
bbox     = [-100,20,-20,75]

debug    = False

#%% Main Script
nyrs     = len(years)
varnames = ["thetao","zos","uo","vo","mlotst",]
nvars    = len(varnames)

for v in range(nvars):
    varname  = varnames[v]
    keepvars = ["longitude","latitude","depth","time", varname]
    
    for y in range(nyrs):
        
        year    = years[y]
        montime = []
        mon_ds  = []
        for im in range(12): # Begin Month Loop
            
            # Get list of files for the month
            mon   = mons[im]
            flist = glob.glob("%s%s/mercatorglorys12v1_gl12_mean_%04i%02i*.nc" % (datpath,year,year,mon))
            flist.sort()
            ndays = len(flist)
            
            # Open and grab daily data at a given level/region
            daily_ds = []
            for d in tqdm.tqdm(range(ndays)): # Begin Daily Loop
                st         = time.time()
                ds         = xr.open_dataset(flist[d])      # Open NetCDF View
                ds         = crop_ds(ds,keepvars)
                ds         = ds.load()
                if debug:
                    print("Loaded in %.2fs" % (time.time()-st))
                daily_ds.append(ds)
                # <End Daily Loop> --------
            
            # Concatenate daily data and take mean over montyh
            daily_ds = xr.concat(daily_ds,dim='time')
            monavg   = daily_ds.mean('time')
            montime.append(daily_ds.isel(time=0).time.values) # Append first value of the month
            mon_ds.append(monavg)
            print("Completed merging for year %04i, Month %02i..." % (year,mon))
            # <End Month Loop> --------
        
        # Save files
        stsv     = time.time()
        mon_ds   = xr.concat(mon_ds,dim="time") # Concatenate Files
        mon_ds   =  mon_ds.assign_coords({'time':montime}) # Add Time dimension
        savename = "%sglorys12v1_%s_NAtl_%04i.nc" % (outpath,varname,year)
        edict    = proc.make_encoding_dict(mon_ds)
        mon_ds.to_netcdf(savename,encoding=edict)
        print("Saved file for year %04i in %.2fs" % (year,time.time()-stsv))
        # <End Year Loop> --------
    # End variable loop



