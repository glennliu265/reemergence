#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Combine Precip Data (Large scale and small scale, snow and liquid)
Pi Control
Created on Wed Jan  3 14:20:35 2024

# Note that data are not anomalized.

@author: gliu

"""
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import sys
import glob

import tqdm
import time

#%% Import Custom Modules
amvpath = "/home/glliu/00_Scripts/01_Projects/00_Commons/" # amv module
scmpath = "/home/glliu/00_Scripts/01_Projects/01_AMV/02_stochmod/stochmod/model/" # scm module

sys.path.append(amvpath)
sys.path.append(scmpath)

from amv import proc,viz
import scm
import amv.loaders as dl

#%% User Inputs

varnames    = ("PRECC","PRECL","PRECSC","PRECSL")
nvars = len(varnames)

#b.e11.B1850C5CN.f09_g16.005.cam.h0.PRECL.040001-049912.nc   

datpath     = "/vortex/jetstream/climate/data1/yokwon/CESM1_LE/downloaded/atm/proc/tseries/monthly/"
scenariostr = "b.e11.B1850C5CN.f09_g16.005.cam.h0.*.nc" 
scenariofn  = "PIC_FULL"

outpath     = "/stormtrack/data3/glliu/01_Data/02_AMV_Project/02_stochmod/PRECIP/"

bbox          = [-80,0,0,65]
bboxfn        = "lon%ito%i_lat%ito%i" % (bbox[0],bbox[1],bbox[2],bbox[3])

#%% 


#%% Load and concatenate precip fields by time, crop to North Atlantic Region



for v in range(nvars): # Took around 20 min per iteration
    st = time.time()
    
    # Find a list of files
    vname     = varnames[v]
    searchstr =  "%s/%s/%s" % (datpath,vname,scenariostr)
    flist     = glob.glob(searchstr)
    flist.sort()
    nfiles    = len(flist)
    print("Found %i files!" % (nfiles))
    
    # Crop to Region and Combine
    keepvars  = [vname,"lon","lat","time"]
    ds_var    = []
    for f in tqdm.tqdm(range(nfiles)):
        
        # Open and reformat DS
        ds = xr.open_dataset(flist[f])
        ds = proc.ds_dropvars(ds,keepvars)
        ds = proc.format_ds(ds)
        
        # Select a Region
        dsreg = proc.sel_region_xr(ds,bbox).load()
        
        # Append
        ds_var.append(dsreg)
    
    # Next: Write Code to concatenate and save
    ds_out = xr.concat(ds_var,dim='time')
    
    # Save output
    savename = "%s%s_%s.nc" % (outpath,vname,scenariofn)
    edict    = proc.make_encoding_dict(ds_out)
    ds_out.to_netcdf(savename,encoding=edict)
    print("Combined %s in %.2fs" % (vname,time.time()-st))
    
        
#%% Reload the files and combine them.

stc = time.time()

precip_ds = []

for v in tqdm.tqdm(range(nvars)):
    
    # Find a list of files
    vname     = varnames[v]
    
    # Save output
    savename = "%s%s_%s.nc" % (outpath,vname,scenariofn)
    ds = xr.open_dataset(savename)[vname].load()
    precip_ds.append(ds)
        

# Combine the precip
for v in range(nvars):
    ds=precip_ds[v]
    if v ==0:
        ds_tot = ds.copy()
    else:
        ds_tot = ds_tot + ds
        

# Check Sum
proc.check_sum_ds(precip_ds,ds_tot,t=1,fmt="%.2e")

def check_sum_ds(add_list,sum_ds,lonf=50,latf=-30,t=0,fmt="%.2f"):
    """
    Check sum of list dataarrays
    
    Parameters
    ----------
    add_list : List of xr.DataArrays that were summed. have lat, lon, time dims
    sum_ds   : DataArray containing the summed result
    lonf,latf: NUMERIC, optional. Lon/Lat indices to check at. The default is 50,-30
    t        : INT, optional. Time indices to check. default is 0

    fmt      : STR, optional. Font Format String.The default is "%.2f".
    
    Returns
    -------
    chkstr : STR. Sum Check String
    """
    # Get Values
    out_pt  = sum_ds.sel(lon=lonf,lat=latf,method='nearest').isel(time=t).values
    list_pt = [ds.sel(lon=lonf,lat=latf,method='nearest').isel(time=t).values for ds in add_list]
    vallist = list_pt + [np.array(list_pt).sum(),out_pt] # List of values
    
    # Make Format String
    fmtstr = ""
    for ii in range(len(list_pt)):
        fmtstr += "%s + " % fmt
    fmtstr = fmtstr[:-2] # Drop last addition sign
    fmtstr += "= %s (obtained %s)" % (fmt,fmt)
    
    # Make Check String
    chkstr  = fmtstr % tuple(vallist)
    print(chkstr)
    return chkstr

# Rename the DataArray
ds_tot = ds_tot.rename("PRECTOT")
edict  = {'PRECTOT':{'zlib':True}}
savenametot = "%s%s_%s.nc" % (outpath,"PRECTOT",scenariofn)
ds_tot.to_netcdf(savenametot,encoding=edict)

print("Combined files in %.2f" % (time.time()-stc))

    
    
    