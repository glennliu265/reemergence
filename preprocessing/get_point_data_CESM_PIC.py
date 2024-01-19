#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Get Point Data from CESM1 PIC

Based on [get_point_data_stormtrack.py]

Based on get_point_data_stormtrack (copied 2024.01.18)
NOTE: Currently only supports ocean points... will expand to other points

Created on Thu Jan 18 16:13:36 2024

@author: gliu
"""
import xarray as xr
import matplotlib.pyplot as plt
import sys
from tqdm import tqdm
import glob

#%%

# Select the Point
lonf         = 330
latf         = 50

# Indicate Dataset and Path
outpath      = "/stormtrack/data3/glliu/01_Data/02_AMV_Project/03_reemergence/proc/ptdata/"
dataset_name = "CESM1"

exp          = 'FULL_PIC'

#%% Import Custom Packages

# stormtrack
amvpath = "/home/glliu/00_Scripts/01_Projects/00_Commons/" # amv module
scmpath = "/home/glliu/00_Scripts/01_Projects/01_AMV/02_stochmod/stochmod/model/" # scm module

sys.path.append(amvpath)
sys.path.append(scmpath)

from amv import proc,viz
import scm
import amv.loaders as dl

#%% Indicate point and load the directory
# ---------------------------------------
# Note for SSS and ocn vairables need ot make some changes
mnum                = dl.get_mnum()
locfn,loctitle      = proc.make_locstring(lonf,latf)
outdir              = outpath + "%s/" % locfn
proc.makedir(outdir)


#%% Load some atmosphere variables
# # ---------------------------------
vnames  = ["LHFLX",]#"SHFLX","FSNS","FLNS","PRECC","PRECL","PRECSC","PRECSL","TAUX","TAUY"]
renames = [None,]*len(vnames)

# vnames  += ["TS",]
# renames += ["SST",]

for v in range(len(vnames)):
    
    vname = vnames[v]
    vname_rename = renames[v]
    
    # Open view of netcdfs
    dslist    = dl.load_pic(vname,atm=True)
    
    # Restrict to Point and concatenate
    dspt      = [ds.sel(lon=lonf,lat=latf,method='nearest')[vname].load() for ds in dslist]
    dsptall      = xr.concat(dspt,dim='time')
    
    # Rename if option is set
    if vname_rename is not None:
        dsptall = dsptall.rename(vname_rename)
        vname_out = vname_rename
    else:
        vname_out = vname
    
    # For each file..., extract the point data
    encoding_dict = {vname_out:{'zlib':True}}
    
    # Save output
    savename = "%s%s_%s_%s.nc" % (outdir,dataset_name,exp,vname_out,)
    dsptall.to_netcdf(savename,encoding=encoding_dict)

#%% Load the oceanic variables
# ----------------------------

vnames  = ["HBLT",] # Currently supports HBLT, SALT... and that's it?
renames = [None,] * len(vnames)

v                = 0
for v in range(len(vnames)):
    vname        = vnames[v]
    vname_rename = renames[v]
    
    searchdeg=0.2
    atm = False
    
    # Set up Searchpath
    if vname == "SALT":
        datpath = "/stormtrack/data4/glliu/01_Data/CESM1_PIC/SALT/"
    elif vname == "HBLT":
        datpath = "/stormtrack/data4/glliu/01_Data/CESM1_PIC/HBLT/"
    else:   
        print("Warning, variable path not included. Need to update code.")
        datpath = None
    
    # Glob file list
    nclist = glob.glob(datpath+"*.nc")
    nclist.sort()
    nfiles = len(nclist)
    print("Found %i files" % nfiles)
    
    # Start LOOP HERE
    dsptall = []
    for f in tqdm(range(nfiles)):
        
        # Looping by Ensemble member
        ds    = xr.open_dataset(nclist[f])
        dspt  = proc.getpt_pop(lonf,latf,ds,returnarray=False,searchdeg=searchdeg)
        dsptall.append(dspt.load())
    
    # Concatenate
    #test = xr.concat(dsptall,dim="time") # [Ens x Time]
    dsptall = xr.concat(dsptall,dim="time") # [Ens x Time]
    
    # Rename variable if option is set
    if vname_rename is not None:
        dsptall = dsptall.rename(vname_rename)
        vname_out = vname_rename
    else:
        vname_out = vname
    encoding_dict = {vname_out:{'zlib':True}}
    
    # Save output
    savename = "%s%s_%s_%s.nc" % (outdir,dataset_name,exp,vname_out,)
    dsptall.to_netcdf(savename,encoding=encoding_dict)

# #%% ENSO (do this outside in terminal)
# # Computed with scripts from hfcalc
# # ------------------------------------
# ensoloc="/stormtrack/data3/glliu/01_Data/02_AMV_Project/01_hfdamping/hfdamping_HTR/01_PREPROC/enso/"

# #%% Get the computed heat flux damping
# # Computed with scripts from hfcalc
# # ------------------------------------
# flxs = ["LHFLX","SHFLX","FLNS","FSNS","qnet"]

# for v in range(5):
#     if v < 4:
#         continue
#     vname   = flxs[v]
#     datpath = "/stormtrack/data3/glliu/01_Data/02_AMV_Project/01_hfdamping/hfdamping_%s/%s_damping/" % (exp.upper(),vname)
    
#     ds_all = []
#     for e in tqdm(range(42)):
        
#         ncname = "%s%s_hfdamping_ensorem1_detrend1_1920to2006_ens%02i.nc" % (datpath,exp,e+1)
#         ds     = xr.open_dataset(ncname)
#         ds     = ds.sel(lon=lonf,lat=latf,method='nearest').load()
        
#         ds_all.append(ds)
        
#     dsptall=xr.concat(ds_all,dim="ens") # [Ens x Time]
    
#     encoding_dict = proc.make_encoding_dict(dsptall)
#     vname_out     = vname + "_damping"
    
#     # Save output
#     savename = "%s%s_%s_%s.nc" % (outdir,dataset_name,exp,vname_out,)
#     dsptall.to_netcdf(savename,encoding=encoding_dict)

