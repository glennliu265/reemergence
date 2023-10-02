#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Preprocess Data By Ensemble

- Slice to Region
- Take Monthly Anomalies
- Remove Ensemble Average

Copy of [preprocess_data] but for CESM2.
Tried to make it general enough such that it works for other datasets

Created on Thu Sep 28 11:08:57 2023

@author: gliu
"""

import time
import numpy as np
import xarray as xr
import glob
import scipy
from tqdm import tqdm
import matplotlib.pyplot as plt
import sys

#%% Import Custom Modules
amvpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/03_Scripts/" # amv module
scmpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/03_Scripts/stochmod/model/"

sys.path.append(amvpath)
sys.path.append(scmpath)

from amv import proc,viz
import scm
import amv.loaders as dl

#%% Data Information

# Information and Names
dataset       = "CESM2"
varname       = "SST" # "HMXL"
keepvars      = [varname,'lon','lat','time']

# Data Path
datpath       = "/Users/gliu/Globus_File_Transfer/CESM2_LE/1x1/%s/" % varname
outpath       = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/proc/CMIP6/"

# Set Bounding Box
bbox          = [-80,0,0,65] # Set Bounding Box
bboxfn        = "lon%ito%i_lat%ito%i" % (bbox[0],bbox[1],bbox[2],bbox[3])

# Set Time Period
start         = '1850-01-01'
end           = '2014-12-31'

# Additional options
savesep       = False # Set to True to save each ensemble member separately
save_ensmean  = True  # Set to True to save the ensemble mean
#%% Get the list of files for dataset


nclist = glob.glob(datpath+"*.nc")
nclist.sort()
print(nclist)
nfiles = len(nclist)
print("Found %i files!" % nfiles)

#%%


dsa_all = []
for f in tqdm(range(nfiles)):
    
    # Load DS
    ds = xr.open_dataset(nclist[f])
    
    # Preprocess DS to [time x lat x lon]
    ds = proc.ds_dropvars(ds,keepvars) # Drop Un-needed dimensions
    ds = ds.squeeze(drop=True) # Drop z_t + singleton dims
    ds = proc.format_ds(ds,verbose=False) # Flip longitude and latitude
    
    # Select Region
    dsreg = proc.sel_region_xr(ds,bbox)
    
    # Select Time
    dsreg = dsreg.sel(time=slice(start,end))
    ystart = dsreg.time.values[0].year
    yend   = dsreg.time.values[-1].year
    if f == 0:
        print("Subset time to between %s and %s" % (dsreg.time.values[0],dsreg.time.values[-1]))
    
    # Load
    ds   = dsreg.load()
    
    # Compute Monthly Anomalies
    dsa  = proc.xrdeseason(ds)
    
    # Append
    dsa_all.append(dsa.copy())


# Compute and Save the Ensemble Average
dsa_allens  = xr.concat(dsa_all,dim="ens") # [ens x time x lat x lon]

# Make additional strings to saving
timestr = "%sto%s" % (ystart,yend)

# Save ensemble mean
ensmean     = dsa_allens.mean('ens')
if save_ensmean:
    savename_ensmean = "%s%s_%s_%s_ensmean.nc"% (outpath,dataset,varname,timestr,)
    ensmean.to_netcdf(savename_ensmean)

    
# Remove ensemble mean and save
dsa_dt = dsa_allens - ensmean
encodedict = proc.make_encoding_dict(dsa_dt)
if savesep is False:
    st = time.time()
    savename = "%s%s_%s_%s.nc"% (outpath,dataset,varname,timestr,)
    dsa_dt.to_netcdf(savename,encoding=encodedict)
    print("Saved to %s in %.2fs" % (savename,time.time()-st))
    

