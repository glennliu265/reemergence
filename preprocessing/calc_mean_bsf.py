#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calc_Mean_BSF

Using data from [predict_nasst/regrid_ocean_variable.py], compute the monthly mean
BSF and SSH to get a sense of the mean circulation in the model...

(time, lat, lon)

Created on Wed Mar  6 15:26:33 2024

@author: gliu

"""

import numpy as np
import xarray as xr
import sys
import time
import matplotlib.pyplot as plt
import matplotlib as mpl
import cartopy.crs as ccrs
import os
from tqdm import tqdm

# ----------------------------------
#%% Import custom modules and paths
# ----------------------------------

# Indicate the Machine!
machine = "stormtrack"

# First Load the Parameter File
sys.path.append("../")
import reemergence_params as rparams

# Paths and Load Modules
pathdict   = rparams.machine_paths[machine]

sys.path.append(pathdict['amvpath'])
sys.path.append(pathdict['scmpath'])
from amv import proc,viz
import scm
import amv.loaders as dl
import yo_box as ybx

# Set needed paths
figpath     = pathdict['figpath']
proc.makedir(figpath)
input_path  = pathdict['input_path']
output_path = pathdict['output_path']
procpath    = pathdict['procpath']

#%% User Edits

# Note, this actually only runs on stormtrack for now...
datpath = "/stormtrack/data3/glliu/01_Data/04_DeepLearning/CESM_data/CESM1_Ocean_Regridded/"
outpath = "/home/glliu/01_Data/"
regrid  = "bilinear"
nens    = 42

vnames   = ["BSF","SSH"]


#searchstr = "%s%s_HTR_%s_regridded_ens%02i.nc" % (datpath,vname,regrid,e+1)

#%% Take the ensemble mean

v     = 1
# Eventually write this into a loop....

vname  = vnames[v]
st     = time.time()
fns = []
ds_all = []
for e in tqdm(range(nens)):
    ncname = "%s%s_HTR_%s_regridded_ens%02i.nc" % (datpath,vname,regrid,e+1)
    fns.append(ncname)
    
    ds = xr.open_dataset(ncname).load()
    ds_all.append(ds)
    
ds_all = xr.concat(ds_all,dim='ens')
# Load netCDFs and compute monthly means

#ds_all = xr.open_mfdataset(fns,concat_dim='ens',combine='nested').load() # Note can remove load and parallelize better next time...
print("Loaded data in %.2fs" % (time.time()-st))

# Compute the monthly mean
ds_all_scycle=ds_all.groupby('time.month').mean('time')
ds_all_scycle = ds_all_scycle.transpose('ens','month','lat','lon').rename(dict(month='mon'))

# Flip the longitude
ds_all_scycle180 = proc.lon360to180_xr(ds_all_scycle)

savename         = "%sCESM1_HTR_%s_%s_regridded_AllEns.nc" % (outpath,vname,regrid)
edict            = proc.make_encoding_dict(ds_all_scycle180)
ds_all_scycle180.to_netcdf(savename,encoding=edict)

# Do same for ensemble average
savename_ensavg  = "%sCESM1_HTR_%s_%s_regridded_EnsAvg.nc" % (outpath,vname,regrid)
ds_out_ensavg    = ds_all_scycle180.mean('ens')
ds_out_ensavg.to_netcdf(savename_ensavg,encoding=edict)
print("Completed calculations in %.2fs" % (time.time()-st))
