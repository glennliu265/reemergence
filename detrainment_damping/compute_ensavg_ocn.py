#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Compute the ensemble average of ocean variables
Works with output produced in [process_bylevel_ens]

Created on Wed Feb 14 13:35:22 2024

@author: gliu
"""

import xarray as xr
import numpy as np
import glob
import time

#%% User edits

datpath = "/stormtrack/data3/glliu/01_Data/02_AMV_Project/03_reemergence/proc/CESM1/NATL_proc/ocn_var_3d/"
vnames  = ["SALT","TEMP"]
nens    = 42

#%% generate filelists

v      = 0
vname  = vnames[v]

# Open Datasets (took 534.99s)
st     = time.time()
nclist = ["%s%s_NATL_ens%02i.nc" % (datpath,vname,e+1) for e in range(nens)]
ds_all = xr.open_mfdataset(nclist,concat_dim='ens',combine='nested').load() # (ens, time, z_t, nlat, nlon)
print("Opened data view in %.2fs" % (time.time()-st))

# First, check if any of the ensembles are all NaN
ntime = len(ds_all.time)
chknan = []
for t in range(ntime):
    ds_in = ds_all.sum(('time','z_t'),skipna=False)
    chknan.append(np.all(np.isnan(ds_in)))

ds_sum = ds_all.sum(('time','z_t'))

# Take Ensemble mean
st = time.time()
ds_ensmean = ds_all.mean('ens')
print("Took Ens Mean in %.2fs" % (time.time()-st))

# Save output
savename = "%s%s_NATL_EnsAvg.nc" % (datpath,vname)



