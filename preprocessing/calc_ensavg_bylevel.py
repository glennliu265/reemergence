#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Takes output of process_bylevel_ens
Computes ensemble average.


Created on Mon Feb 26 12:03:35 2024

@author: gliu

"""

import time
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import sys
from tqdm import tqdm
import scipy as sp
import glob
#%%

# stormtrack
amvpath = "/home/glliu/00_Scripts/01_Projects/00_Commons/" # amv module
scmpath = "/home/glliu/00_Scripts/01_Projects/01_AMV/02_stochmod/stochmod/model/" # scm module

sys.path.append(amvpath)
sys.path.append(scmpath)

from amv import proc,viz
import scm
import amv.loaders as dl

#%% User Edits

vname    = "SALT"
outpath  = "/stormtrack/data3/glliu/01_Data/02_AMV_Project/03_reemergence/proc/CESM1/NATL_proc/ocn_var_3d/"
nens     = 42

#%%

# Make the file list
e = 0
fnlist = []
for e in range(nens):
    fn = "%s%s_NATL_ens%02i.nc" % (outpath,vname,e+1)
    fnlist.append(fn)
    
# Open and Take the Ens Avg
ds_all    = xr.open_mfdataset(fnlist,concat_dim='ens',combine='nested')
ds_ensavg = ds_all.mean('ens')

savename = "%s%s_NATL_EnsAvg.nc" % (outpath,vname)
edict    = proc.make_encoding_dict(ds_ensavg)

# Save output
st = time.time()
ds_ensavg.to_netcdf(savename,encoding=edict)
print("Saved ens. avg values in %.2fs" % (time.time()-st))


#%% Debugging: Visualize teh ensemble average values
vnames       = ["TEMP","SALT"]

dsvar        = []
for vv in range(2):
    savename = "%s%s_NATL_EnsAvg.nc" % (outpath,vnames[vv])
    ds = xr.open_dataset(savename).load()
    dsvar.append(ds)

iz    = 0
itime = 0
iv    = 0
test  = dsvar[iv][vnames[iv]].isel(time=itime,z_t=iz)

plt.pcolormesh(test),plt.show(),plt.colorbar()

.plot.pcolormesh(x='nlon',y='nlat'),plt.colorbar(),plt.show()

fig,ax = plt.subplots(1,1)

    
#%%
