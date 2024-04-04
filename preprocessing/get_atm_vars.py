#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Get point data for target variables.
Works with data preprocessed with prep_data_byvariable_monthly
Copied upper section from pointwise_crosscorrelation



Currently Functions on stormtrack

Created on Tue Apr  2 14:59:17 2024

@author: gliu


"""

import sys
import time
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from tqdm import tqdm

#%%



#%% User Edits

stormtrack   = True

# # Dataset Parameters <General Settings>
# # ---------------------------
outname_data = "CESM1_1920to2005_FprimeACF_nomasklag1_nroll0"
vname_base   = "Fprime"
vname_lag    = "Fprime"
nc_base      = "CESM1_HTR_FULL_Fprime_timeseries_nomasklag1_nroll0_NAtl.nc" # [ensemble x time x lat x lon 180]
nc_lag       = "CESM1_HTR_FULL_Fprime_timeseries_nomasklag1_nroll0_NAtl.nc" # [ensemble x time x lat x lon 180]

datpath      = "/stormtrack/data3/glliu/01_Data/02_AMV_Project/03_reemergence/proc/CESM1/NATL_proc/"
preprocess   = True # If True, demean (remove ens mean) and deseason (remove monthly climatology)

#%% Set Paths for Input (need to update to generalize for variable name)


if stormtrack:
    # Module Paths
    sys.path.append("/home/glliu/00_Scripts/01_Projects/00_Commons/")
    sys.path.append("/home/glliu/00_Scripts/01_Projects/01_AMV/02_stochmod/stochmod/model/")
else:
    # Module Paths
    sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/03_Scripts/")
    sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/03_Scripts/stochmod/model/")

# Import modules
from amv import proc,viz
import scm

#%%

# Indicate variables to grab
vnames = ["SST",
          "qnet",
          "Fprime",
          "Umod"]

# Indicate Location
lonf           = -30
latf           = 50
locfn,loctitle = proc.make_locstring(lonf,latf,lon360=True)




#%% Some Functions


def chk_dimnames(ds,longname=False):
    if longname:
        if "ens" in ds.dims:
            ds = ds.rename({'ens':'ensemble'})
    else:
        if "ensemble" in ds.dims:
            ds = ds.rename({'ensemble':'ens'})
    return ds


#%% 



nvars = len(vnames)
dsall = []
for vv in range(nvars):
    
    vname = vnames[vv]
    
    
    if vname == "Fprime": # Load data that has been processed by pointwise_crosscorrelation
        nc = datpath+"CESM1_HTR_FULL_Fprime_timeseries_nomasklag1_nroll0_NAtl.nc"
    else:
        nc = datpath+"CESM1LE_%s_NAtl_19200101_20050101_bilinear.nc" % vname
    
    # Load dataset, change dims
    ds   = xr.open_dataset(nc)[vname].load()
    ds   = chk_dimnames(ds,longname=False)
    dspt = proc.selpt_ds(ds,lonf,latf).load()
    dsall.append(dspt)
    
    
    
def replace_dim(ds,dimname,dimval):
    ds = ds.assign_coords({dimname:dimval})
    return ds
    
ensdim = np.arange(1,43,1)
dsall  = [replace_dim(ds,'ens',ensdim) for ds in dsall]
dsall  = [ds.transpose('ens','time') for ds in dsall]
dsout  = xr.merge(dsall)

savename = "%sCESM1_HTR_FULL_NAtl_AtmoVar_%s.nc" % (datpath,locfn)
edict     = proc.make_encoding_dict(dsout)
dsout.to_netcdf(savename,encoding=edict)
print("Saved output to %s" % savename)


    
    
    