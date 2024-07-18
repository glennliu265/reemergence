#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 17:42:00 2024

@author: gliu
"""


import numpy as np
import xarray as xr
import sys
import time
import matplotlib.pyplot as plt


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


#%% Indicate Paths

#figpath   = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/02_Figures/20240411/"
datpath   = pathdict['raw_path']
outpath   = pathdict['input_path']+"forcing/"


def rmse(ds):
    return ((ds**2).sum('mode'))**(0.5)

#%%
    



# Load the Coarsned EOF Analysis
dataset   = "cesm1le_5degbilinear"
dampstr   = "nomasklag1"
rollstr   = "nroll0"
regstr    = "Global"
dp1       = "/stormtrack/data3/glliu/01_Data/02_AMV_Project/01_hfdamping/output/proc/"
nceof     = dp1 + "%s_EOF_Monthly_NAO_EAP_Fprime_%s_%s_%s.nc" % (dataset,dampstr_qnet,rollstr,regstr)

ds1       = xr.open_dataset(nceof).load()

dp2       = "/stormtrack/data3/glliu/01_Data/02_AMV_Project/03_reemergence/proc/CESM1/NATL_proc/"
nceof2    = "EOF_Monthly_NAO_EAP_Fprime_nomasklag1_nroll0_NAtl.nc"
ds2       = xr.open_dataset(dp2+nceof2).load()
#dp1 = ""



#%% First Check the EOF Pattern (This does not seem to be the case)

eof1 = rmse(ds1.eofs)

eof2 = rmse(ds2.eofs)


im  = 1
ens = 0

bbox = [-80,0,20,65]
fig,axs=viz.geosubplots(1,2,)

ax = axs[0]
ax.set_extent(bbox)
ax.coastlines()
eof1.isel(ens=ens,mon=im).plot(vmin=0,vmax=80,ax=ax)#,plt.show()

ax = axs[1]
ax.set_extent(bbox)
ax.coastlines()
eof2.isel(ens=ens,mon=im).plot(vmin=0,vmax=80,ax=ax)#,plt.show()
plt.show()

#%% Next Check the PC

fig,ax = plt.subplots(1,1,figsize=(12,4.5))
ax.plot(ds1.pcs.isel(mode=1,mon=2,ens=1),label="Coarse")
ax.plot(ds2.pcs.isel(mode=1,mon=2,ens=1),label="Original")
ax.legend()

#%% Check the EVAP


dslhflx2 = "CESM1_HTR_FULL_Eprime_timeseries_LHFLXnomasklag1_nroll0_NAtl.nc"
lhori = xr.open_dataset(dp2+dslhflx2).load()

dslhflx1 = "cesm1le_htr_5degbilinear_Eprime_timeseries_cesm1le5degLHFLX_nroll0_Global.nc"
lhnew = xr.open_dataset(dp1+dslhflx1).load()


#%%



fig,ax = plt.subplots(1,1,figsize=(12,4.5))
ax.plot(dslhflx1.Eprime.isel(ens=0).sel(lon=330,lat=50,method='nearest'),label="Coarse")
ax.plot(dslhflx1.Eprime.isel(ens=0).sel(lon=-30,lat=50,method='nearest'),label="Original")
ax.legend()
plt.show()


