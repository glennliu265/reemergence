#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Convert NPZ files to DataSets
For pointwise autocorrelation, CESM1

Works only on Astraeus
Copied from basinwide_T2_osm.py

Created on Wed Mar 13 16:55:07 2024

@author: gliu
"""


import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import sys
from tqdm import tqdm
import copy
import glob

import matplotlib as mpl

#%% Import Custom Modules
amvpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/03_Scripts/" # amv module
scmpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/03_Scripts/stochmod/model/"

sys.path.append(amvpath)
sys.path.append(scmpath)

from amv import proc,viz
import scm
import amv.loaders as dl
import yo_box as ybx


#%% Figure Path

datpath      = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/proc/"
figpath      = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/01_hfdamping/02_Figures/20240216/"

output_path  = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/sm_experiments/"
proc.makedir(figpath)

#%% Set Experiment Names

expnames = [
    "SM_SST_OSM_Tddamp",
    "SM_SST_OSM_Tddamp_noentrain",
    "SM_SSS_OSM_Tddamp",
    "SM_SSS_OSM_Tddamp_noentrain"]
vnames   = ["SST",
            "SST",
            "SSS",
            "SSS"]

expnames_long = [
    "Stochastic Model, Entraining",
    "Stochastic Model, Non-entraining",
    "Stochastic Model, Entraining",
    "Stochastic Model, Non-entraining"
    
    ]
nexps   = len(expnames)
ldnames = [ "%s%s_%s_autocorrelation_thresALL_lag00to60.nc" % (datpath,expnames[ex],vnames[ex],) for ex in range(nexps)]


cesm_files = ["HTR-FULL_SST_autocorrelation_thres0_lag00to60.npz","HTR-FULL_SSS_autocorrelation_thres0_lag00to60.npz"]

ldsst = np.load(datpath+cesm_files[0],allow_pickle=True)
ldsss = np.load(datpath+cesm_files[1],allow_pickle=True)
print(ldsst.files)
print(ldsss.files)

#%%  Load data for stochastic model

ds_sm   = [xr.open_dataset(ldnames[nn])[vnames[nn]] for nn in range(nexps)]
acfs_sm = [ds.values.squeeze() for ds in ds_sm] # [lon x lat x kmonth x thres x lag]

t2_sm   = [proc.calc_T2(acf,axis=3) for acf in acfs_sm] # [(65, 69, 12)]

#%% Load data for CESM1

ld_cesm = [np.load(datpath+cesm_files[ii],allow_pickle=True) for ii in range(2)]
acfs_cesm = [ld['acs'][:,:,:,:,-1,:] for ld in ld_cesm] # [ lon x lat x ens x mon x thres x lag], subset threshold -1

t2cesm = [proc.calc_T2(acf,axis=-1) for acf in acfs_cesm]


# Place into dataarray
ld     = ld_cesm[0]
coords = dict(lon=ld['lon'],lat=ld['lat'],ens=np.arange(1,43,1),mons=np.arange(1,13,1),lags=ld['lags'])
acfscesm = [xr.DataArray(vv,coords=coords,dims=coords) for vv in acfs_cesm]


#%% Save DataArray

vvout = ["SST","SSS"]
for vv in range(2):
    
    # Set up output DataArray
    ds_in        = acfscesm[vv].rename(vvout[vv])
    edict        = {vvout[vv]:{"zlib":True}}
    
    # Set up output Paths
    metrics_path = "%s%s_CESM/Metrics/" % (output_path,vvout[vv])
    savename_vv  = "%sPointwise_Autocorrelation_thresALL_lag0to60.nc" % metrics_path
    
    # Save output
    ds_in.to_netcdf(savename_vv,encoding=edict)
    
    
