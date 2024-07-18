#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure out what is happening with the coarse runs

Created on Fri Jul 12 09:21:06 2024

@author: gliu
"""




import xarray as xr
import numpy as np
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import cartopy.crs as ccrs

import matplotlib.pyplot as plt
import sys
import glob
import os

import tqdm
import time

# ----------------------------------
# %% Import custom modules and paths
# ----------------------------------

# Import re-eergemce parameters

# Indicate the Machine!
machine = "Astraeus"

# First Load the Parameter File
cwd = os.getcwd()
sys.path.append(cwd+ "/..")
import reemergence_params as rparams

# Paths and Load Modules
pathdict = rparams.machine_paths[machine]

sys.path.append(pathdict['amvpath'])
sys.path.append(pathdict['scmpath'])

# Set needed paths
figpath     = pathdict['figpath']
input_path  = pathdict['input_path']
output_path = pathdict['output_path']
procpath    = pathdict['procpath']
rawpath     = pathdict['raw_path']

#%% Import Custom Modules

# Import AMV Calculation
from amv import proc,viz
import amv.loaders as dl

# Import stochastic model scripts
import scm

#%% 

#compare_name = "CESM1LE"

# Indicate files containing ACFs
cesm_name   = "CESM1_1920to2005_%sACF_lag00to60_ALL_ensALL.nc"
vnames      =  ["SST","SSS"] #["SST","SSS","TEMP"]
sst_expname = "SM_SST_EOF_LbddCorr_Rerun_SST_autocorrelation_thresALL_lag00to60.nc"
sss_expname = "SM_SSS_EOF_LbddCorr_Rerun_lbdE_neg_SSS_autocorrelation_thresALL_lag00to60.nc"

#sst_expname = "SM_SST_EOF_LbddCorr_Rerun_SST_autocorrelation_thresALL_lag00to60.nc"
#sss_expname = "SM_SSS_EOF_LbddCorr_Rerun_lbdE_neg_SSS_autocorrelation_thresALL_lag00to60.nc"

# Indicate Experiment Names (copying format from compare_regional_metrics)
comparename     = "CESM_Coarse_Draft1"
expnames        = ["SST_CESM","SSS_CESM","SST_CESM1_5deg_lbddcoarsen_rerun","SSS_CESM1_5deg_lbddcoarsen"]
expvars         = ["SST","SSS","SST","SSS"]
expnames_long   = ["SST (CESM1)","SSS (CESM1)","SST (SM Coarse)","SSS (SM Coarse)"]
expnames_short  = ["CESM_SST","SCESM_SSS","SM5_SST","SM5_SSS"]
ecols           = ["firebrick","navy","hotpink","cornflowerblue"]
els             = ["solid",'solid','dashed','dashed']
emarkers        = ["d","x","o","+"]

#%% Plotting variables

# Plotting Information
bbplot                      = [-80,0,20,65]
mpl.rcParams['font.family'] = 'Avenir'
proj                        = ccrs.PlateCarree()
mons3                       = proc.get_monstr()

# Font Sizes
fsz_title                   = 32
fsz_tick                    = 18
fsz_axis                    = 24
fsz_legend                  = 18

#%%


ds = dl.load_smoutput(expnames[-2],output_path)
expname  = expnames[-2]
vname    = expvars[-2]
#%%

def getfirstnan(ts):
    return np.argmax(np.isnan(ts))

ts = ds.SSS.isel(lat=1,lon=8,run=0).data
getfirstnan(ts)



plt.plot(ds.SSS.isel(lat=1,lon=8,run=0).data)

ds.SSS.isel(run=0,time=20).plot()


#%% Try to get the first NaN for each point

# Apply looping through basemonth, lon, lat. ('lon', 'lat', 'mon', 'rem_year')
firstnan = xr.apply_ufunc(
    getfirstnan,
    ds[vname],
    input_core_dims=[['time']],
    output_core_dims=[[]],
    vectorize=True,
    )


firstnan.isel(run=2).plot()

#%% 

fig,ax=viz.geosubplots()
ax =viz.add_coast_grid(ax=ax,bbox=bbplot)
pcm = ax.pcolormesh(firstnan.lon,firstnan.lat,firstnan.isel(run=0))
fig.colorbar(pcm,ax=ax)
ax.set_title("Time Index of Model Run Failure")


#%% Retrieve points and save them


problem_points = xr.where(firstnan.sum('run')>0,1,0)

ds_out = xr.merge([problem_points,firstnan.rename("time_index")])


savename = "%s%s/Metrics/Problem_Points_Debug_20240712.nc" % (output_path,expname,)
ds_out.to_netcdf(savename)
