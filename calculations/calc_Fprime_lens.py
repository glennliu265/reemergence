#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

calculate_Fprime_lens.py
========================

Given Damping, MLD, SST , and Qnet, compute Fprime where:
    F' = lbd*T + Qnet
    
Where T and Qnet are not anomalized. Written to run on Astraeus...
This script will be used by NHFLX_EOF_monthly.

Inputs:
------------------------

    varname : dims                              - units                 - processing script
    SST     : (ensemble, time, lat, lon)        [degC]                  ????
    qnet    : (ensemble, time, lat, lon)        [W/m2]                  ????
    h       : (mon, ens, lat, lon)              [meters]                ????
    damping : (mon, ens, lat, lon)              [degC/W/m2] OR [1/mon]  ????

Outputs: 
------------------------

    varname : dims                              - units 
    Fprime  : (time, ens, lat, lon)             [W/m2]

Output File Name: "%sCESM1_HTR_FULL_Fprime_timeseries_%s_%s_NAtl.nc" % (rawpath1,dampstr,rollstr)

What does this script do?
------------------------
    1) Load, deseasonalize, and detrend qnet/TS
    2) Load (and optionally convert) HFF
    3) Tile HFF and compute Fprime
    4) Save output

Script History
------------------------
 - Moved from NHFLX_EOF_monthly_lens.py on 2024.02.12
 - Copied Fprime calculation step from preproc_sm_inputs_SSS
 - Created on Mon Feb 12 15:40:17 2024

@author: gliu
"""


import numpy as np
import xarray as xr
import sys
import time
import matplotlib.pyplot as plt

#%% Import Custom Modules
amvpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/03_Scripts/" # amv module
scmpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/03_Scripts/stochmod/model/"

sys.path.append(amvpath)
sys.path.append(scmpath)

from amv import proc,viz
import scm
import amv.loaders as dl
import yo_box as ybx

#%% Set Paths

stormtrack = 0

# Path to variables processed by prep_data_byvariable_monthly, Output will be saved to rawpath1
if stormtrack:
    rawpath1 = "/stormtrack/data3/glliu/01_Data/02_AMV_Project/03_reemergence/proc/CESM1/NATL_proc/"
    dpath    = "/stormtrack/data3/glliu/01_Data/02_AMV_Project/03_reemergence/proc/model_input/damping/"
    mldpath  = "/stormtrack/data3/glliu/01_Data/02_AMV_Project/03_reemergence/proc/model_input/mld/"
else:
    rawpath1 = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/proc/CESM1/NATL_proc/"
    mldpath  = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/proc/model_input/mld/"
    dpath    = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/proc/model_input/damping/"


# Indicate Search String for qnet/SST files ------
ncstr1   = "CESM1LE_%s_NAtl_19200101_20050101_bilinear.nc"

# Indicate Mixed-Layer Depth File
mldnc    = "%sCESM1_HTR_FULL_HMXL_NAtl.nc" % mldpath

# Fprime Calculation Options
nroll    = 0
rollstr  = "nroll%0i"  % nroll

# Damping Options ----------
"""
Current List of Damping Strings
-- Name             -- ncfile                                               -- Description
"None"              "CESM1_HTR_FULL_qnet_damping_nomasklag1.nc"             Default Qnet Damping as calculated from covariance-based method.
"Expfitlbda123"     "CESM1_HTR_FULL_Expfit_lbda_damping_lagsfit123.nc"      Exp Fit to SST - Expfit to SSS; Mean of Lags 1,2,3
"ExpfitSST123"      "CESM1_HTR_FULL_Expfit_SST_damping_lagsfit123.nc"       Exp Fit to SST (total); Mean of Lags 1,2,3
"""
dampstr = None # Damping String  (see "load damping of choice")
if dampstr == "Expfitlbda123":
    convert_wm2=True
    hff_nc   = "CESM1_HTR_FULL_Expfit_lbda_damping_lagsfit123.nc"
elif dampstr == None:
    convert_wm2=False
    hff_nc = "CESM1_HTR_FULL_qnet_damping_nomasklag1.nc"
elif dampstr == "ExpfitSST123":
    convert_wm2=True
    hff_nc   = "CESM1_HTR_FULL_Expfit_SST_damping_lagsfit123.nc"#"CESM1_HTR_FULL_qnet_damping_nomasklag1.nc"
# Conversion Factors
dt  = 3600*24*30
cp0 = 3996
rho = 1026

# -----------------------------------------------------------------------------
#%% Part 1: Load, Deseasonalize, Detrend qnet and SST
# -----------------------------------------------------------------------------
# Note this was copied from preproc_sm_inputs_SSS.py
st       = time.time()

# Load TS, flux and preprocess -------------------------
varnames =['SST','qnet']
ds_load  =[xr.open_dataset(rawpath1+ ncstr1 % vn).load() for vn in varnames]

# Anomalize
ds_anom  = [proc.xrdeseason(ds) for ds in ds_load]

# Detrend
ds_dt    = [ds-ds.mean('ensemble') for ds in ds_anom] # [ens x time x lat x lon]

# Transpose to [mon x ens x lat x lon]
ds_dt    = [ds.transpose('time','ensemble','lat','lon') for ds in ds_dt]

# -----------------------------------------------------------------------------
#%% Part 2: Load and Convert Damping
# -----------------------------------------------------------------------------

# Load HFF
dshff    = xr.open_dataset(dpath + hff_nc) # [mon x ens x lat x lon]

# Load mixed layer depth for conversion
ds_mld   = xr.open_dataset(mldnc)

# Check sizes, make sure they are all the same...
if dampstr is not None: # Not sure why, but it seems that the hff default is wrongly cropped
    ds_list = ds_dt + [dshff,ds_mld]
    ds_rsz  = proc.resize_ds(ds_list)
    ds_dt = ds_rsz[:2]
    dshff = ds_rsz[2]
    ds_mld = ds_rsz[3]

# Convert HFF (1/mon to W/m2 per degC) if needed
if convert_wm2:

    dshff = dshff.damping * (rho*cp0*ds_mld.h) / dt  *-1 #need to do a check for - value!!
else:
    dshff= dshff.damping

# Load output to numpy
hff     = dshff.values
sst     = ds_dt[0].SST.values
qnet    = ds_dt[1].qnet.values

# -----------------------------------------------------------------------------
#%% Part 3: Tile heat flux feedback and make Fprime 
# -----------------------------------------------------------------------------
ntime,nens,nlat,nlon        = qnet.shape # Check sizes and get dimensions for tiling
ntimeh,nensh,nlath,nlonh    = hff.shape
nyrs                        = int(ntime/12)
hfftile                     = np.tile(hff.transpose(1,2,3,0),nyrs)
hfftile                     = hfftile.transpose(3,0,1,2)
# Check plt.pcolormesh(hfftile[0,0,:,:]-hfftile[12,0,:,:]),plt.colorbar(),plt.show()

#% Calculate F'
Fprime   = qnet + hfftile*np.roll(sst,nroll)

# -----------------------------------------------------------------------------
#%% Part 4: Save Fprime output (full timeseries) (Optional)
# -----------------------------------------------------------------------------
coords   = dict(time=ds_dt[0].time.values,ens=dshff.ens.values,lat=dshff.lat.values,lon=dshff.lon.values)
daf      = xr.DataArray(Fprime,coords=coords,dims=coords,name="Fprime")
savename = "%sCESM1_HTR_FULL_Fprime_timeseries_%s_%s_NAtl.nc" % (rawpath1,dampstr,rollstr)
edict    = {"Fprime":{'zlib':True}}
daf.to_netcdf(savename,encoding=edict)
print("Script ran to completion in %.2fs" % (time.time()-st))
