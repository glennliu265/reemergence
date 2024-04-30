#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Debug LeadLagCorr (using CESM Data at SPG Point)

Copied xarray Debug script from viz_SST_SSS_coupling

Created on Tue Apr 30 15:22:57 2024

@author: gliu
"""


import xarray as xr
import numpy as np
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

"""
SSS_EOF_Qek_Pilot

Note: The original run (2/14) had the incorrect Ekman Forcing and used ens01 detrainment damping with linear detrend
I reran this after fixing these issues (2/29)

"""

vnames       = ["SST","SSS"]

#%% Load in CESM1 For comparison


cesm_vars = []
for vv in range(2):
    ncname = "%sCESM1LE_%s_NAtl_19200101_20050101_bilinear.nc" % (rawpath,vnames[vv])
    ds     = xr.open_dataset(ncname).sel(lon=-30,lat=50,method='nearest')[vnames[vv]].load()
    cesm_vars.append(ds)
    
# Deseasonalize, Anomalize
def preproc_cesm(ds):
    dsdt = ds - ds.mean('ensemble')
    dsda = proc.xrdeseason(dsdt)
    dsda = dsda.rename({"ensemble":"ens"})
    return dsda

cesm_vanom = [preproc_cesm(ds) for ds in cesm_vars]

#%% Do some filtering (this could be a good testing script)
#codetemplate

hicutoff  = 12 # In Months
hipass    = lambda x: proc.lp_butter(x,hicutoff,6,btype='highpass')

locutoff  = 60 # In Months
lopass    = lambda x: proc.lp_butter(x,locutoff,6,btype='lowpass')

filtername = "filter_hi%02i_low%02i" % (hicutoff,locutoff)

cesm_hipass = []
cesm_lopass = []
for vv in range(2):
    hpout = xr.apply_ufunc(
        hipass,
        cesm_vanom[vv],
        input_core_dims=[['time']],
        output_core_dims=[['time']],
        vectorize=True, 
        )
    cesm_hipass.append(hpout)
    
    lpout = xr.apply_ufunc(
        lopass,
        cesm_vanom[vv],
        input_core_dims=[['time']],
        output_core_dims=[['time']],
        vectorize=True, 
        )
    cesm_lopass.append(lpout)

# Check Output
e = 0
v = 0
fig,ax = plt.subplots(1,1,constrained_layout=True,figsize=(8,3))

ax.plot(cesm_vanom[v][e,:],label="raw",color='gray')
ax.plot(cesm_hipass[v][e,:],label="highpass (%i months)" % hicutoff,color='r',lw=0.75)
ax.plot(cesm_lopass[v][e,:],label="lowpass (%i months)" % locutoff,color='b',lw=1.5)

ax.set_title("Low and Hi-Pass Test (%s, ens=%02i)" % (vnames[v],e+1))
ax.legend(ncol=3)
ax.set_xlim([0,1032])
ax.set_ylabel("%s Anomaly" % (vnames[v]))
ax.set_xlabel("Months")

savename = "%sFilter_Test_%s_Ens%02i_%s.png" % (figpath,vnames[v],e+1,filtername)
plt.savefig(savename,dpi=150)

#%% Try the overall leadlag (Daily)

# Compute SST Leading

def leadlag_corr(varbase,varlag,lags,corr_only=False):
    ntime = varbase.shape[0]
    nlags = len(lags)
    # Lags
    leadcorrs = []
    lagcorrs  = []
    for ii in range(nlags):
        lag     = lags[ii]
        lagcorr  = np.corrcoef(varbase[:(ntime-lag)],varlag[lag:])[0,1]
        leadcorr = np.corrcoef(varbase[lag:],varlag[:(ntime-lag)])[0,1]
        lagcorrs.append(lagcorr)
        leadcorrs.append(leadcorr)
    leadlagcorr = np.concatenate([np.flip(leadcorrs)[:-1],lagcorrs])
    leadlags    = np.concatenate([np.flip(-1*lags)[:-1],lags],)
    if corr_only:
        return leadlagcorr
    return leadlags,leadlagcorr

# -----------------------------------------------------------------------------
#%% Test Ufuncs Version (with CESM Output), All Lags
# -----------------------------------------------------------------------------

lags        = np.arange(37)
leadlags    = np.concatenate([np.flip(-1*lags)[:-1],lags],)

# Here is the manual loop (by ens)
llmanual = []
for e in range(42):
    varbase = cesm_vanom[0].isel(ens=e).values
    varlag  = cesm_vanom[1].isel(ens=e).values
    
    _,ll      =  leadlag_corr(varbase,varlag,lags)
    llmanual.append(ll)
llmanual = np.array(llmanual)
coords   = dict(ens=cesm_vanom[0].ens,lag=leadlags)
llmanual = xr.DataArray(llmanual,coords=coords,dims=coords)


# Here is the xarray ufunc version
calc_leadlag    = lambda x,y: leadlag_corr(x,y,lags,corr_only=True)
llufunc = xr.apply_ufunc(
    calc_leadlag,
    cesm_vanom[0],
    cesm_vanom[1],
    input_core_dims=[['time'],['time']],
    output_core_dims=[['lags']],
    vectorize=True,
    )
leadlags     = np.concatenate([np.flip(-1*lags)[:-1],lags],) 
llufunc['lags'] = leadlags

#%% Compare ufunc vs manual loop

e = 4

fig,ax = plt.subplots(1,1,constrained_layout=True,figsize=(8,4))
ax.plot(leadlags,llmanual.isel(ens=e),lw=2.5,label="Manual Loop")
ax.plot(leadlags,llufunc.isel(ens=e),lw=2,ls='dashed',label="xr.ufuncs")
ax.legend()

ax.set_ylabel("Correlation")
ax.set_xlabel("<--- SSS Leads SST | SSS Lags SST --->")
ax.set_ylim([-.5,.5])

ax.axvline([0],lw=0.55,c="k",zorder=-3)
ax.axhline([0],lw=0.55,c="k",zorder=-3)
ax.set_title("Manual Loop vs. xr.ufuncs Application of leadlag_corr")

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
