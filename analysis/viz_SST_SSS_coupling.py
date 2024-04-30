#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Visualize output from run_SSS_pointmode_coupled

Also explores relationships between hi-pass filtering and SST-SSS Cross Correlation


(1) Load in stochastic model output (Pointmode, Couped), SPG Point
(2) Load in the corresponding CESM1 Data
(3) Calculate the SST-SSS Lagged Cross Correlation (Monthly, and All Months)


Created on Mon Apr 29 14:55:09 2024

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

# Paths and Experiment
expname      = "SST_SSS_LHFLX_NoLbdd" # Borrowed from "SST_EOF_LbddCorr_Rerun"
metrics_path = output_path + expname + "/Metrics/" 
exp_output   = output_path + expname + "/Output/" 

vnames       = ["SST","SSS"]

#%% Load the variables


# For some reason, 2 lat values are saved for SSS (50 and 50.42). 
# Need to fix this
ds_all = []
var_all = []
for vv in range(2):
    
    globstr = "%s%s_runid*.nc" % (exp_output,vnames[vv])
    nclist  = glob.glob(globstr)
    nclist.sort()
    ds      = xr.open_mfdataset(nclist,combine='nested',concat_dim="run").load()
    
    if len(ds.lat) > 1: # Related to SSS error...
        remake_ds = []
        for nr in range(len(ds.run)):
            invar = ds.isel(run=nr)[vnames[vv]]
            
            if np.all(np.isnan(invar.isel(lat=0))): 
                klat = 1
            if np.all(np.isnan(invar.isel(lat=1))):
                klat = 0
            print("Non-NaN Latitude Index was %i for run %i" % (klat,nr))
            invar = invar.isel(lat=klat)
            #invar['lat'] = 50.
            remake_ds.append(invar.values.copy())
        coords = dict(run=ds.run,time=ds.time)
        ds     = xr.DataArray(np.array(remake_ds).squeeze(),coords=coords,name=vnames[vv])
    else:
        ds = ds[vnames[vv]]
    
    #.sel(lat=50.42,method='nearest')
    ds_all.append(ds)
    var_all.append(ds.values.squeeze()) # [Run x Time]
    
var_flat = [v.flatten() for v in var_all]


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






#%% Chedk for NaNs

for vv in range(2):
    for rr in range(10):
        invar = var_all[vv][rr,:]
        if np.any(np.isnan(invar)):
            print("NaN Detected for v=%s, rr=%i" % (vnames[vv],rr))
    
#

#%% Compute the ACFs (auto and cross)


# Compute the Autocorrelation
lags     = np.arange(37)
acfs_all = [scm.calc_autocorr_mon(v,lags,verbose=False,return_da=False) for v in var_flat]#]scm.calc_autocorr(var_flat,lags,)

#%% Plot
xtks   = np.arange(0,37,1)
kmonth = 1
fig,ax = plt.subplots(1,1,constrained_layout=True,figsize=(12,4.5))
ax,_   = viz.init_acplot(kmonth,xtks,lags,ax=ax)

for vv in range(2):
    ax.plot(lags,acfs_all[vv][kmonth,:],label="%s" %  vnames[vv])  
ax.legend()

#%% Compute the cross-correlation (Monthly)

# First, SST Leads SSS
sst_leads = scm.calc_autocorr_mon(var_flat[0],lags,ts1=var_flat[1],return_da=False)
sss_leads = scm.calc_autocorr_mon(var_flat[1],lags,ts1=var_flat[0],return_da=False)

# Merge
leadlag_corr = np.concatenate([np.flip(sss_leads)[:,:-1],sst_leads],axis=1)
leadlags     = np.concatenate([np.flip(-1*lags)[:-1],lags],) 


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

leadlags,leadlagcorr_all = leadlag_corr(var_flat[0],var_flat[1],lags)

# -----------------------------------------------------------------------------
#%% Compute (with CESM Output), All Lags
# -----------------------------------------------------------------------------

# Here is the xarray ufunc version
calc_leadlag    = lambda x,y: proc.leadlag_corr(x,y,lags,corr_only=True)
llcesm = xr.apply_ufunc(
    calc_leadlag,
    cesm_vanom[0],
    cesm_vanom[1],
    input_core_dims=[['time'],['time']],
    output_core_dims=[['lags']],
    vectorize=True,
    )
leadlags     = np.concatenate([np.flip(-1*lags)[:-1],lags],) 
llcesm['lags'] = leadlags

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

#%%


    
fig,ax = plt.subplots(1,1,constrained_layout=True,figsize=(12,4.5))
ax.plot(leadlags,leadlagcorr_all,lw=2.5,label="SST-SSS Cross Correlation")

ax.set_ylabel("Correlation")
ax.set_ylim([-.1,.1])

ax.axvline([0],lw=0.55,c="k",zorder=-3)
ax.axhline([0],lw=0.55,c="k",zorder=-3)

        
        
        
    
    
    



#%% Plot it

fig,ax = plt.subplots(1,1,constrained_layout=True,figsize=(12,4.5))
ax.plot(leadlags,leadlag_corr[kmonth,:],lw=2.5,label="SST-SSS Cross Correlation")

ax.set_ylabel("Correlation")
ax.set_ylim([-.5,.5])

ax.axvline([0],lw=0.55,c="k",zorder=-3)
ax.axhline([0],lw=0.55,c="k",zorder=-3)





#%% Experiment (1) Explore how hipass filtering impacts SST-SSS Lag Correlation

'''
General Procedure

For each cutoff
(1) Filter Timeseries
(2) Compute Lag Correlation
(3) Visualize

'''



hicutoffs      = [6,12,24,48] # Indicate Thresholds
nthres         = len(hicutoffs)
lags           = np.arange(37)

crosscorrs     = [] # [thres][lag]
hipass_bythres = [] # [thres][ens x time]
for th in range(nthres):
    
    
    # (1) Compute Hi Pass Output
    hicutoff  = hicutoffs[th] # In Months
    hipass    = lambda x: proc.lp_butter(x,hicutoff,6,btype='highpass')
    
    cesm_hipass = []
    for vv in range(2):
        hpout = xr.apply_ufunc(
            hipass,
            cesm_vanom[vv],
            input_core_dims=[['time']],
            output_core_dims=[['time']],
            vectorize=True, 
            )
        cesm_hipass.append(hpout)
        
        
    # (2) Compute Cross-Correlations
    
    calc_leadlag    = lambda x,y: leadlag_corr(x,y,lags,corr_only=True)
    cesm_crosscorrs = xr.apply_ufunc(
        calc_leadlag,
        cesm_hipass[0],
        cesm_hipass[1],
        input_core_dims=[['time'],['time']],
        output_core_dims=[['lags']],
        vectorize=True,
        )
    leadlags     = np.concatenate([np.flip(-1*lags)[:-1],lags],) 
    cesm_crosscorrs['lags'] = leadlags
    
    # Append Output
    crosscorrs.append(cesm_crosscorrs)
    hipass_bythres.append(cesm_hipass)



#%% Experiment (2) Explore how lowpass filtering impacts SST-SSS Lag Corration





