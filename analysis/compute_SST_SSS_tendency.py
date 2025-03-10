#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Compute dS'/dt and dT'/dt

copied upper section from compute_mld_variability_term.py

Created on Thu Mar  6 11:40:32 2025

@author: gliu
"""

import xarray as xr
import numpy as np
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import cartopy.crs as ccrs
import scipy as sp

import matplotlib.pyplot as plt
import sys
import glob
import os

import tqdm
import time

#from cmcrameri import cm
import matplotlib.patheffects as pe

# ----------------------------------
# %% Import custom modules and paths
# ----------------------------------

# Import re-eergemce parameters

# Indicate the Machine!
machine = "stormtrack"

# First Load the Parameter File
cwd             = os.getcwd()
sys.path.append(cwd + "/..")

# Paths and Load Modules
import reemergence_params as rparams
pathdict        = rparams.machine_paths[machine]

sys.path.append(pathdict['amvpath'])
sys.path.append(pathdict['scmpath'])

# Set needed paths
figpath         = pathdict['figpath']
input_path      = pathdict['input_path']
output_path     = pathdict['output_path']
procpath        = pathdict['procpath']
rawpath         = pathdict['raw_path']

# %% Import Custom Modules

from amv import proc, viz
import scm
import amv.xrfunc as xrf
import amv.loaders as dl

# Import stochastic model scripts
proc.makedir(figpath)

#%% Load Plotting Parameters
bboxplot    = [-80,0,20,60]

#%% Indicate Dataset to Filter

vnames = ["SST","SSS",]#"HMXL"]
ds_all = []
for vv in range(2):
    st = time.time()
    
    ncname  = rawpath + "CESM1LE_%s_NAtl_19200101_20050101_bilinear.nc" % vnames[vv]
    ds      = xr.open_dataset(ncname)[vnames[vv]].load()
    ds_all.append(ds)
    print("Loaded %s in %.2fs" % (vnames[vv],time.time()-st))


#%% Compute the anomalies

ds_anom         = [proc.xrdeseason(ds) for ds in ds_all]
#ds_tend         = [ds.differentiate('time',datetime_unit='m') for ds in ds_anom] # <!!> Note, m is minute....
ds_detrend      = [ds - ds.mean('ensemble') for ds in ds_anom]
ds_dt           = [ds.differentiate('time') for ds in ds_detrend]

dtmon           = 60*60*24*30
ds_dt_mon       = [ds * dtmon for ds in ds_dt]
Tprime,Sprime   = ds_detrend
dTdt,dSdt       = ds_dt_mon

# # Convert to monthly step
# def get_seconds_bymonth(ts):
#     ndays = np.array([31,28,31,30,31,30,31,31,30,31,30,31])
#     dtmon = ndays * 60*60*24
#%% Compute Pointwise Cross Correlation
# First, lets save the output

nens,ntime,nlat,nlon = dSdt.shape


crosscorr_all = np.zeros((nens,nlat,nlon)) * np.nan

# Do a silly loop...
for a in tqdm.tqdm(range(nlat)):
    
    for o in range(nlon):
        
        dSdt_pt     = dSdt.isel(lat=a,lon=o).data
        Tprime_pt   = Tprime.isel(lat=a,lon=o).data
        
        
            
        for e in range(nens):
            
            x_in = dSdt_pt[e,:]
            y_in = Tprime_pt[e,:]
            
            if np.any(np.isnan(x_in)) or np.any(np.isnan(y_in)):
                continue
            else:
            
                ccout = np.corrcoef(x_in,y_in)[0,1]
                crosscorr_all[e,a,o] = ccout.copy()
            
        
#%% Plot the cross-correlation


proj            = ccrs.PlateCarree()
fig,ax,mdict    = viz.init_orthomap(1,1,bboxplot=bboxplot,figsize=(16,12))
ax              = viz.add_coast_grid(ax,bbox=bboxplot)


plotvar         = np.nanmean(crosscorr_all,0)

pcm             = ax.pcolormesh(Tprime.lon,Tprime.lat,plotvar,transform=proj,
                         vmin=-1,vmax=1,cmap='cmo.balance')
cb              = viz.hcbar(pcm,ax=ax)


plt.show()

#%% Save some output

# Save the Tendency

vname         = 'dSprime_dt'
ds_dSprime_dt = dSdt.rename(vname)
outname       = "%sCESM1LE_%s_NAtl_19200101_20050101_bilinear.nc" % (rawpath,vname)
edict         = proc.make_encoding_dict(ds_dSprime_dt)
ds_dSprime_dt.to_netcdf(outname,encoding=edict)

vname         = 'dTprime_dt'
ds_dTprime_dt = dTdt.rename(vname)
outname       = "%sCESM1LE_%s_NAtl_19200101_20050101_bilinear.nc" % (rawpath,vname)
edict         = proc.make_encoding_dict(ds_dTprime_dt)
ds_dTprime_dt.to_netcdf(outname,encoding=edict)

#%% Sanity Check (centered vs forward difference at a point)

testpt = [proc.selpt_ds(ds_anom[0],lonf=-30,latf=50).isel(ensemble=0),
          proc.selpt_ds(ds_dt[0],lonf=-30,latf=50).isel(ensemble=0)]

t_1    = testpt[0].data[1:]
t_0    = testpt[0].data[:(-1)]

t_diff = t_1 - t_0

t_func = testpt[1] * dtmon
fig,ax = plt.subplots(1,1)
ax.plot(t_diff,c="red",label="Forward Difference")
ax.plot(t_func,c="gray",label="Centered Diff (using xr.differentiate)")
ax.set_xlim([100,200])
ax.legend()

#%% Do Same Calculation for Stochastic Model Tendencies

# Indicate and load experiment
expname = "SSS_Draft03_Rerun_QekCorr"
st      = time.time()
ds      = dl.load_smoutput(expname,output_path)
print("Loaded stochastic model output in %.2fs" % (time.time()-st))

#%% Anomalize and detrend
ds_anom         = proc.xrdeseason(ds)
ds_detrended    = proc.xrdetrend(ds_anom.SSS)

#%% Differentiate
dtmon           = 3600*24*30
ds_dt           = ds_detrended.differentiate('time')
ds_dt           = ds_dt * dtmon


#%% Quick Salinity Check (Stochastic Model Differentiation)

lonf   = -30
latf   = 50
testpt = [ds_detrended,ds_dt]
testpt = [proc.selpt_ds(ds,lonf=lonf,latf=latf).isel(run=0) for ds in testpt]

t_1    = testpt[0].data[1:]
t_0    = testpt[0].data[:(-1)]

t_diff = t_1 - t_0

t_func = testpt[1] #* dtmon
fig,ax = plt.subplots(1,1)
ax.plot(t_diff,c="red",label="Forward Difference")
ax.plot(t_func,c="gray",label="Centered Diff (using xr.differentiate)")
ax.set_xlim([100,200])
ax.legend()
plt.show()


#%% Save the output (need to do reverse of load_smoutput)

expname_new     = expname.replace("SSS","dSprime_dt")

# Make new experiment directory with output
expdir          = output_path + expname_new #+ "/Output/"
proc.makedir(expdir)
outdir = expdir + "/Output/"
proc.makedir(outdir)

# Get the nclist
nclist_ori = dl.load_smoutput(expname,output_path,return_nclist=True)
nclist_new = [ncname.replace("SSS","dSprime_dt") for ncname in nclist_ori]

# Save everything
if len(ds_dt.run) == len(nclist_new):
    
    nruns = len(nclist_new)
    
    for ii in tqdm.tqdm(range(nruns)):
        
        outname = nclist_new[ii]
        ds_out  = ds_dt.isel(run=ii)
        ds_out  = ds_out.rename("dSprime_dt")
        edict   = proc.make_encoding_dict(ds_out)
        ds_out.to_netcdf(outname,encoding=edict)
    
else:
    print("ERROR: Number of runs in nclist is not equal! Check the nclist")

    
