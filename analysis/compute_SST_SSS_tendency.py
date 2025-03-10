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

from cmcrameri import cm
import matplotlib.patheffects as pe

# ----------------------------------
# %% Import custom modules and paths
# ----------------------------------

# Import re-eergemce parameters

# Indicate the Machine!
machine = "Astraeus"

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

#%% Define some functions

# From viz_inputs_basinwide
def init_monplot(bboxplot=[-80,0,20,60],fsz_axis=24):
    mons3         = proc.get_monstr()
    plotmon       = np.roll(np.arange(12),1)
    fig,axs,mdict = viz.init_orthomap(4,3,bboxplot=bboxplot,figsize=(18,18))
    for a,ax in enumerate(axs.flatten()):
        im = plotmon[a]
        ax = viz.add_coast_grid(ax,bbox=bboxplot,fill_color="lightgray")
        ax.set_title(mons3[im],fontsize=fsz_axis)
    return fig,axs

#%% Load Plotting Parameters
bboxplot    = [-80,0,20,60]

# Load Land Ice Mask
icemask     = xr.open_dataset(input_path + "masks/CESM1LE_HTR_limask_pacificmask_enssum_lon-90to20_lat0to90.nc")


mask        = icemask.MASK.squeeze()
mask_plot   = xr.where(np.isnan(mask),0,mask)#mask.copy()


mask_reg_sub    = proc.sel_region_xr(mask,bboxplot)
mask_reg_ori    = xr.ones_like(mask) * 0
mask_reg        = mask_reg_ori + mask_reg_sub


mask_apply  = icemask.MASK.squeeze().values
#mask_plot[np.isnan(mask)] = 0

# Load Gulf Stream
ds_gs   = dl.load_gs()
ds_gs   = ds_gs.sel(lon=slice(-90,-50))
ds_gs2  = dl.load_gs(load_u2=True)

#%% Load more infomation on the points

# Get Point Info
pointset        = "PaperDraft02"
ptdict          = rparams.point_sets[pointset]
ptcoords        = ptdict['bboxes']
ptnames         = ptdict['regions']
ptnames_long    = ptdict['regions_long']
ptcols          = ptdict['rcols']
ptsty           = ptdict['rsty']

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

#%%

