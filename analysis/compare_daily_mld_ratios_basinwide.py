#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Compare the intramonthly vs. interannual variability of mixed-layer depth anomalies
Uses output from check_daily_mld...

Created on Thu Oct 24 15:50:05 2024

@author: gliu
"""

import xarray as xr
import numpy as np
import matplotlib as mpl

import matplotlib.pyplot as plt
import sys
import glob
import os

import tqdm
import time

import cartopy.crs as ccrs
import matplotlib.patheffects as pe
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

#%% Plotting Information

bboxplot                    = [-80,0,20,65]
mpl.rcParams['font.family'] = 'Avenir'
mons3                       = proc.get_monstr(nletters=3)
fsz_tick                    = 18
fsz_axis                    = 22
fsz_title                   = 28

proj                        = ccrs.PlateCarree()

#%%
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


#%% Get Point Information

# Get Point Info
pointset        = "PaperDraft02"
ptdict          = rparams.point_sets[pointset]
ptcoords        = ptdict['bboxes']
ptnames         = ptdict['regions']
ptnames_long    = ptdict['regions_long']
ptcols          = ptdict['rcols']
ptsty           = ptdict['rsty']


#%% Load Mixed-Layer Depth Variability commputed from check_daily_mld

# Load Daily Information from Ensemble Member 2
ncname = "CESM1_Daily_HMXL_Ens02_SumStats.nc"
ds_dailysum = xr.open_dataset(rawpath+ncname).load()

bbox_daily = [ds_dailysum.lon[0].item(),
              ds_dailysum.lon[-1].item(),
              ds_dailysum.lat[0].item(),
              ds_dailysum.lat[-1].item()]

# Load total MLD
ncname_hmxl = "CESM1LE_HMXL_NAtl_19200101_20050101_bilinear.nc"
ds_mld = xr.open_dataset(rawpath+ncname_hmxl).isel(ensemble=1).HMXL.load()
ds_mld = proc.sel_region_xr(ds_mld,bbox=bbox_daily)

#%% Compute same variables by month

monmld_mean = ds_mld.groupby('time.month').mean('time')
monmld_std  = ds_mld.groupby('time.month').std('time')
monmld_max  = ds_mld.groupby('time.month').max('time')
monmld_min  = ds_mld.groupby('time.month').min('time')


#%% Check lat Lon



#%% Compute the ratio

sigma_ratio = (ds_dailysum.stats.isel(metric=1)).data / monmld_std.rename(dict(month='mon')).data

coords_new = dict(mon=np.arange(1,13,1),lat=monmld_mean.lat,lon=monmld_mean.lon)
sigma_ratio = xr.DataArray(sigma_ratio,coords=coords_new,dims=coords_new)


#%%


#%% Check the values at a latitude
lonf = -33
latf = 53

fig,ax = viz.init_monplot(1,1)
ax.plot(mons3,proc.selpt_ds(monmld_std,lonf,latf)/100,label="Monthly")
ax.plot(mons3,proc.selpt_ds(ds_dailysum.stats.isel(metric=1),lonf,latf)/100,label="Daily")
ax.legend()

ax2 = ax.twinx()
ax2.scatter(mons3,proc.selpt_ds(sigma_ratio,lonf,latf))
ax2.set_ylim([0.5,4])

#%% Plot a spatial pattern of the mixed-layer depth ratio



cints       = np.arange(1,5.5,0.5)

pmesh       = True

#cints  = np.arange(0,1.1,0.1)
# Initialize Plot and Map

fig,ax,_    = viz.init_orthomap(1,1,bboxplot,figsize=(24,10))
ax          = viz.add_coast_grid(ax,bbox=bboxplot,fill_color="lightgray",fontsize=fsz_tick)

plotvar     = sigma_ratio.max('mon')
if pmesh:
    pcm         = ax.pcolormesh(plotvar.lon,plotvar.lat,plotvar,transform=proj,
                                vmin=cints[0],vmax=cints[-1],
                                cmap='cmo.turbid')
else:
    pcm         = ax.contourf(plotvar.lon,plotvar.lat,plotvar,transform=proj,levels=cints,cmap='cmo.turbid')
cl          = ax.contour(plotvar.lon,plotvar.lat,plotvar,transform=proj,levels=cints,colors="k",
                         linewidths=0.75)
ax.clabel(cl,fontsize=fsz_tick)


cb          = viz.hcbar(pcm,ax=ax)
cb.ax.tick_params(labelsize=fsz_tick)
cb.set_label(r"MLD Stdev. Ratio: $\frac{\sigma_{h,Monthly}}{\sigma_{h,Daily}}$",fontsize=fsz_axis)

#ax.set_title("SST-SSS Coherence Squared \n@ Period = %.f years" % (1/(coh_lf.freq*dt*12).item()),fontsize=fsz_axis)


# Plot Gulf Stream Position
ax.plot(ds_gs2.lon.mean('mon'),ds_gs2.lat.mean('mon'),transform=proj,lw=2.5,c='k',ls='dashdot')

# Plot Ice Edge
ax.contour(icemask.lon,icemask.lat,mask_plot,colors="cyan",linewidths=2.5,
           transform=proj,levels=[0,1],)


nregs = len(ptnames)
for ir in range(nregs):
    pxy   = ptcoords[ir]
    ax.plot(pxy[0],pxy[1],transform=proj,markersize=20,markeredgewidth=.5,c=ptcols[ir],
            marker='*',markeredgecolor='k')
        
        
savename = "%sMLD_Variability_Ratio_Daily_v_Monthly_Ens02.png" % (figpath)
plt.savefig(savename,dpi=150,bbox_inches='tight')



