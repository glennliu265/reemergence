#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Map Flux Ratios

Identify where Latent Heat Flux Plays a Major Role Relative to the Net Heat Flux

Created on Mon Oct  2 13:40:38 2023

@author: gliu
"""

import xarray as xr
import sys
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
import cmocean as cmo
 
from tqdm import tqdm
#%%
datpath = "/stormtrack/data3/glliu/01_Data/02_AMV_Project/01_hfdamping/hfdamping_HTR/01_PREPROC/"
outpath = "/stormtrack/data3/glliu/01_Data/02_AMV_Project/03_reemergence/proc/variances/"
#%% Import custom modules

amvpath = "/home/glliu/00_Scripts/01_Projects/00_Commons/" # amv module
scmpath = "/home/glliu/00_Scripts/01_Projects/01_AMV/02_stochmod/stochmod/model/" # scm module

sys.path.append(amvpath)
sys.path.append(scmpath)

from amv import proc,viz
import scm
import amv.loaders as dl

#%% Load Net and Sensible heat fluxes

nens     = 42
varnames = ['qnet','LHFLX']
bbox     = [-80,0,0,65]

#%% Note: This section takes 18 minutes, shouldn't take this long... need to fix it.

ds_all = []
for v in varnames:
    ds_merge = []
    for e in tqdm(range(nens)):
        ds     = xr.open_dataset(datpath+"enso/htr_%s_detrend1_ENSOrem_lag1_pcs3_monwin3_1920to2006_ens%02i.nc" % (v,e+1))
        ds     = proc.format_ds(ds,verbose=False)
        ds     = proc.sel_region_xr(ds,bbox)[v].load()
        ds_merge.append(ds)
    
    # Merge Left to take indices (latitude) of first xarray file
    ds_merge = xr.concat(ds_merge,dim="ens",join='left')
    ds_all.append(ds_merge)

# -------------------------------------------------------
#%% Load out each of the arrays and compute the ratios
# -------------------------------------------------------
qnet  = ds_all[0].values
lhflx = ds_all[1].values

hflx_ratio = np.var(lhflx,1)/np.var(qnet,1)

plt.pcolormesh(np.var(lhflx,1)[0,...]),plt.show()
plt.pcolormesh(np.var(qnet,1)[0,...]),plt.show()

lon   = ds.lon.values
lat   = ds.lat.values
times = ds.time.values

# -------------------------------------------------------
#%% Map individually the variance of both lhflx and qnet
# -------------------------------------------------------

fig,axs = plt.subplots(1,2,figsize=(12,6),subplot_kw={'projection':ccrs.PlateCarree()})

for v in  range(2):
    
    if v == 0:
        title   = "1$\sigma$ $Q_L$"
        plotvar = lhflx
        clvls = np.arange(0,55,5)
        cmap  = 'cmo.rain_r'
    elif v == 1:
        title = "1$\sigma$ $Q_{net}$"
        plotvar = qnet
        clvls = np.arange(0,85,5)
        cmap  = 'inferno'
    
    ax      = axs[v]
    ax      = viz.add_coast_grid(ax,bbox=bbox)
    #ax.set_title(title)
    
    plotvar = np.nanmean(np.std(plotvar,1),0)
    cf      = ax.contourf(lon,lat,plotvar,extend='both',levels=clvls,cmap=cmap)
    cl    = ax.contour(lon,lat,plotvar,colors='k',linewidths=0.75,levels=clvls)
    ax.clabel(cl)
    ax.plot(-30,50,marker="x",color="blue")
    cb = fig.colorbar(cf,ax=ax,orientation='horizontal',fraction=0.05,pad=0.01)
    cb.set_label("%s [W/m2]" % (title))
plt.show()

# --------------------------------------------
#%% Make a map of the heat flux variance ratio
# --------------------------------------------

clvls   = np.arange(0,1.1,0.1)
fig,ax  = plt.subplots(1,1,figsize=(12,6),subplot_kw={'projection':ccrs.PlateCarree()})

#plotvar = np.nanmean(hflx_ratio,0)

#plotvar = np.nanmean(np.std(lhflx,1),0)/np.nanmean(np.std(qnet,1),0)
#plotvar = np.nanmean(np.std(lhflx,1)/np.std(qnet,1),0)
plotvar = np.nanmean(np.var(lhflx,1),0)/np.nanmean(np.var(qnet,1),0)
#plotvar = np.nanmean(hflx_ratio,0)
ax      = viz.add_coast_grid(ax,bbox=bbox)
cf      = ax.contourf(lon,lat,plotvar,levels=clvls,extend='both')
cl      = ax.contour(lon,lat,plotvar,colors='k',linewidths=0.75,levels=clvls)
ax.clabel(cl)
fig.colorbar(cf,ax=ax)
ax.set_title("Ratio var($Q_{L}$) / var($Q_{net}$) (ensemble mean)")
plt.show()

#%% Compute the monthly variance

nens,ntime,nlat,nlon = lhflx.shape
nyrs    = int(ntime/12)
monvar  = np.zeros((2,12,nens,nlat,nlon))
ovrvar  = np.zeros((2,nens,nlat,nlon))

invars  = [qnet,lhflx]

for v in range(2):
    
    invar = invars[v]
    ovrvar[v,...] = np.var(invar,1)
    invar = invar.reshape(nens,nyrs,12,nlat,nlon)
    
    for im in range(12):
        
        monvar[v,im,:,:,:] = np.var(invar[:,:,im,:,:],1)
        


coords= {
    'varnames':["QNET","LHFLX"],
    'mons'    :np.arange(1,13,1),
    'ens'     :np.arange(1,43,1),
    'lat'     :lat,
    'lon'     :lon
    }

da        = xr.DataArray(monvar,coords=coords)
#edict = proc.make_encoding_dict(da
savename  = "%sMonthly_Variances_Qnet_LHFLX.nc" % outpath
da.to_netcdf(savename)
#%%

coords= {
    'varnames':["QNET","LHFLX"],
    'ens'     :np.arange(1,43,1),
    'lat'     :lat,
    'lon'     :lon
    }
da2        = xr.DataArray(ovrvar,coords=coords)
#edict = proc.make_encoding_dict(da
savename  = "%sOverall_Variances_Qnet_LHFLX.nc" % outpath
da2.to_netcdf(savename)
#%%