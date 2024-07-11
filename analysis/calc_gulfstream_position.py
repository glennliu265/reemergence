#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Compute the Gulf Stream position 

Based on Terry Joyce's method using dynamic topography
With additional suggestions by Lilli Enders


Created on Tue Jun 25 11:30:39 2024

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
import warnings

# ----------------------------------
# %% Import custom modules and paths
# ----------------------------------

# Import re-eergemce parameters

# Indicate the Machine!
machine = "stormtrack"

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

#%% Load the dataset

# Load SSH
ncname  = "CESM1LE_SSH_NAtl_19200101_20050101_bilinear.nc"
ds_all  = xr.open_dataset(rawpath + ncname)

# Limit to Region
bboxsel = [-80,0,20,60]
ds_reg  = proc.sel_region_xr(ds_all,bboxsel).SSH.load()

# Compute mean and stedev
ds_mean = ds_reg.mean('time')
ds_std  = ds_reg.std('time')


dsa     = proc.xrdeseason(ds_reg)


#%% Simplied search of GS Position

def get_gs_position(ds_std,bbox=[-80,-10,20,50]):
    
    # Crop to Gulf Stream Region
    gs_region = proc.sel_region_xr(ds_std,bbox)
    
    # Get Indices of max latitude
    latmax_id = gs_region.argmax('lat')
    
    # Retrieve latitude values
    nens = 42
    gsi_lats = []
    for e in range(42):
        lats_ens = gs_region.lat.data[latmax_id.isel(ensemble=e).values]
        gsi_lats.append(lats_ens)
    
    # Place into Data Array
    coords  = dict(ens=np.arange(1,43,1),lon=gs_region.lon.data)
    gsi_lat = np.array(gsi_lats)
    gsi_lat = xr.DataArray(gsi_lat,coords=coords,dims=coords)
    
    return gsi_lat
    
gsi_lat = get_gs_position(ds_std)
    


fig,ax,_    = viz.init_orthomap(1,1,bboxplot,figsize=(12,4.5))
ax          = viz.add_coast_grid(ax,bbox=bboxplot,fill_color='lightgray')

nens = 42
for e in range(nens):
    plotlat = gsi_lat.isel(ens=e)
    #plotlat = ds_std.lat.data[latmax_id.isel(ensemble=e).values]#gsi_lat.isel(ensemble=e).data
    plotlon = gsi_lat.lon.data#gsi_lon.data
    ax.plot(plotlon,plotlat,transform=proj,alpha=0.4)

ax.plot(gsi_lat.lon,gsi_lat.mean('ens'),transform=proj,color="k")
plt.show() 



# Save the output
edict    = {'lat':{'zlib':True}}
outpath  = "/home/glliu/01_Data/"
savename = "%sGSI_Location_CESM1_HTR_MaxStdev.nc" % outpath
gsi_lat = gsi_lat.rename('lat')
gsi_lat.to_netcdf(savename,encoding=edict)

#%% ===========================================================================
#% In the section below, I tried to replicate more exactly what Lilli's function does...

ds_format = ds_reg.rename(dict(lon='longitude',
                               lat='latitude',
                               ))

#%% Test Lilli's Function

def gs_index_joyce(dataset,lonbounds=[290,308],lon360=True,latbounds=[20,50]):
    """
    Written by Lilli Enders
    
    Calculate the Locations of Gulf Stream Indices using Terry Joyce's Maximum Standard Deviation Method (Pérez-Hernández and Joyce (2014))
    Inputs:
    - dataset: containing longitude, latitude, sla, sla_std
    Returns:
    - gsi_lon: longitudes of gulf stream index points
    - gsi_lat: latitudes of gulf stream index points
    - std_ts: time series of gulf stream index
    
    """
    
    # Load data array, trim longtiude to correct window
    if lon360:
        ds = dataset.sel(longitude=slice(lonbounds[0], lonbounds[1]),
                         latitude=slice(latbounds[0],latbounds[1]))
    else:
        ds = dataset.sel(longitude=slice(lonbounds[0]-360,lonbounds[1]-360),
                         latitude=slice(latbounds[0],latbounds[1]))
    # Coarsen data array to nominal 1 degree (in longitude coordinate)
    crs_factor = int(1 / (ds.longitude.data[1] - ds.longitude.data[
        0]))  # Calculate factor needed to coarsen by based on lon. resolution
    if crs_factor != 0:
        ds = ds.coarsen(longitude=crs_factor,
                        boundary='pad').mean()  # Coarsen grid in longitude coord using xarray built in
    gsi_lon = ds.longitude.data  # Save gsi longitudes in array
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        # Take time average to produce a single field
        mn_std = ds.sla_std.data # Lat x Lon
        mn_sla = np.nanmean(ds.sla.data, axis=0)
        # Calculate location (latitude) of maximum standard deviation
        gsi_lat_idx = np.nanargmax(mn_std, axis=0)
        gsi_lat = ds.latitude[gsi_lat_idx].data
        sla_flt = ds.sla.data.reshape(len(ds.time), len(ds.latitude), len(ds.longitude))
        temp = np.zeros(len(ds.longitude))
        sla_ts = np.zeros(len(ds.time))
        sla_ts_std = np.zeros(len(ds.time))

        for t in range((len(ds.time))):
            for lon in range(len(ds.longitude)):
                temp[lon] = sla_flt[t, gsi_lat_idx[lon], lon]
            sla_ts[t] = np.nanmean(temp)
            sla_ts_std[t] = np.nanstd(temp)
    return (gsi_lon, gsi_lat, sla_ts, sla_ts_std)


#%%
lon         = dsa.lon
lat         = dsa.lat
coords      = dict(latitude=lat.data,longitude=lon.data)
xx,yy       = np.meshgrid(lon,lat)
#longitude = xr.DataArray(xx,coords=coords,dims=coords,name='longitude')
#latitude  = xr.DataArray(yy,coords=coords,dims=coords,name='latitude')

iens        = 0
ds_in       = xr.merge([ds_format.isel(ensemble=iens).rename('sla'),ds_format.isel(ensemble=iens).std('time').rename('sla_std')])


joyceout    = gs_index_joyce(ds_in,lonbounds=[-80+360,0+360],lon360=False)
gsi_lon, gsi_lat, sla_ts, sla_ts_std = joyceout

#%% Plot Things
clvls       = np.arange(-150,160,10)
proj        = ccrs.PlateCarree()
bboxplot    = [-80,0,20,60]
fig,ax,_    = viz.init_orthomap(1,1,bboxplot,figsize=(12,4.5))
ax          = viz.add_coast_grid(ax,bbox=bboxplot,fill_color='lightgray')

pcm         = ax.pcolormesh(lon,lat,ds_std.isel(ensemble=iens),cmap='inferno',transform=proj,zorder=-4)
cl          = ax.contour(lon,lat,ds_mean.isel(ensemble=iens),colors="k",linewidths=1.5,transform=proj,levels=clvls,zorder=-3)
ax.clabel(cl)


# Plot Gulf Stream Location
ax.scatter(gsi_lon,gsi_lat,c='w',transform=proj,marker="x",label="Gulf Stream Location")

ax.legend()
cb = viz.hcbar(pcm,ax=ax,fraction=0.045)
cb.set_label("Stdev (Sea Level Anomaly) [cm]")


plt.show()
