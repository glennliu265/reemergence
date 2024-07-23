#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Compute the location of the gulf stream by simply taking the maximum of Uvel/Vvel
From the seasonal cycle

Created on Mon Jul 22 16:23:09 2024

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


#%% Indicate calculation settings

gs_lons = [-90 + 360, -50 + 360]

#northbound = 45 # Northern most latitude (Change this below!!)
outpath    = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/01_Data/CESM1_LE/proc/NATL/"

#%% Plotting Information

bboxplot                    = [-80,0,20,65]
mpl.rcParams['font.family'] = 'Avenir'
mons3                       = proc.get_monstr(nletters=3)
fsz_tick                    = 18
fsz_axis                    = 14
fsz_title                   = 16

proj                        = ccrs.PlateCarree()

#%% Load necessary data

ds_uvel,ds_vvel = dl.load_current()
ds_bsf          = dl.load_bsf(ensavg=False)
ds_ssh          = dl.load_bsf(ensavg=False,ssh=True)

# Load data processed by [calc_monmean_CESM1.py]
ds_sss          = dl.load_monmean('SSS')
ds_sst          = dl.load_monmean('SST')


tlon  = ds_uvel.TLONG.mean('ens').values
tlat  = ds_uvel.TLAT.mean('ens').values


# Load Mixed-Layer Depth
mldpath = input_path + "mld/"
mldnc   = "CESM1_HTR_FULL_HMXL_NAtl_EnsAvg.nc"
ds_mld  = xr.open_dataset(mldpath+mldnc).h.load()

#ds_h          = dl.load_monmean('HMXL')

#%% Load Gulf Stream and masks

# Load Land Ice Mask
icemask     = xr.open_dataset(input_path + "masks/CESM1LE_HTR_limask_pacificmask_enssum_lon-90to20_lat0to90.nc")

mask        = icemask.MASK.squeeze()
mask_plot   = xr.where(np.isnan(mask),0,mask)#mask.copy()

mask_apply  = icemask.MASK.squeeze().values
#mask_plot[np.isnan(mask)] = 0



ds_gs = dl.load_gs()
ds_gs = ds_gs.sel(lon=slice(-90,-50))


#%%

u2      = (ds_uvel.UVEL**2 + ds_vvel.VVEL**2)**0.5
u2_mean = u2.mean('ens')

#%% For each tlon, find the maximum latitude

northbound = 45

nlon   = len(ds_uvel.nlon)

lonpos = np.zeros((nlon,12)) * np.nan # Longitude Values
latpos = np.zeros((nlon,12)) * np.nan# Latitude Values

ds_tlon = ds_uvel.TLONG.mean('ens')
ds_tlat = ds_uvel.TLAT.mean('ens')

for im in range(12):
    for o in range(nlon):
        
        
        
        lonval = ds_tlon.isel(nlon=o)  # [Lat,] Longitude values corresponding to each latitude
        latval = ds_tlat.isel(nlon=o)  # [Lat,]
        u2val  = u2_mean.isel(nlon=o,month=im) # [Lat,]
        
        # Search for points south of the location
        u2val  = xr.where(latval > northbound,np.nan,u2val)
        
        if np.all(np.isnan(u2val.data)):
            print("All Nan Slice at %.2f" % (lonval.data.mean()))
            continue
        else:
            
            idmax  = np.nanargmax(u2val.data)
            
            
            lonout = lonval.isel(nlat=idmax).data.item()
            if (lonout > gs_lons[1]) or (lonout < gs_lons[0]):
                continue
            lonpos[o,im] = lonval.isel(nlat=idmax).data.item()
            latpos[o,im] = latval.isel(nlat=idmax).data.item()
            
        
#%% Drop points beyond the bounds

first_non_nan = [np.where(~np.isnan(lonpos[:,im]))[0][0] for im in range(12)]
print(first_non_nan)


kstart = np.nanmin(first_non_nan)

lonpos = lonpos[kstart:,:]
latpos = latpos[kstart:,:]

    
#%% Plot to Check

bboxplot                        = [-80,0,20,65]
mpl.rcParams['font.family']     = 'Avenir'

qint                            = 2

fig,ax,_    = viz.init_orthomap(1,1,bboxplot,figsize=(18,6.5))
ax          = viz.add_coast_grid(ax,bbox=bboxplot,fill_color="lightgray")


for im in range(12):
    ax.scatter(lonpos[:,im],latpos[:,im],transform=proj)
    ax.plot(lonpos[:,im],latpos[:,im],transform=proj,lw=2.5)

# Plot Currents
plotu = ds_uvel.UVEL.mean('ens').mean('month').values
plotv = ds_vvel.VVEL.mean('ens').mean('month').values
ax.quiver(tlon[::qint,::qint],tlat[::qint,::qint],plotu[::qint,::qint],plotv[::qint,::qint],
          color='navy',transform=proj,alpha=0.75)


#%% Save the Data Array

lon_mean    = lonpos.mean(1) #Mean Longitude across the months
coords      = dict(lon_mean=lon_mean,mon=np.arange(1,13,1))


lonpos = xr.DataArray(lonpos,coords=coords,dims=coords,name='lon')
latpos = xr.DataArray(latpos,coords=coords,dims=coords,name='lat')

ds_out = xr.merge([lonpos,latpos])
edict  = proc.make_encoding_dict(ds_out)
savename = "%sGSI_Location_CESM1_HTR_MaxU2Mag.nc" % (outpath)

ds_out.to_netcdf(savename,encoding=edict)
print("Saved output to %s" % savename)


#%%

# Check the difference values
londiff = lonpos[:,:] - lonpos.mean(1)[:,None]



londiff.shape




#%% Check first index where it is nan


    
    





