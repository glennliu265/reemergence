#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Td' Sd' Deepest

Lowerbound estimate on damping of detrained temperatures.

1. Load in mixed layer depths
2. Load in Td/Sd damping estimated from deepest MLD point (for a single ensemble member....)
3. Select month of deepest MLD
4. Compute a single, year-round value for Td', Sd'

Created on Tue Feb  6 18:02:57 2024

@author: gliu

"""

import xarray as xr
import numpy as np

#%%

#%% Load Td' Sd' estimated from [calc_Td_decay.py]

fn_td = "CESM1_LENS_Td_Sd_lbd_exponential_fit.nc"
fp_td = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/proc/CESM1/"
ds_td = xr.open_dataset(fp_td + fn_td) # [Var, Lag_Max, Mon, Lat, Lon]

#lbd_d = ds_td.lbd.mean('lag_max')#.values # [var x mon x lat x lon]

#%% Load MLD

fp = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/proc/model_input/mld/"
fn = "CESM1_HTR_FULL_HMXL_NAtl.nc"


ds_mld = xr.open_dataset(fp+fn).h

#  Take Ens Avg
ds_mld_ensavg = ds_mld.mean('ens')

# Find month of maximum
ds_monmax = ds_mld_ensavg.argmax('mon') 

#hmonmax = ds_monmax#.values

#%% Make sure they cover the same region

dslist_rsz = proc.resize_ds([ds_td,ds_monmax])
ds_td,ds_monmax = dslist_rsz

#%% Output the data


# Read out the variables
hmonmax = ds_monmax.values # [lat x lon]
lbd_d   = ds_td.lbd.mean('lag_max').values # [var x mon x lat x lon]

# Smooth out the information

# Loop by variable
nv,nm,nlat,nlon = lbd_d.shape
npts            = nlon*nlat

lbd_d_byvar     = []
for vv in range(2):
    #
    lbd_in       = lbd_d[vv,:,:,:].reshape(nm,npts) # [mon x pts]
    idh_in       = hmonmax.flatten()[None,:]        # [1 x pts]
    
    # TAke Along Axis
    lbd_d_monmax = np.take_along_axis(lbd_in,idh_in,axis=0) # [1 x pts]
    lbd_d_byvar.append(lbd_d_monmax.squeeze(0).reshape(nlat,nlon))
    

#%% Now save each one

vnames = ["SST","SSS"]

das_fin = []
vv = 0
outpath= "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/proc/model_input/damping/"
for vv in range(2):
    
    # Tile Along Month
    lbdout = np.tile(lbd_d_byvar[vv][None,:,:].transpose(1,2,0),12).transpose(2,0,1)*-1 # Mon x Lat x Lon
    
    # There seems to be some odd values, so do some cleaning
    lbdout[lbdout>1] = 0. # Set to Zero
    
    vcoords = {
        'mon' : np.arange(1,13,1),
        'lat' : ds_td.lat.values,
        'lon' : ds_td.lon.values,
        }
    
    # Also convert to positive value
    edict = {'lbd_d':{'zlib':True}}
    da_out =xr.DataArray(lbdout,coords=vcoords,dims=vcoords,name="lbd_d")
    savename = "%sCESM1_HTR_FULL_%s_Expfit_lbdd_maxhclim_lagsfit123_Ens01.nc" % (outpath,vnames[vv])
    
    da_out.to_netcdf(savename)
    
    das_fin.append(da_out)
    
#%%
da_out = lbd_d_byvar[vv]


#lbd_d_byvar = []
#%% Check a value

klon = 40
klat = 53

lonf = -30
latf = 50

klon,klat=proc.find_latlon(lonf,latf,ds_td.lon.values,ds_td.lat.values)

fig,ax = plt.subplots(1,1)
ax.plot(lbd_d[vv,:,klat,klon])
ax.plot(hmonmax[klat,klon],lbd_d[vv,hmonmax[klat,klon],klat,klon],marker="x",markersize=24)
ax.axhline([lbd_d_byvar[vv][klat,klon]])

ax2 = ax.twinx()
ax2.plot(ds_mld_ensavg.sel(lon=lonf,lat=latf,method='nearest'),color='red')

ax.legend()


#%%

das_fin[1].isel(mon=0).plot(vmin=0,vmax=0.5)

#%%

hmax_byens       = ds.h.max(('lat','lon','mon'))  # Mean by Ens
hmax_bypoint     = ds.h.max(('mon','ens'))        # Ens Mean




#%%


