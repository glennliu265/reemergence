#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Regrid Detrainment Damping calculated by preproc_detrainment_data


Inputs:
------------------------
    
    varname : dims                              - units                 - processing script
    lbd_d   :  (mon, nlat, nlon)                [-1/mon]                preproc_detrainment_data
    tau     :  (mon, z_t, nlat, nlon)           [-1/mon]
    acf_est :  (mon, lag, z_t, nlat, nlon)      [corr]
    acf_mon :  (mon, lag, z_t, nlat, nlon)      [corr]


Outputs: 
------------------------

    varname : dims                              - units 
    lbd_d   :  (mon, lat, lon)                  [-1/mon]         
    tau     :  (mon, z_t, lat, lon)             [-1/mon]
    acf_est :  (mon, lag, z_t, lat, lon)        [corr]
    acf_mon :  (mon, lag, z_t, lat, lon)        [corr]


Output File Name: 

What does this script do?
------------------------
(1) Load in data processed by 
Script History
------------------------


Created on Wed Feb 14 11:11:20 2024

@author: gliu
"""

import time
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import xarray as xr
import sys
from tqdm import tqdm
import scipy as sp
import cartopy.crs as ccrs

#%%

stormtrack = 0
if stormtrack:
    amvpath = "/home/glliu/00_Scripts/01_Projects/00_Commons/" # amv module
    scmpath = "/home/glliu/00_Scripts/01_Projects/01_AMV/02_stochmod/stochmod/model/" # scm module
else:
    amvpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/03_Scripts/" # amv module
    scmpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/03_Scripts/stochmod/model/"

sys.path.append(amvpath)
sys.path.append(scmpath)

from amv import proc

#%% Indicate file to load

# outpath = 
# vname   = '' 
# detrend = 'linear'
# lagmax  = 3
# e       = 0
# savename   = "%sCESM1_HTR_FULL_lbd_d_params_%s_detrend%s_lagmax%i_ens%02i.nc" % (outpath,vname,detrend,lagmax,e+1)

if stormtrack:
    outpath = "/stormtrack/data3/glliu/01_Data/02_AMV_Project/03_reemergence/proc/CESM1/NATL_proc/ocn_var_3d/"
    mldpath = "/stormtrack/data3/glliu/01_Data/02_AMV_Project/03_reemergence/proc/model_input/mld/"
else:
    outpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/proc/CESM1/NATL_proc/ocn_var_3d/"
    mldpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/proc/model_input/mld/"

vnames  = ["SALT","TEMP"]

# fns     = ["CESM1_HTR_FULL_lbd_d_params_SALT_detrendlinear_lagmax3_ens01.nc",
#           "CESM1_HTR_FULL_lbd_d_params_TEMP_detrendlinear_lagmax3_ens01.nc"]

fns     = ["CESM1_HTR_FULL_lbd_d_params_SALT_detrendensmean_lagmax3_ens01.nc",
          "CESM1_HTR_FULL_lbd_d_params_TEMP_detrendensmean_lagmax3_ens01.nc"]

# bounding box for final output
bbox    = [-80,0,20,65]
#%% Retrieve TLAT/TLON from a file in outpath
# Note this part should change as I modified preproc_detrainment_data to include tlat and tlon as a coord
fnlat  = "SALT_NATL_ens01.nc" # Name of file to take tlat/tlon information from

dstl   = xr.open_dataset(outpath+fnlat)
tlat   = dstl.TLAT.values
tlon   = dstl.TLONG.values


#%% Retrieve dimensions of CESM1 from another file


mldnc   = "CESM1_HTR_FULL_HMXL_NAtl.nc"
dsh     = xr.open_dataset(mldpath+mldnc)

dshreg  = dsh.sel(lon=slice(bbox[0],bbox[1]),lat=slice(bbox[2],bbox[3]))

outlat = dshreg.lat.values
outlon = dshreg.lon.values

#%% Seems the easier way might just be to do this loopwise (silly but I guess...)

v      = 1
ds     = xr.open_dataset(outpath+fns[v]).load()
dsl    = ds.assign(lon=(['nlat','nlon'],tlon),lat=(['nlat','nlon'],tlat))
dsvars = ["lbd_d","tau","acf_est","acf_mon"]


nlon,nlat   = len(outlon),len(outlat)

# Preallocate
nz          =  len(dsl.z_t)
nlags       =  len(dsl.lag)

lbd_d_all   = np.zeros((12,nlat,nlon)) * np.nan          # Estimated Detrainment Damping
tau_est_all = np.zeros((12,nz,nlat,nlon))  * np.nan      # Fitted Timescales
acf_est_all = np.zeros((12,nlags,nz,nlat,nlon)) * np.nan # Fitted ACF
acf_mon_all = np.zeros((12,nlags,nz,nlat,nlon)) * np.nan # Actual ACF

for o in tqdm(range(nlon)): # Took (1h 11 min if you don't load, 2 sec if you load, T-T)
    
    lonf = outlon[o]
    
    if lonf < 0:
        lonf += 360
        
    for a in range(nlat):
        latf   = outlat[a]
        
        # Get the nearest point
        outids = proc.get_pt_nearest(dsl,lonf,latf,tlon_name="lon",tlat_name="lat",returnid=True)
        dspt   = dsl.isel(nlat=outids[0],nlon=outids[1])
        
        # Reassign variables
        lbd_d_all[:,a,o]        = dspt['lbd_d'].values
        tau_est_all[:,:,a,o]    = dspt['tau'].values
        acf_est_all[:,:,:,a,o]  = dspt['acf_est'].values
        acf_mon_all[:,:,:,a,o]  = dspt['acf_mon'].values

#%% Apply mask based on h

mask    = np.sum(dshreg.h.values,(0,1))
mask[~np.isnan(mask)] = 1

da_mask = xr.DataArray(mask,coords=dict(lat=outlat,lon=outlon))

#%% Remap and save variables

nlat         = outlat
nlon         = outlon
mons         = np.arange(1,13,1)
z            = dsl.z_t.values
lags         = np.arange(0,37,1)

# Make data arrays
lcoords      = dict(mon=mons,lat=nlat,lon=nlon)
da_lbdd      = xr.DataArray(lbd_d_all * mask[None,:,:],coords=lcoords,dims=lcoords,name="lbd_d") #* da_mask

taucoords    = dict(mon=mons,z_t=z,lat=nlat,lon=nlon)
da_tau       = xr.DataArray(tau_est_all * mask[None,None,:,:],coords=taucoords,dims=taucoords,name="tau")

acfcoords    = dict(mon=mons,lag=lags,z_t=z,lat=nlat,lon=nlon)
da_acf_est   = xr.DataArray(acf_est_all * mask[None,None,None,:,:],coords=acfcoords,dims=acfcoords,name="acf_est")
da_acf_mon   = xr.DataArray(acf_mon_all* mask[None,None,None,:,:],coords=acfcoords,dims=acfcoords,name="acf_mon")

ds_out       = xr.merge([da_lbdd,da_tau,da_acf_est,da_acf_mon,])
edict        = proc.make_encoding_dict(ds_out)

savename     = outpath+fns[v]
savename_new = proc.addstrtoext(savename,"_regridNN",adjust=-1)

ds_out.to_netcdf(savename_new,encoding=edict)

#%% Section below here doesn't seem to work :( ................. Still Debugging

# #%%
# # Loop for variables in dataset
# dv     = 0
# dsv    = ds[dsvars[dv]]

# #%% Perform Regridding (Xesmf): Note, doesn't seem to work, only ok with global data?

# v  = 0
# ds = xr.open_dataset(outpath+fns[v])

# # Define new grid (note: seems to support flipping longitude)
# method = 'bilinear'
# ds_out = xr.Dataset({'lat':outlat,
#                      'lon':outlon})

# # Assign Lat/Lon coordinates
# dsl = ds.assign(lon=(['nlat','nlon'],tlon),lat=(['nlat','nlon'],tlat))

# # Initialize regridder
# regridder = xe.Regridder(dsl,ds_out,method,)#periodic=True)

# # Regrid
# st = time.time()
# dsl_regrid = regridder(dsl)
# print("Completed regridding in %.2fs" % (time.time()-st))

# #%% Double Check Regridded Output (doesn't seem to be working?)

# iz   = 0
# im   = 0
# proj = ccrs.PlateCarree()
# fig,axs = plt.subplots(1,2,subplot_kw={'projection':proj})


# ax = axs[0]
# #dsl.tau.isel(mon=im,z_t=iz).plot.pcolormesh(ax=ax,x='nlon',y='nlat')
# ax.scatter(dsl.lon,dsl.lat,c=dsl.tau.isel(mon=im,z_t=iz).values,transform=proj)
# ax.coastlines()
# #ax.set_extent(bbox)

# ax = axs[1]
# ax.pcolormesh(dsl_regrid.lon,dsl_regrid.lat,dsl_regrid.tau.isel(mon=im,z_t=iz).values)#),transform=proj)

# ax.coastlines()
# #ax.set_extent(bbox)
# plt.show()

# #%%

#             ds = xr.open_dataset(nclist[nc])
            
#             ds = ds.rename({"TLONG": "lon", "TLAT": "lat"})
#             da = ds
#             #da = ds[varname] # Using dataarray seems to throw an error
            
#             # Initialize Regridder
#             regridder = xe.Regridder(da,ds_out,method,periodic=True)
#             #print(regridder)
                        
#             # Regrid
#             daproc = regridder(da[varname]) # Need to input dataarray
            
#             print("Finished regridding in %f seconds" % (time.time()-start))
#             ds_rgrd.append(daproc)
            
#             # Save each ensemble member separately (or time period)
#             if savesep: 
#                 savename = "/stormtrack/data3/glliu/01_Data/02_AMV_Project/02_stochmod/%s_%s_%s_num%02i.nc" % (varname,mconfig,method,nc)
#                 daproc.to_netcdf(savename,
#                                  encoding={varname: {'zlib': True}})
                
                

# #%% Load reference regridding file




# #%% Load netCDF files

# v  = 0
# ds = xr.open_dataset(outpath+fns[v])

# #%% Load tlat and tlon






