#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute the anomalous gradient of SST and SSS

Based on the script [calc_ekman_advection_htr]

Created on Mon Aug 12 16:00:37 2024

@author: gliu
"""
import numpy as np
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import sys
import cmocean
import xarray as xr
import time
from   tqdm import tqdm

#%% Import modules

stormtrack    = 1

if stormtrack == 0:
    projpath   = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/"
    datpath     = projpath + '01_Data/model_output/'
    outpathdat  = datpath + '/proc/'
    figpath     = projpath + "02_Figures/20240207"
   
    lipath  = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/01_Data/landicemask_enssum.npy"
    rawpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/proc/CESM1/NATL_proc/"
    
    sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/03_Scripts/stochmod/model/")
    sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/03_Scripts/")
    
    # Path of model input
    outpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/proc/model_input/forcing/"

elif stormtrack == 1:
    datpath     = "/stormtrack/data3/glliu/01_Data/02_AMV_Project/02_stochmod/Model_Data/model_output/"
    #rawpath     = "/stormtrack/data3/glliu/01_Data/02_AMV_Project/02_stochmod/Model_Data/model_input/"
    rawpath     = "/stormtrack/data3/glliu/01_Data/02_AMV_Project/03_reemergence/proc/CESM1/NATL_proc/"
    outpathdat  = datpath + '/proc/'
    
    sys.path.append("/home/glliu/00_Scripts/01_Projects/00_Commons/")
    sys.path.append("/home/glliu/00_Scripts/01_Projects/01_AMV/02_stochmod/stochmod/model/")
    figpath     = "/home/glliu/02_Figures/00_Scrap/"
    
    outpath     = "/stormtrack/data3/glliu/01_Data/02_AMV_Project/03_reemergence/proc/model_input/forcing/"

from amv import proc,viz
import scm
import tbx

proc.makedir(figpath)

#%% User Edits

# Set Constants
omega = 7.2921e-5 # rad/sec
rho   = 1026      # kg/m3
cp0   = 3996      # [J/(kg*C)]
mons3 = proc.get_monstr()#('Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec')

# Variable Input options -----------------------

# End -----------------------------------
# CESM1 LE Inputs -----------------------
varname         = "SSS"
rawpath         = rawpath # Read from above
ncname_var      = "CESM1LE_%s_NAtl_19200101_20050101_bilinear.nc" % varname
savename_grad   = "%sCESM1_HTR_FULL_Monthly_gradT_%s.nc" % (rawpath,varname)

#----------------------------------------------

# Calculation Options
centered    = True  # Set to True to load centered-difference temperature
calc_dT     = True # Set to True to recalculate temperature gradients (Part 1)
debug       = True

crop_sm     = True
bbox_crop   = [-90,0,0,90]
regstr_crop = "NAtl"

#%% Functions

def calc_grad_centered(ds): # Copied from structure in calc_ekman_advection_htr
    # Copied from calc_geostrophic_advection
    dx,dy = proc.calc_dx_dy(ds.lon.values,ds.lat.values,centered=True)
    
    # Convert to DataArray
    daconv   = [dx,dy]
    llcoords = {'lat':ds.lat.values,'lon':ds.lon.values,}
    da_out   = [xr.DataArray(ingrad,coords=llcoords,dims=llcoords) for ingrad in daconv]
    dx,dy = da_out
    
    # Roll and Compute Gradients (centered difference)
    ddx = (ds.roll(lon=-1) - ds.roll(lon=1)) / dx
    ddy = (ds.roll(lat=-1) - ds.roll(lat=1)) / dy
    ddy.loc[dict(lat=ddy.lat.values[-1])] = 0 # Set top latitude to zero (since latitude is not periodic)
    
    return ddx,ddy


# -----------------------------------------
#%% Part 1: CALCULATE TEMPERATURE GRADIENTS
# -----------------------------------------


#% Load the data (temperature, not anomalized)
st   = time.time()
ds   = xr.open_dataset(rawpath + ncname_var).load()
print("Completed in %.2fs"%(time.time()-st))

# Deseason and Detrend To get the anomaly
dsa  = proc.xrdeseason(ds[varname])
dsa  = dsa - dsa.mean('ensemble')


nens,ntime,nlat,nlon = dsa.shape
#anom_grad_x          = np.zeros(dsa.shape) * np.nan
#anom_grad_y          = np.zeros(dsa.shape) * np.nan

ds_ens_x = []
ds_ens_y = []
for e in range(nens):
    ds_time_x = []
    ds_time_y = []
    
    for t in tqdm(range(ntime)): # Compute for each timestep (~9sec)
        ds_in   = dsa.isel(time=t,ensemble=e)
        
        
        ddx,ddy = calc_grad_centered(ds_in)
        ds_time_x.append(ddx)
        ds_time_y.append(ddy)
        
    ds_time_x = xr.concat(ds_time_x,dim='time')
    ds_time_y = xr.concat(ds_time_y,dim='time')
    
    ds_ens_x.append(ds_time_x)
    ds_ens_y.append(ds_time_y)
    
    
ds_ens_x = xr.concat(ds_ens_x,dim='ensemble')
ds_ens_y = xr.concat(ds_ens_y,dim='ensemble')

ds_out   = xr.merge([ds_ens_x.rename('dx'),ds_ens_y.rename('dy')])
            
    
edict    = proc.make_encoding_dict(ds_out)
savename = "%sCESM1_HTR_FULL_Monthly_gradT_%sprime.nc" % (rawpath,varname)
ds_out.to_netcdf(savename,encoding=edict)




