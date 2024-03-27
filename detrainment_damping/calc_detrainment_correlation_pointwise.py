#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

calc_detrainment_correlation_pointwise
========================

Pointwise computation of detrainment damping using
corr(Detrain Month, Entrain Month-1) of anomalies.
Uses output from preproc_detrainment_data.pt

Inputs:
------------------------
    
    varname : dims                              - units                 - processing script
    SALT    : (time, ens, lat, lon)             [W/m2]                  process_bylevel_ens
    TEMP    : (time, ens, lat, lon)
    h       : (mon, ens, lat, lon)              [m]                     ???? preproc_SM_inputs_SSS?
    
Outputs: 
------------------------
    
    varname : dims                              - units 
    lbd_d   :  (mon, nlat, nlon)                [-1/mon]
    tau     :  (mon, z_t, nlat, nlon)           [-1/mon]
    acf_est :  (mon, lag, z_t, nlat, nlon)      [corr]
    acf_mon :  (mon, lag, z_t, nlat, nlon)      [corr]
    
Output File Name: 

What does this script do?
------------------------
(1) Loads in ACFs, Mean MLD, for a given dataset for an ensemble member
(2) For a month...
(3)     A. Get the detrainment time
(4)     B. Compute retrieve the correlations from the lag ACF
(5)     C. Save the correlation

Script History
------------------------

Created on Tue Mar 26 17:26:45 2024

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

from scipy.io import loadmat


#%% # Stormtrack or Local

stormtrack = 1
if stormtrack:
    amvpath = "/home/glliu/00_Scripts/01_Projects/00_Commons/" # amv module
    scmpath = "/home/glliu/00_Scripts/01_Projects/01_AMV/02_stochmod/stochmod/model/" # scm module
else:
    amvpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/03_Scripts/" # amv module
    scmpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/03_Scripts/stochmod/model/"

sys.path.append(amvpath)
sys.path.append(scmpath)

from amv import proc,viz
import scm
import amv.loaders as dl

#%% User EDits


# Indicate the Path
if stormtrack:
    outpath  = "/stormtrack/data3/glliu/01_Data/02_AMV_Project/03_reemergence/proc/CESM1/NATL_proc/ocn_var_3d/"
    mldpath  = "/stormtrack/data3/glliu/01_Data/02_AMV_Project/03_reemergence/proc/model_input/mld/"
else:
    outpath  = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/proc/CESM1/NATL_proc/ocn_var_3d/"
    mldpath  = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/proc/model_input/mld/"
    
# Variable Names and # Members
vnames      = ["SALT","TEMP",] # "SALT",
nens        = 42
loopens     = np.arange(43)#[32,] # Indicate specific indices of ensemble members to loop thru

# MLD Information
mldnc       = "CESM1_HTR_FULL_HMXL_NAtl.nc"

# Correlation Options ---
detrainceil = False # True to use ceil rather than floor of the detrainment month
interpcorr  = True  # True to interpolate between ceil and floor of detrianment month
imshift     = 1     # Stop [imshift] months before the entraining month
# -----------------------

# Load grid to process
bboxsim     = [-80,0,0,65]
detrend     = 'ensmean' # [ensmean, linear]
lagmax      = 3 # Maximum Number of lags to fit to
lags        = np.arange(0,37,1)
nlags       = len(lags)

if stormtrack:
    #lon,lat     = scm.load_latlon()
    llfile      = "/home/glliu/01_Data/CESM1_LATLON.mat"
    ldll        = loadmat(llfile)
    lat         = ldll['LAT'].squeeze()
    lon360      = ldll['LON'].squeeze()
    lon,b       = proc.lon360to180(lon360,np.zeros((len(lon360),len(lat),1)))
    lon         = lon
else:
    lon,lat     = scm.load_latlon()
    
_,lonr,latr     = proc.sel_region(np.ones((len(lon),len(lat),1)),lon,lat,bboxsim)
nlonr,nlatr     = len(lonr),len(latr)

# Other Toggles
debug           = False


#%% Start Script

# Load Mixed Layer Depth
dsmld = xr.open_dataset(mldpath+mldnc).h.load()

# Loop for Variable
# Loop for Ens

#for vv in range(len(vnames)):

#ie = 0
#vv = 0

for vv in range(len(vnames)):
    vname = vnames[vv]
    
    for ie in range(len(loopens)):
        e     = loopens[ie]
        
        
        # Load the Data
        st=time.time()
        fn      = "%sCESM1_HTR_FULL_lbd_d_params_%s_detrend%s_lagmax%i_ens%02i_regridNN.nc" % (outpath,vname,detrend,lagmax,e+1)
        ds      = xr.open_dataset(fn)
        lon     = ds.lon
        lat     = ds.lat
        z_t     = ds.z_t
        acf_ens = ds.acf_mon.load()
        print("Loaded dataset for ens %02i in %.2fs" % (e+1,time.time()-st))
        
        
        
        corr_out    = np.zeros(ds.lbd_d.shape)*np.nan
        _,nlat,nlon = corr_out.shape
        
        # Loop for each point
        for a in tqdm(range(nlat)):
            for o in range(nlon):
                
                # Select MLD at point
                hpt    = dsmld.isel(ens=e,lat=a,lon=o) # [Mon,]
                if np.all(np.isnan(hpt)): # Skip for land point
                    continue
                
                # Compute kprev for the ensemble member
                kprev,_ = scm.find_kprev(hpt)
                
                for im in range(12): # Loop for entrain month
                
                    # Get the detrain month
                    detrain_mon = kprev[im]
                    if detrain_mon == 0.:
                        continue # Skip when there is no entrainment
                    
                    # Get indices for autocorrelation
                    dtid_floor = int(np.floor(detrain_mon)) - 1 
                    dtid_ceil  = int(np.ceil(detrain_mon)) - 1
                    entrid     = im - imshift
                    if debug:
                        print("Detaining Months [%i,%f,%i], Entrain Month [%i]" % (dtid_floor+1,detrain_mon,dtid_ceil+1,entrid+1))
                    
                    # First, get the depths
                    h_floor    = hpt.isel(mon=dtid_floor).values.item()
                    h_ceil     = hpt.isel(mon=dtid_ceil).values.item()
                    zz_floor   = proc.get_nearest(h_floor,z_t.values)
                    zz_ceil    = proc.get_nearest(h_ceil,z_t.values)
                    
                    # Retrieve the ACF, with lag 0 at the detrain month
                    acf_floor = acf_ens.isel(lat=a,lon=o,mon=dtid_floor,z_t=zz_floor)     # [Lag]
                    acf_ceil  = acf_ens.isel(lat=a,lon=o,mon=dtid_ceil,z_t=zz_ceil)       # [Lag]
                    
                    # Calculate dlag
                    dlag_floor = entrid - dtid_floor
                    if dlag_floor < 1:
                        dlag_floor = dlag_floor + 12
                    dlag_ceil  = entrid - dtid_ceil
                    if dlag_ceil < 1:
                        dlag_ceil = dlag_ceil + 12
                        
                    # Retrieve Correlation
                    corr_floor = acf_floor.isel(lag=dlag_floor).values.item()
                    corr_ceil  = acf_ceil.isel(lag=dlag_ceil).values.item()
                    
                    # Interp if option is chosen
                    if interpcorr:
                        corr_mon = np.interp(detrain_mon,[dtid_floor+1,dtid_ceil+1],[corr_floor,corr_ceil])
                        
                    elif detrainceil:
                        corr_mon = corr_ceil
                    else:
                        corr_mon = corr_floor
                    corr_out[im,a,o] = corr_mon.copy()
                    
                    
        
        
        # Save the output (for a variable and ensemble member)
        coords = dict(mon=np.arange(1,13,1),lat=lat,lon=lon)
        da_out = xr.DataArray(corr_out,coords=coords,dims=coords,name="lbd_d")
        edict  = {'lbd_d':{'zlib':True}}
        savename = "%sCESM1_HTR_FULL_corr_d_%s_detrend%s_lagmax%i_interp%i_ceil%i_imshift%i_ens%02i_regridNN.nc" % (outpath,vname,detrend,lagmax,
                                                                                                          interpcorr,detrainceil,imshift,e+1)
        
        da_out.to_netcdf(savename,encoding=edict)
# Loop for Month
# for im in range(12):
#im = 0

# Get Entrain Month
# Get Detrain Month
# Apply Floor/Ceil, Corrections
# 



#%%

#%%



