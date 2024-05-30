#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""


calc_detrainment_depth

Copied from [calc_detrainment_correlation_pointwise]


========================

Pointwise computation of detrainment damping using
corr(Detrain Month, Entrain Month-1) of anomalies.
Uses output from preproc_detrainment_data.pt

Selects ensemble of compuitation

Inputs:
------------------------
    
    varname : dims                              - units                 - processing script
    acf_mon : (mon, lag, z_t, nlat, nlon)      [corr]                  preproc_detrainment_data.pt
    h       : (mon, ens, lat, lon)              [m]                     
    
Outputs: 
------------------------
    
    varname : dims                              - units 
    lbd_d   :  (mon, nlat, nlon)                [correlation]          

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

#%% User Edits


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
loopens     = np.arange(42)#[32,] # Indicate specific indices of ensemble members to loop thru

# MLD Information
mldnc       = "CESM1_HTR_FULL_HMXL_NAtl.nc"

# Correlation Options ---
detrainceil = False # True to use ceil rather than floor of the detrainment month
interpcorr  = True  # True to interpolate between ceil and floor of detrianment month
dtdepth     = True  # Set to true to retrieve temperatures at the detrainment depth
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

# Compute detrainment depths
if dtdepth:
    # Compute kprev for ens-mean mixed layer depth cycle
    infunc = lambda x: scm.find_kprev(x,debug=False,returnh=False)
    st     = time.time()
    kprevall = xr.apply_ufunc(
        infunc, # Pass the function
        dsmld, # The inputs in order that is expected
        input_core_dims =[['mon'],], # Which dimensions to operate over for each argument... 
        output_core_dims=[['mon'],], # Output Dimension
        vectorize=True, # True to loop over non-core dims
        )
    print("Completed kprev calc in %.2fs" % (time.time()-st))
    
    
    # Compute detrainment depths for ens-mean mld
    st = time.time()
    hdetrainall = xr.apply_ufunc(
        scm.get_detrain_depth, # Pass the function
        kprevall, # The inputs in order that is expected
        dsmld,
        input_core_dims=[['mon'],['mon']], # 
        output_core_dims=[['mon'],],#['ens'],['lat'],['lon']],
        vectorize=True,
        )
    print("Completed hdetrain calc in %.2fs" % (time.time()-st))

#%% Save the output

# 

outname = "%sCESM1_HTR_FULL_hdetrain_NAtl.nc" % (mldpath)
hdetrainall.to_netcdf(outname)

outname = "%sCESM1_HTR_FULL_kprev_NAtl.nc" % (mldpath)
kprevall.to_netcdf(outname)

# Note, the above section was moved back to the original script ^^^^^^^^^^^^^^

#%%

# #%%
# # Loop for Variable
# # Loop for Ens

# #for vv in range(len(vnames)):

# #ie = 0
# #vv = 0

# vv = 0
# for ie in range(len(loopens)):
#     e     = loopens[ie]
    
    
#     # Load the Data
#     st=time.time()
#     fn      = "%sCESM1_HTR_FULL_lbd_d_params_%s_detrend%s_lagmax%i_ens%02i_regridNN.nc" % (outpath,vname,detrend,lagmax,e+1)
#     ds      = xr.open_dataset(fn)
#     lon     = ds.lon
#     lat     = ds.lat
#     z_t     = ds.z_t
#     print("Loaded dataset for ens %02i in %.2fs" % (e+1,time.time()-st))
    
    
    
#     hdetrain_all    = np.zeros(ds.lbd_d.shape)*np.nan
#     nlon=len(lon)
#     nlat=len(lat)
    
#     # Loop for each point
#     for a in tqdm(range(nlat)):
#         for o in range(nlon):
            
#             # Select MLD at point
#             if dtdepth:
#                 hpt    = dsmld.sel(lon=lon[o],lat=lat[a],method='nearest').isel(ens=e)
#             else:
#                 hpt    = dsmld.isel(ens=e,lat=a,lon=o) # [Mon,]
#             if np.all(np.isnan(hpt)): # Skip for land point
#                 continue
            
#             immax = hpt.argmax().values.item()
#             immin = hpt.argmin().values.item()
            
#             # Compute kprev for the ensemble member
#             if dtdepth:
#                 kprev   = kprevall.sel(lon=lon[o],lat=lat[a],method='nearest').isel(ens=e).values
#             else:
#                 kprev,_ = scm.find_kprev(hpt)
            
#             for im in range(12): # Loop for entrain month

                
#                 # Get the detrain month
#                 detrain_mon = kprev[im]
#                 if detrain_mon == 0.:
#                     continue # Skip when there is no entrainment
                
#                 h_detrain  = hdetrainall.sel(lon=lon[o],lat=lat[a],method='nearest').isel(ens=e,mon=im).values.item(0)
                
#                 hdetrain_all[im,a,o] = h_detrain
                
#     # Save the output (for a variable and ensemble member)
#     coords = dict(mon=np.arange(1,13,1),lat=lat,lon=lon)
#     #da_out = xr.DataArray(hdetrain_all,coords=coords,dims=coords,name="lbd_d")
#     edict  = {'lbd_d':{'zlib':True}}
#     savename = "%sCESM1_HTR_FULL_corr_d_%s_detrend%s_lagmax%i_interp%i_ceil%i_imshift%i_dtdepth%i_ens%02i_regridNN.nc" % (outpath,vname,detrend,lagmax,
#                                                                                                       interpcorr,detrainceil,imshift,dtdepth,e+1)
    
#     da_out.to_netcdf(savename,encoding=edict)
# # Loop for Month
# # for im in range(12):
# #im = 0

# # Get Entrain Month
# # Get Detrain Month
# # Apply Floor/Ceil, Corrections
# # 

# #%% Combine files for each ensemble member

# ensmerge = np.arange(42)
# nens     = len(ensmerge)

# for vv in range(len(vnames)):
#     vname = vnames[vv]
    
#     dsall = []
#     for e in tqdm(range(nens)):
        
#         savename = "%sCESM1_HTR_FULL_corr_d_%s_detrend%s_lagmax%i_interp%i_ceil%i_imshift%i_dtdepth%i_ens%02i_regridNN.nc" % (outpath,vname,detrend,lagmax,                     
#                                                                                          interpcorr,detrainceil,imshift,dtdepth,e+1)
#         ds = xr.open_dataset(savename).load()
#         dsall.append(ds)
        
#     dsall = xr.concat(dsall,dim='ens')
#     savename2 = "%sCESM1_HTR_FULL_corr_d_%s_detrend%s_lagmax%i_interp%i_ceil%i_imshift%i_dtdepth%i_ensALL_regridNN.nc" % (outpath,vname,detrend,lagmax,                     
#                                                                                   interpcorr,detrainceil,imshift,dtdepth)
#     edict = {'lbd_d':{'zlib':True}}
#     dsall.to_netcdf(savename2,encoding=edict)
    
    
                         

#%%

#%%



