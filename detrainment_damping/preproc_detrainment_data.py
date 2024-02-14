#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

preproc_detrainment_data
========================

Loads in data cropped by [process_bylevel_ens] and prepares for detrainment
damping calculations. Processing a single level at a time.

Inputs:
------------------------
    
    varname : dims                              - units                 - processing script
    SALT    : (time, ens, lat, lon)             [W/m2]                  process_bylevel_ens
    TEMP    : (time, ens, lat, lon)
    h       : (mon, ens, lat, lon)              [m]                     ???? preproc_SM_inputs_SSS?
    


Outputs: 
------------------------

    varname : dims                              - units 
    eofs    : (mode,mon,ens,lat,lon)            - [W/m2/stdevPC]

    
Output File Name: 

What does this script do?
------------------------
(1)
(2)
(3)

Script History
------------------------

Created on Tue Feb 13 11:36:52 2024

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

#%% # Stormtrack or Local

stormtrack = 0
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
    
    
mldnc    = "CESM1_HTR_FULL_HMXL_NAtl.nc"

# Variable Names and # Members
vnames   = ["SALT","TEMP"]
nens     = 42

# Detrainment Options
lags     = np.arange(0,37,1)
nlags    = len(lags)
lagmax   = 3 # Maximum Number of lags to fit to
detrend  = 'linear'


# Load grid to process
bboxsim     = [-80,0,0,65]
lon,lat     = scm.load_latlon()
_,lonr,latr = proc.sel_region(np.ones((len(lon),len(lat),1)),lon,lat,bboxsim)
nlonr,nlatr = len(lonr),len(latr)

# Other Toggles
debug    = True

#%% Helper Functions

# Functions ---
def calc_acf_ens(ts_monyr,lags):
    # tsens is the anomalized values [yr x mon x z]
    acfs_mon = []
    for im in range(12):
        basemonth   = im+1
        varin       = ts_monyr[:,:,:]  # Month x Year x Npts
        out         = proc.calc_lagcovar_nd(varin, varin, lags, basemonth, 1)
        
        acfs_mon.append(out)
        # <End Month Loop>
    return np.array(acfs_mon) # [Mon Lag Depth]

def fit_exp_ens(acfs_mon,lagmax):
    # acfs_mon [month x lag x depth] : Monthly lagged ACFs
    
    _,nlags,nz = acfs_mon.shape
    tau_est = np.zeros((12, nz))
    acf_est = np.zeros((12, nlags, nz))
    
    for im in range(12):
        for zz in range(nz):
            acf_in = acfs_mon[im, :, zz] # Select Depth and Month
            
            outdict             = proc.expfit(acf_in, lags, lagmax=lagmax)
            tau_est[im, zz]     = outdict['tau_inv'].copy()
            acf_est[im, :, zz]  = outdict['acf_fit'].copy()
    return tau_est,acf_est


#%% Now try this for a single ensemble member (load and preprocess data)


e = 0
v = 1

# Variable Loop
# for v in range(2):

vname = vnames[v]
# Ens Loop
#   for e in range(nens):

# Load data for ensemble member
nc = "%s%s_NATL_ens%02i.nc" % (outpath,vname,e+1)
st = time.time()
ds = xr.open_dataset(nc).load() # [time x z_t x nlat x nlon]

# Load dimensions for ease
if e == 0:
    times = ds.time.values
    z = ds.z_t.values/100 # Convert to meters
    tlon = ds.TLONG.values
    tlat = ds.TLAT.values

# Remove seasonal cycle
ds_anom = proc.xrdeseason(ds)

# Remove Trend
if detrend == "linear":
    invar    = ds_anom[vname].values
    dtdict   = proc.detrend_dim(invar,0,return_dict=True)
    print(dtdict.keys())
    dtvar = dtdict['detrended_var']
    #invar_dt = sp.signal.detrend(invar,axis=0) # Detrend Along Time Axis
elif detrend == "ensmean":
    print("%s currently not implemented" % (detrend))
    # Load Ens Mean
    # Remove
    # Load out 
else:
    print("%s detrend not implemented" % (detrend))



# Check Detrending
if debug:
    # Check Detrend
    klon = 59
    klat = 44
    kz   = 22
    fig,ax = plt.subplots(1,1,figsize=(12,4))
    ax.plot(invar[:,kz,klat,klon],label='raw',color='gray')
    ax.plot(dtvar[:,kz,klat,klon],label="Detrended",alpha=1,ls='dashed',color='cornflowerblue')
    ax.legend()
    ax.set_title("%s Detrend @ Lon %.3f, Lat %.3f, Depth %.3f [meters]" % (vname,tlon[klat,klon],tlat[klat,klon],z[kz]))


#%% Load the Mixed Layer Depth

dsmld = xr.open_dataset(mldpath+mldnc).h



#%% Compute Autocorr across depths and months (copied over from calc_detrainment damping)



# Replace Detrended Anomalies into dataset
#da_dt = xr.DataArray(dtvar,coords=ds[vname].coords,dims=ds[vname].coords)


# Retrieve Dimensions and reshape to year x mon
ntime,nz,nlat,nlon=dtvar.shape
nyr               =int(ntime/12)
dtvar_yrmon       =dtvar.reshape((nyr,12,nz,nlat,nlon))


lbd_d_all   = np.zeros((12,nlat,nlon)) * np.nan          # Estimated Detrainment Damping
tau_est_all = np.zeros((12,nz,nlat,nlon))  * np.nan      # Fitted Timescales
acf_est_all = np.zeros((12,nlags,nz,nlat,nlon)) * np.nan # Fitted ACF
acf_mon_all = np.zeros((12,nlags,nz,nlat,nlon)) * np.nan # Actual ACF

for o in tqdm(range(nlon)):
    
    for a in range(nlat):
        
        # Retrieve variable at point
        varpt = dtvar_yrmon[:,:,:,a,o].copy() # Yr x Mon x Depth
        if np.all(np.isnan(varpt)):
            continue # Skip the Point because it is on land
            
        # Crop to depth for the point
        depthsum = np.sum(varpt,(0,1))
        idnan_z  = np.where(np.isnan(depthsum))[0]
        varpt[:,:,idnan_z] = 0 # Set to Zeros
        
        # Retrieve Mixed layer depth cycle at a point
        lonf = tlon[a,o]
        if lonf > 180:
            lonf -= 360 # Change to degrees west
        latf = tlat[a,o]
        hpt  = dsmld.isel(ens=0).sel(lon=lonf,lat=latf,method='nearest').values#[month]
        
        if np.any(np.isnan(hpt)):
            continue
            
        # Section here is taken from calc_detrainemtn_damping_pt -------------
        # Input Data
        ts_monyr     = varpt.transpose(1,0,2)        # Anomalies [mon x yr x otherpts (z)]
        hclim        = hpt                           # MLD Cycle [mon]
        
        # (1) Estimate ACF
        acfs_mon = calc_acf_ens(ts_monyr,lags) # [mon x lag x depth]
        acfs_mon[:,:,idnan_z] = 0 # To Avoid throwing an error
        
        # (2) Fit Exponential Func
        tau_est,acf_est = fit_exp_ens(acfs_mon,lagmax) # [mon x depth], [mon x lags x depth]
        
        # (3) Compute Detraiment dmaping
        kprev,_ = scm.find_kprev(hclim)
        lbd_d   = scm.calc_tau_detrain(hclim,kprev,z,tau_est,debug=False)
        
        # Correct zeros back to nans
        acfs_mon[:,:,idnan_z] = np.nan
        tau_est[:,idnan_z] = np.nan
        acf_est[:,:,idnan_z] = np.nan
        
        
        # Save Output
        lbd_d_all[:,a,o]       = lbd_d.copy()
        tau_est_all[:,:,a,o]   = tau_est.copy()
        acf_est_all[:,:,:,a,o] = acf_est.copy()
        acf_mon_all[:,:,:,a,o] = acfs_mon.copy()
    
#%% Save the output (intermediate, unregridded)


nlat       = np.arange(0,nlat)
nlon       = np.arange(0,nlon)
mons       = np.arange(1,13,1)

# Make data arrays
lcoords    = dict(mon=mons,nlat=nlat,nlon=nlon)
da_lbdd    = xr.DataArray(lbd_d_all,coords=lcoords,dims=lcoords,name="lbd_d")

taucoords  = dict(mon=mons,z_t=z,nlat=nlat,nlon=nlon)
da_tau     = xr.DataArray(tau_est_all,coords=taucoords,dims=taucoords,name="tau")

acfcoords  = dict(mon=mons,lag=lags,z_t=z,nlat=nlat,nlon=nlon)
da_acf_est = xr.DataArray(acf_est_all,coords=acfcoords,dims=acfcoords,name="acf_est")
da_acf_mon = xr.DataArray(acf_mon_all,coords=acfcoords,dims=acfcoords,name="acf_mon")


ds_out     = xr.merge([da_lbdd,da_tau,da_acf_est,da_acf_mon])
edict      = proc.make_encoding_dict(ds_out)
savename   = "%sCESM1_HTR_FULL_lbd_d_params_%s_detrend%s_lagmax%i_ens%02i.nc" % (outpath,vname,detrend,lagmax,e+1)

ds_out.to_netcdf(savename,encoding=edict)
#%%
# for o in range(nlonr):
    
#     lonf = lonr[o]
#     if lonf < 0:
#         lonf += 360 # Convert to Degrees East
    
#     for a in range(nlatr):
        
#         latf = latr[a]
        
#         # Find closest lat
        
        
        # Check Mask
fig,ax = plt.subplots(1,1,subplot_kw={'projection':ccrs.PlateCarree()})

ax.scatter(tlon,tlat,c=lbd_d_all[5,:,:])
#ax.scatter(tlon,tlat,c=tau_est_all[0,0,:,:])
ax.coastlines()
        


#np.zeros(())

#lbd_d = np.zeroes(()) # Estimated Detrainment Damping



#%%

nc   = "%sCESM1_HTR_FULL_lbd_d_params_%s_detrend%s_lagmax%i_ens%02i.nc" % (outpath,vname,detrend,lagmax,e+1)
ds   = xr.open_da(nc)




#%% Load the data (This was for a single level file)

e = 1
v = 0
z = 0

# Variable Loop
# for v in range(2):

vname = vnames[v]
# Ens Loop
#   for e in range(nens):

    
# For a single level, load all ensemble members
ds_all = []
for e in tqdm(range(nens)):
    nc = "%s%s_NATL_ens%02i.nc" % (outpath,vname,e+1)
    st = time.time()
    ds = xr.open_dataset(nc).load()
    #print("Loaded data in %.2fs" % (time.time()-st))
    ds_all.append(ds)
ds_all = xr.concat(ds_all,dim='ens') # [Ens x Time x Lat x Lon]

# Read out variable
ds_var = ds_all[vname]

# Deseason
ds_anom = proc.xrdeseason(ds_var)

# Detrend
ds_anom = ds_anom - ds_anom.mean('ens')

# Load out some variables for ease
tlon = ds_all.TLONG.isel(ens=0).values
tlat = ds_all.TLAT.isel(ens=0).values

# Debugging, plot the region, scatterplots, etc
if debug:
    fig,ax = plt.subplots(1,1,subplot_kw={'projection':ccrs.PlateCarree()})
    #ax.scatter(ds_all.TLONG,ds_all.TLAT) # Cropped region appears to be correct
    #ax.pcolormesh(ds_all.TLONG.isel(ens=0),ds_all.TLAT.isel(ens=0),ds_anom.isel(ens=0,time=0).values) # Pcolormesh is working
    ax.scatter(ds_all.TLONG.isel(ens=0),ds_all.TLAT.isel(ens=0),c=ds_anom.isel(ens=0,time=0).values) # Scatter also appears correct
    ax.coastlines()
    plt.show()
    
#%% Load Mixed Layer Depth (for each ensemble member)


#h     = 

def maskmaker(arr,sumdims):
    mask = np.sum(arr.copy(),axis=sumdims)
    mask[~np.isnan(mask)] = 1
    return mask

#%% Check Dimensions, and repair where necessary

invar                = ds_anom.values # [Ens x time x TLat x TLon]
nens,ntime,nlat,nlon = invar.shape

# Check Ensemble Dimension
# Q: Is there an ensemble member for which all points are nan?
chkdim    = 0
sumdim    = 1
sumvar    = np.sum(invar,sumdim,keepdims=True) # ens x 1 x lat x lon (summed along time)

# Sum along lat,lon
#sumvar_pts = np.nansum(sumvar,axis=(1,2))
#sum_isnan  = np.isnan(sumvar_pts.flatten())

#np.take(sumvar,ii,axis=chkdim)
#sum_isnan = [np.all(np.isnan())] 








#%% Repair file


# For each month, compute the lagged autocorrelation



mask = maskmaker(invar,(1))[0,:,:]

if debug:
    # Check Mask
    fig,ax = plt.subplots(1,1,subplot_kw={'projection':ccrs.PlateCarree()})
    plt.scatter(tlon,tlat,mask),plt.show()
    ax.coastlines()


acfmon = []
for im in range(12):
    
    
