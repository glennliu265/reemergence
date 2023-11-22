#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Map the relationship between T and S in CESM1
Created on Wed Oct 11 07:50:38 2023

@author: gliu
"""



import xarray as xr
import sys
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
import cmocean as cmo
 
from tqdm import tqdm
#%% Modules

amvpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/03_Scripts/" # amv module
scmpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/03_Scripts/stochmod/model/"

sys.path.append(amvpath)
sys.path.append(scmpath)

from amv import proc,viz
import scm
import amv.loaders as dl


#%% User Edits

datpath  = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/proc/CESM1/"
outpath  = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/proc/"

varnames = ["TS","SSS","LHFLX"]
e        = 1 # Ensemble Number
# ncs= ["CESM1LE_SST_NAtl_19200101_20051201_bilinear_detrend1_regridNone.nc",
#       "CESM1LE_SSS_NAtl_19200101_20051201_bilinear_detrend1_regridNone.nc"]
bbox     = [-80,0,0,65]
#%% Load the data

ds_all   = []
vars_arr = []
for v in range(3):
    varname = varnames[v]
    ncname = "%shtr_%s_manom_detrend1_1920to2006_ens%02i.nc" % (datpath,varname,e+1)
    ds = xr.open_dataset(ncname).load()
    ds = proc.format_ds(ds)
    ds = proc.sel_region_xr(ds,bbox)
    
    ds_all.append(ds)
    vars_arr.append(ds[varname].values)


ts,sss,lhflx = vars_arr
#%% Check the values

# Load some values
times = ds.time.values
lat   = ds.lat.values
lon   = ds.lon.values
ntime,nlat,nlon = ts.shape
nyrs  = int(ntime/12)


# Make correction for SSS
for t in range(ntime):
    if np.all(np.isnan(sss[t,:,:].flatten())):
        print("All is nan at t=%i (%s). Replacing with zeros." % (t,times[t]))
        sss[t,:,:]=0

#%% Do reprocess (need to get in the form of [mon x lag x npts])

# Get variables
invar_base = vars_arr[0] # ts
invar_lag  = sss # sss
invars     = [invar_base,invar_lag]


# Make a shared mask
nanmask    = invar_base.sum(0) * invar_lag.sum(0)
nanmask[~np.isnan(nanmask)] = 1
invars     = [v*nanmask[None,...] for v in invars]

# Clean Data
nandicts  = []
cleandata = []
for v in range(2):
    invar = invars[v]
    
    # Remove NaN Pts
    invar   = invar.reshape(ntime,nlat*nlon)
    nandict = proc.find_nan(invar,0,return_dict=True) 
    okdata  = nandict['cleaned_data']
    oksize  = okdata.shape[1]
    cleandata.append(okdata.reshape(nyrs,12,oksize).transpose(1,0,2)) # [mon x yr x npts]

#%% Calculate Lag Correlation

basemonths = np.arange(1,13,1)
detrendopt = 1
lags       = np.arange(0,61)
nlags      = len(lags)

# Preallocate
corr_lags  = np.zeros((nlags,12,oksize))
corr_leads = np.zeros((nlags,12,oksize))

# Basemonth Loop
for bm in range(12):
    
    basemonth          = basemonths[bm]
    
    lagcorr            = proc.calc_lagcovar_nd(cleandata[1],cleandata[0],lags,basemonth,detrendopt) # first variable is lagged
    leadcorr           = proc.calc_lagcovar_nd(cleandata[0],cleandata[1],lags,basemonth,detrendopt) # first variable is lagged
    
    corr_lags[:,bm,:]  = lagcorr.copy()
    corr_leads[:,bm,:] = leadcorr.copy()

# Combine it
var_leadlag = np.concatenate([np.flip(corr_leads,axis=0),corr_lags[1:,...]],axis=0)
leadlags    = np.concatenate([-1*np.flip(lags),lags[1:]])

# Covariance
full_corrlags  = np.zeros((nlags,12,oksize))
full_corrleads = np.zeros((nlags,12,oksize))

#%% Debug Plot

fig,ax = plt.subplots(1,1)
ax.plot(lags,lagcorr[:,2233],label="SSS Lags")
ax.plot(np.flip(lags)*-1,np.flip(leadcorr[:,2233]),label="SSS Leads")
ax.plot(leadlags,var_leadlag[:,bm,2233],label="LeadLagConcat",ls='dashed')
ax.legend()
    
#%% Reshape variable and prepare to plot

nleadlag        = len(leadlags)
var_leadlag_fin = np.zeros((nleadlag,12,nlat*nlon)) * np.nan
var_leadlag_fin[:,:,nandict['ok_indices']] = var_leadlag
var_leadlag     = var_leadlag_fin.reshape(nleadlag,12,nlat,nlon)


#%% Plot leadlag by basemonth

p       = 0.05
tails   = 2
rhocrit = proc.ttest_rho(p,tails,nyrs)

klon,klat = proc.find_latlon(-30,50,lon,lat)
mons3     = proc.get_monstr(3)
bms       = [10,11,0,1,2]
fig,ax    = plt.subplots(1,1,figsize=(12,4))
for bm in bms:
    ax.plot(leadlags,var_leadlag[:,bm,klat,klon],label="%s" % (mons3[bm]))
ax.legend()
ax.axhline([0],lw=0.55,color="k")
ax.axhline([rhocrit],lw=0.55,color="r",label="$\rho$=%.2f" % rhocrit)
ax.set_xlim([-36,36])

#%% Plot map for a given basemonth

bm    = 1
lags  = [-3,-2,-1,0,1,2,3]

fig,axs = plt.subplots(1,7,figsize=(18,5),subplot_kw={'projection':ccrs.PlateCarree()},
                       constrained_layout=True)

for a in range(len(lags)):
    ax = axs[a]
    lag = lags[a]
    ilag = list(leadlags).index(lag)
    
    ax.coastlines()
    
    pcm = ax.pcolormesh(lon,lat,var_leadlag[ilag,bm,:,:],cmap="RdBu_r",vmin=-0.5,vmax=0.5)
    ax.set_title("SSS Lag " + str(lag)+" mons")

cb=fig.colorbar(pcm,ax=axs.flatten(),fraction=0.006)


#%%











