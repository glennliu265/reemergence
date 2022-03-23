#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calculate Gridwise (by lag and depth) Correlation and Make a Plot

Created on Thu Nov  4 11:12:51 2021

@author: gliu

"""

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import time
import cmocean as cmo

import sys
sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/03_Scripts/")
sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/03_Scripts/stochmod/model/")
from amv import proc,viz
import scm

#%% Set Paths

datpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/"
figpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/02_Figures/Scrap/"
proc.makedir(figpath)

lonf = -30
latf = 50
ncname  = "CESM1LE_Ens01_TEMP_lon330_lat50.nc"

mons3   = proc.get_monstr(3)

#%% Load Data

# Load dataset (Potential Temperature)
ds    = xr.open_dataset(datpath+ncname)
z     = ds.z_t.values/100 # Convert cm --> m!
temp  = ds.TEMP.values
times = ds.time.values

# Convert times
times = proc.cftime2str(times)
t     = np.arange(0,len(times))

# Quick plot (temperature over first plott times and plotz depths)
plott = 24
plotz = 30
fig,ax = plt.subplots(1,1,figsize=(12,4))
cf = ax.contourf(t[:plott],z[:plotz],temp[:plott,:plotz].T,cmap=cmo.cm.thermal)
cb = fig.colorbar(cf,ax=ax)
cb.set_label("Potential Temperature ($\degree C$)")
plt.gca().invert_yaxis()
ax.set_xlabel("Timestep (Months)")
ax.set_ylabel("Depth (m)")
ax.set_title("Temperature for %s to %s" % (times[0],times[plott]))

# Quickly calculate monthly anomalies
climavg,tsyrmon = proc.calc_clim(temp,0,returnts=True)
tempa = tsyrmon - climavg[None,...]
tempaplt = tempa.reshape(np.prod(tempa.shape[:2]),tempa.shape[2]) # For plotting


# Quick plot (mean seasonal cycle)
plott = 24
plotz = 35
fig,ax = plt.subplots(1,1,figsize=(8,4))
cf = ax.contourf(mons3,z[:plotz],climavg[:,:plotz].T,cmap=cmo.cm.thermal)
cb = fig.colorbar(cf,ax=ax)
cb.set_label("Potential Temperature ($\degree C$)")
plt.gca().invert_yaxis()
ax.set_xlabel("Timestep (Months)")
ax.set_ylabel("Mean Seasonal Cycle")

# Quick plot Anom
plott = 48
plotz = 30
fig,ax = plt.subplots(1,1,figsize=(12,4))


cb = fig.colorbar(cf,ax=ax)
cb.set_label("Potential Temperature ($\degree C$)")
plt.gca().invert_yaxis()
ax.set_xlabel("Timestep (Months)")
ax.set_ylabel("Depth (m)")
ax.set_title("Temperature Anomaly for %s to %s" % (times[0],times[plott]))

#%% Time to calculate the reemergence

# Load the mixed layer depth for this point

# User Inputs
lags   = np.arange(0,37,1)
kmonth = 1 # The base month
kz     = 0 # The base level
invar  = tempa.transpose(1,0,2) # [mon, year, depth] 
p      = 0.05
tails  = 1
neff   = True

# 
print("Calculating lagged correlation w.r.t. " +
      "Temperature Anomaly at %i m during %s" % (z[kz],mons3[kmonth]))

# Get base timeseries
base_ts = invar[...,kz] # Index for surface level

# Preallocate, and Calculate Lagged correlation
nz      = len(z)
nlags   = len(lags) 
nt      = invar.shape[1]
lagcorr = np.zeros((nz,nlags))*np.nan
lagconf = np.zeros((nz))*np.nan
for iz in range(nz):
    lag_ts = invar[...,iz]
    if np.any(np.isnan(lag_ts)):
        continue
    
    if neff:
        flatts = base_ts.flatten()
        r1   = np.corrcoef(flatts[:-1],flatts[1:])[0,1]
        n_in =  nt*(1-r1)/(1+r1)# reduced DOF
    else:
        n_in = nt # DOF is just number of years
    
    output = proc.calc_lagcovar(base_ts,lag_ts,lags,kmonth+1,detrendopt=1) # [nlag,]
    lagcorr[iz,:]   = output.copy()
    #lagconf[iz,:,:] = proc.calc_conflag(output,conf,tails,n_in)
    lagconf[iz] = proc.ttest_rho(output,p,tails,n_in)

#%% Load mixed layer depth data
mconfig = 'FULL_HTR'
input_path = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/01_Data/model_input/"
h          = np.load(input_path+"%s_HMXL_hclim.npy" % mconfig) # Climatological MLD
kprevall   = np.load(input_path+"%s_HMXL_kprev.npy" % mconfig) # Entraining Month

lon,lat = scm.load_latlon()

klon,klat = proc.find_latlon(lonf,latf,lon,lat)
mldpt =h[klon,klat,:]

looph = proc.tilebylag(kmonth,mldpt,lags)
#%% Visualize things

xtk2 = np.arange(0,nlags,3)

# TO DO
# add the monthly mean MLD cycle as a dotted line
# add significance

mask =  lagcorr >= lagconf[0]

#fig,ax = plt.subplots(1,1)
#ax.plot(lagconf[0,:,1],ax'upper')

# Plot Lagged Correlation
clvl=np.arange(-1,1.05,.05)
plotz = 30
fig,ax = plt.subplots(1,1,figsize=(8,4))
cf = ax.contourf(lags,z[:plotz],lagcorr[:plotz,:],levels=clvl,cmap=cmo.cm.balance)
cl = ax.contour(lags,z[:plotz],lagcorr[:plotz,:],levels=clvl,colors="k",linewidths=0.5)
ax.clabel(cl,fontsize=8)
ax.plot(lags,looph,color='k',ls='dashed')

viz.plot_mask(lags,z[:plotz],mask[:plotz,:].T,reverse=True,color='k',markersize=0.5)
cb = fig.colorbar(cf,ax=ax)
cb.set_label("Correlation")
plt.gca().invert_yaxis()
ax.set_xlabel("Lag (Months) from %s" % mons3[kmonth])
ax.set_xticks(xtk2)
ax.set_ylabel("Depth (meters)")
ax.grid(True,ls='dotted')
ax.set_title("Potential Temperature Anomaly Lagged Correlation \n"+
             "with %s, at %i m" % (mons3[kmonth],z[kz]))
plt.savefig("%s2D_Reemergence_50N30W.png"%(figpath),dpi=200,bbox_inches='tight')







