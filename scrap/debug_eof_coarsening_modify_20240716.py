#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 17:42:00 2024

@author: gliu
"""


import numpy as np
import xarray as xr
import sys
import time
import matplotlib.pyplot as plt


# ----------------------------------
#%% Import custom modules and paths
# ----------------------------------

# Indicate the Machine!
machine = "stormtrack"

# First Load the Parameter File
sys.path.append("../")
import reemergence_params as rparams

# Paths and Load Modules
pathdict   = rparams.machine_paths[machine]

sys.path.append(pathdict['amvpath'])
sys.path.append(pathdict['scmpath'])
from amv import proc,viz
import scm
import amv.loaders as dl
import yo_box as ybx

# Set needed paths
figpath     = pathdict['figpath']
proc.makedir(figpath)
input_path  = pathdict['input_path']
output_path = pathdict['output_path']
procpath    = pathdict['procpath']


#%% Indicate Paths

#figpath   = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/02_Figures/20240411/"
datpath   = pathdict['raw_path']
outpath   = pathdict['input_path']+"forcing/"


def rmse(ds):
    return ((ds**2).sum('mode'))**(0.5)

#%%
    
# Load the Coarsned EOF Analysis
dataset   = "cesm1le_5degbilinear"
dampstr   = "nomasklag1"
rollstr   = "nroll0"
regstr    = "Global"
dp1       = "/stormtrack/data3/glliu/01_Data/02_AMV_Project/01_hfdamping/output/proc/"
#nceof     = dp1 + "%s_EOF_Monthly_NAO_EAP_Fprime_%s_%s_%s.nc" % (dataset,dampstr_qnet,rollstr,regstr)

ds1       = xr.open_dataset(nceof).load()

dp2       = "/stormtrack/data3/glliu/01_Data/02_AMV_Project/03_reemergence/proc/CESM1/NATL_proc/"
nceof2    = "EOF_Monthly_NAO_EAP_Fprime_nomasklag1_nroll0_NAtl.nc"
ds2       = xr.open_dataset(dp2+nceof2).load()
#dp1 = ""



#%% First Check the EOF Pattern (This does not seem to be the case)

eof1 = rmse(ds1.eofs)

eof2 = rmse(ds2.eofs)


im  = 1
ens = 0

bbox = [-80,0,20,65]
fig,axs=viz.geosubplots(1,2,)

ax = axs[0]
ax.set_extent(bbox)
ax.coastlines()
eof1.isel(ens=ens,mon=im).plot(vmin=0,vmax=80,ax=ax)#,plt.show()

ax = axs[1]
ax.set_extent(bbox)
ax.coastlines()
eof2.isel(ens=ens,mon=im).plot(vmin=0,vmax=80,ax=ax)#,plt.show()
plt.show()

#%% Next Check the PC

fig,ax = plt.subplots(1,1,figsize=(12,4.5))
ax.plot(ds1.pcs.isel(mode=1,mon=2,ens=1),label="Coarse")
ax.plot(ds2.pcs.isel(mode=1,mon=2,ens=1),label="Original")
ax.legend()

#%% Check the EVAP


dslhflx2 = "CESM1_HTR_FULL_Eprime_timeseries_LHFLXnomasklag1_nroll0_NAtl.nc"
lhori = xr.open_dataset(dp2+dslhflx2).load()

dslhflx1 = "cesm1le_htr_5degbilinear_Eprime_timeseries_cesm1le5degLHFLX_nroll0_Global.nc"
lhnew = xr.open_dataset(dp1+dslhflx1).load()


# Compute Monthly Variance

lhori_monvar = lhori.groupby('time.month').std('time')
lhnew_monvar = lhnew.groupby('time.month').std('time')

#%%



fig,ax = plt.subplots(1,1,figsize=(12,4.5))
ax.plot(lhnew.Eprime.isel(ens=0).sel(lon=330,lat=50,method='nearest'),label="Coarse")
ax.plot(lhori.LHFLX.isel(ens=0).sel(lon=-30,lat=50,method='nearest'),label="Original")
ax.legend()
plt.show()


#%% Check monthly variance

fig,axs = plt.subplots(2,1,figsize=(12,4.5),constrained_layout=True,)



ax   = axs[0]
ds_in = lhnew_monvar.Eprime
for e in range(42):
    ax.plot(ds_in.isel(ens=e).sel(lon=330,lat=50,method='nearest'))


ax = axs[1]
ds_in = lhori_monvar.LHFLX
for e in range(42):
    ax.plot(ds_in.isel(ens=e).sel(lon=-30,lat=50,method='nearest'))

plt.show()


#%%


#%% Check the Regressed+ corrected Eprime

fp1   = "/stormtrack/data3/glliu/01_Data/02_AMV_Project/03_reemergence/proc/model_input/forcing/"
nc_e1 = "cesm1le_htr_5degbilinear_Eprime_EOF_cesm1le5degLHFLX_nroll0_NAtl_corrected.nc"
#"cesm1le_htr_5degbilinear_Eprime_EOF_cesm1le5degLHFLX_nroll0_NAtl_corrected_EnsAvg.nc"
ds_e1 = xr.open_dataset(fp1+nc_e1).load().LHFLX.drop_duplicates('lon')

nc_e2 = "CESM1_HTR_FULL_Eprime_EOF_nomasklag1_nroll0_NAtl_corrected.nc"
#"CESM1_HTR_FULL_Eprime_EOF_nomasklag1_nroll0_NAtl_corrected_EnsAvg.nc"

ds_e2 = xr.open_dataset(fp1+nc_e2).load().LHFLX


#%%
imode = 0

fig,ax = plt.subplots(1,1,figsize=(12,4.5))
ax.plot(ds_e1.sel(lon=-30,lat=50,method='nearest').isel(mode=imode),label="Coarse")
ax.plot(ds_e2.sel(lon=-30,lat=50,method='nearest').isel(mode=imode),label="Original")
ax.legend()
plt.show()

#%% Plot Spatial Patterns

imode = 0
imon  = 1
iens  = 0


ds_e1.isel(mode=imode,mon=imon,ens=iens).plot(),plt.show()
ds_e2.isel(mode=imode,mon=imon,ens=iens).plot(),plt.show()

# Ensemble mean is a lot smaller for the coarsened results. lets try t make a plot for each ensemble member
ds_e1.isel(mode=imode,mon=imon,).mean('ens').plot(vmin=-40,vmax=40),plt.show()
ds_e2.isel(mode=imode,mon=imon,).mean('ens').plot(vmin=-40,vmax=40),plt.show()


#%%
import cartopy.crs as ccrs

fig,axs = plt.subplots(6,7,figsize=(22,22),constrained_layout=True,
                       subplot_kw={'projection':ccrs.PlateCarree()})

for e in range(42):
    
    ax = axs.flatten()[e]
    ax.set_title("Ens%02i" % e)
    plotvar = ds_e1.isel(mode=imode,mon=imon,ens=e)
    pcm = ax.pcolormesh(plotvar.lon,plotvar.lat,plotvar,vmin=-30,vmax=30)
    fig.colorbar(pcm,ax=ax)
plt.show()
    
#%% Examine Tineseries at a point
# It seems that the Regressedf values of the EOF patterns are already quite different at that point

fig,axs = plt.subplots(2,1,figsize=(12,4.5),constrained_layout=True,)

ax = axs[0]
for e in range(42):
    ax.plot(ds_e1.isel(mode=imode,ens=e).sel(lon=-30,lat=50,method='nearest'))


ax = axs[1]
for e in range(42):
    ax.plot(ds_e2.isel(mode=imode,ens=e).sel(lon=-30,lat=50,method='nearest'))

plt.show()


#%% Check the regression patterns (pre correction)

# 5 deg data
outpath       = "/stormtrack/data3/glliu/01_Data/02_AMV_Project/03_reemergence/proc/model_input/forcing/"
ncevap_eof    = "cesm1le_htr_5degbilinear_Eprime_EOF_cesm1le5degLHFLX_nroll0_Global.nc"#
ncprec_eof    = "cesm1le_htr_5degbilinear_PRECTOT_EOF_cesm1le5degLHFLX_nroll0_Global.nc" #

#cesm1le_htr_5degbilinear_Eprime_EOF_cesm1le5degLHFLX_nroll0_Global.nc
#cesm1le_htr_5degbilinear_Eprime_EOF_cesm1le5degLHFLX_nroll0_Global.nc  
ds_evapreg1 = xr.open_dataset(outpath + ncevap_eof).load().Eprime.drop_duplicates('lon')

# (86, 42, 12, 37, 73)
# ('mode', 'ens', 'mon', 'lat', 'lon')
# Original Data

ncevap_eof2 = "CESM1_HTR_FULL_Eprime_EOF_nomasklag1_nroll0_NAtl.nc"
ds_evapreg2 = xr.open_dataset(outpath + ncevap_eof2).load().LHFLX
 # 

#%% Plot Things (Seasonal Cycle at a point)

imode = 0#"ALL"#0
imon   = 0


fig,axs = plt.subplots(2,1,figsize=(12,4.5),constrained_layout=True,)

ax   = axs[0]
if imode == "ALL":
    ds_in = rmse(ds_evapreg1)
else:
    ds_in = ds_evapreg1.isel(mode=imode)
for e in range(42):
    ax.plot(ds_in.isel(ens=e).sel(lon=330,lat=50,method='nearest'))
    
ax.plot(ds_in.sel(lon=330,lat=50,method='nearest').mean('ens'),c='k',lw=4)
#ax.set_ylim([-40,40])


ax = axs[1]
if imode == "ALL":
    ds_in = rmse(ds_evapreg2)
else:
    ds_in = ds_evapreg2.isel(mode=imode)
for e in range(42):
    ax.plot(ds_in.isel(ens=e).sel(lon=-30,lat=50,method='nearest'))
ax.plot(ds_in.sel(lon=-30,lat=50,method='nearest').mean('ens'),c='k',lw=4)
#ax.set_ylim([-40,40])
plt.show()


#%% Partway run of Correct SSS Forcing script



fig,axs = plt.subplots(2,1,figsize=(12,4.5),constrained_layout=True,)

ax   = axs[0]

ds_in = da_eofs_filt_old.isel(mode=imode).drop_duplicates('lon')
for e in range(42):
    ax.plot(ds_in.isel(ens=e).sel(lon=-30,lat=50,method='nearest'))
    
ax.plot(ds_in.sel(lon=-30,lat=50,method='nearest').mean('ens'),c='k',lw=4)
#ax.set_ylim([-40,40])

ax   = axs[1]

ds_in = da_eofs_filt.isel(mode=imode).drop_duplicates('lon')
for e in range(42):
    ax.plot(ds_in.isel(ens=e).sel(lon=330,lat=50,method='nearest'))
    
ax.plot(ds_in.sel(lon=330,lat=50,method='nearest').mean('ens'),c='k',lw=4)
#ax.set_ylim([-40,40])

plt.show()


#%% Check the pattern




