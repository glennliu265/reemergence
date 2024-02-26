#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Visualize Ekman Forcing Output from calc_ekman_advetion_htr

Created on Mon Feb 12 19:59:52 2024

@author: gliu
"""

import numpy as np
import xarray as xr
import sys
import time
import matplotlib.pyplot as plt

#%% Import Custom Modules
amvpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/03_Scripts/" # amv module
scmpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/03_Scripts/stochmod/model/"

sys.path.append(amvpath)
sys.path.append(scmpath)

from amv import proc,viz
import scm
import amv.loaders as dl
import yo_box as ybx



#%% Load some files

figpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/02_Figures/20240301/"
rawpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/proc/CESM1/NATL_proc/"
proc.makedir(figpath)

vname   = "SSS"

if vname == "SSS":
    vunits = "psu/mon"
elif vname == "SST":
    vunits = "W/m2"
    
dtplot = 3600*24*30

#%%
# Load Ekman Forcing
fp1          = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/proc/model_input/forcing/"
ncek         = "CESM1_HTR_FULL_Qek_%s_NAO_nomasklag1_nroll0_NAtl_EnsAvg.nc" % vname
dsek         = xr.open_dataset(fp1+ncek)

# Load Output of EOF Analysis
nceof        = "EOF_Monthly_NAO_EAP_Fprime_nomasklag1_nroll0_NAtl.nc"
dseof        = xr.open_dataset(rawpath+nceof).load()
varexp       = dseof.varexp

# Load Gradients and TAU
ncgrad       = "CESM1_HTR_FULL_Monthly_gradT2_%s.nc" % vname
#nctaux       = "CESM1LE_TAUX_NAtl_19200101_20050101_bilinear.nc" # Need to save EOF component of Tau at some point....
#nctauy       = "CESM1LE_TAUY_NAtl_19200101_20050101_bilinear.nc" # Need to save EOF component of Tau at some point....
dsgrad       = xr.open_dataset(rawpath+ncgrad).load()#.mean('ens')
dsgrad_emean = dsgrad.mean('ensemble')

# Load NAO-related tau 
savename = "%sCESM1_HTR_FULL_Monthly_TAU_NAO_nomasklag1_nroll0.nc" % (rawpath)
nao_taus = xr.open_dataset(savename)
dstaux = nao_taus.TAUX.mean('ens')
dstauy = nao_taus.TAUY.mean('ens')

# Maybe it would also be helpful to have the mean temperature and salinity gradient
ncvar        = "CESM1LE_%s_NAtl_19200101_20050101_bilinear.nc" % vname
dsvar        = xr.open_dataset(rawpath+ncvar).load()
dsvar_mean   = dsvar.groupby('time.month').mean('time')[vname]


# Load Ekman Currents
ncekcur      = "CESM1_HTR_FULL_Uek_NAO_nomasklag1_nroll0_NAtl.nc"
dsek2        = xr.open_dataset(fp1+ncekcur).mean('ensemble')



#%% Initialize figure for consistency

#%% PLot Ekman Forcing and Currents for a Month, Mode (Ens. Avg.)

im    = 1
n     = 0
if vname == "SSS":
    vmax  = 5e-2
    cfactor = dtplot
elif vname == "SST":
    vmax  = 40
    cfactor = 1
if n == 0:
    windscale = 0.15
    tauscale  = 1.3
else:
    windscale = 0.05
    tauscale  = 0.9


scale = 55#155

#windscale = 0.05
xint  = 2
yint  = 2
mons3 = proc.get_monstr()
bbox  = [-80,0,20,65]

if vname == "SSS":
    mlevels= np.arange(34,38.2,0.2)
else:
    mlevels = np.arange(230,330,2) # Not sure if in Kelvin


fig,ax = viz.geosubplots(1,1,)
ax     = viz.add_coast_grid(ax,bbox=bbox,fill_color='lightgray',line_color='gray')

# Select what to plot
x = dsek.lon.values[::xint]
y = dsek.lat.values[::yint]

U = dsek.Uek.isel(mon=im,mode=n).values[::yint,::xint]
V = dsek.Vek.isel(mon=im,mode=n).values[::yint,::xint]

taux = dstaux.isel(mon=im,mode=n).values[::yint,::xint]
tauy = dstauy.isel(mon=im,mode=n).values[::yint,::xint]

Q = dsek.Qek.isel(mon=im,mode=n) * cfactor

mpat = dsvar_mean.isel(month=im).mean('ensemble')
vexp = varexp.isel(mon=im,mode=n).mean('ens')


qv1  = ax.quiver(x,y,U,V,scale=windscale)
pcm  = ax.pcolormesh(dsek.lon,dsek.lat,Q,zorder=-1,cmap='cmo.balance',vmin=-vmax,vmax=vmax)
cb   = fig.colorbar(pcm,ax=ax,fraction=0.025,pad=0.01)

qv2  = ax.quiver(x,y,taux*-1,tauy*-1,scale=tauscale,color="gray")

cl = ax.contour(mpat.lon,mpat.lat,mpat,colors="k",linewidths=0.75,levels=mlevels,zorder=-1)
ax.clabel(cl)

cb.set_label("$Q_{ek}$ (%s)" % vunits)

ax.set_title("Ekman Forcing, Wind Stress (gray vectors), Currents (black vectors), Mean %s (contours) \n %s | EOF Mode %i (%.2f) | CESM1 Historical Ens Avg." % (vname,mons3[im],n+1,vexp))
savename = "%sQek_Advection_Pattern_CESM1_%s_EnsAvg_Mon%02i_Mode%03i.png" % (figpath,vname,im+1,n+1)
plt.savefig(savename,dpi=150,bbox_inches='tight')

#%% Section below here is still under construction

yr1 = proc.get_xryear()

# #%% Take Seasonal Means

st      = time.time()
dsin    = [dsek.Qek,dsek.Uek,dsek.Vek,dsvar_mean.rename({'month':'mon'}),varexp]
dsin    = [ds.assign_coords({'mon':yr1}).rename({'mon':'time'}) for ds in dsin]
ds_savg = [ds.groupby('time.season').mean('time') for ds in dsin]
Qek,Uek,Vek,varmean,varexp = ds_savg
print("Seasonal Averages computed in %.2fs" % (time.time()-st))

#ds_savg = [ds.groupby('time.season').mean('time') for ds in dsin] 
#%% Plot Seasonally Averaged Patterns

n = 2
sorder  = ['DJF','MAM','JJA','SON']
if vname == "SST":
    scale   = 0.1
    vmax    = 25
    cfactor = 1
else:
    scale = 0.1
    vmax    = 5e-2
    cfactor = dtplot
    
    

fig,axs = viz.geosubplots(1,4,figsize=(16,6),constrained_layout=True)

for a,ax in enumerate(axs):
    
    blabels = [0,0,0,1]
    if a == 0:
        blabels[0] = 1
    ax     = viz.add_coast_grid(ax,bbox=bbox,fill_color='lightgray',blabels=blabels)
    
    # Select what to plot
    seas = sorder[a]
    x = dsek.lon.values[::xint]
    y = dsek.lat.values[::yint]
    U = Uek.isel(mode=n).sel(season=seas).values[::yint,::xint]
    V = Vek.isel(mode=n).sel(season=seas).values[::yint,::xint]
    Q = Qek.isel(mode=n).sel(season=seas) * cfactor
    mpat = varmean.sel(season=seas).mean('ensemble')
    vexp = varexp.isel(mode=n).mean('ens').sel(season=seas)
    
    # Do Plotting
    ax.quiver(x,y,U,V,scale=scale)
    pcm = ax.pcolormesh(dsek.lon,dsek.lat,Q,zorder=-1,cmap='cmo.balance',vmin=-vmax,vmax=vmax)
    cl = ax.contour(mpat.lon,mpat.lat,mpat,colors="k",linewidths=0.75,levels=mlevels,zorder=-1)
    ax.clabel(cl)
    
    ax.set_title("%s (%.2f)" % (seas,vexp))
cb=fig.colorbar(pcm,ax=axs.flatten(),fraction=0.0075,pad=0.01)
cb.set_label("$Q_{ek}$ (%s)" % vunits)
viz.add_ylabel("Mode %i" % (n+1),ax=axs[0],x=-0.15)

savename = "%sQek_Advection_Pattern_CESM1_%s_EnsAvg_Mode%03i_SeasonalAvg.png" % (figpath,vname,n+1)
plt.savefig(savename,dpi=150,bbox_inches='tight')


   # ax.set_title("Ekman Forcing (colors) and Currents (vectors), Mean SSS (contours) \n %s | EOF Mode %i (%.2f) | CESM1 Historical Ens Avg." % (mons3[im],n+1,vexp))
   # savename = "%sQek_Advection_Pattern_CESM1_%s_EnsAvg_Mon%02i_Mode%03i.png" % (figpath,vname,im+1,n+1)
   # plt.savefig(savename,dpi=150,bbox_inches='tight')
#%% Load and check Ekman Currents






#%%
