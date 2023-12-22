#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Animate Re-emergence At A Point

Created on Wed Dec 20 23:07:31 2023

@author: gliu

"""

import numpy as np
import scipy as sp
import xarray as xr
import sys
from tqdm import tqdm

import cmocean as cmo

import matplotlib.pyplot as plt


#%% Import Modules

amvpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/03_Scripts/" # amv module
scmpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/03_Scripts/stochmod/model/"

sys.path.append(amvpath)
sys.path.append(scmpath)

from amv import proc,viz
import scm
import amv.loaders as dl
import cvd_utils as cvd

#%% Load data

# Load the data (already anomalized)
datpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/01_Data/CESM_proc/" 
ncname  = "TS_anom_PIC_FULL.nc"
ds      = xr.open_dataset(datpath+ncname).load()
ds      = proc.format_ds(ds,) # Flip to Lon 180

figpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/02_Figures/20231222/"

# Load Mixed Layer Depth
ncnameh = "HMXL_FULL_PIC_bilinear.nc"
dsh     = xr.open_dataset(datpath+ncnameh)
mask    = dsh.HMXL.values


# Make and save mask
mask    = mask.sum(0)
mask    = np.where(~np.isnan(mask),1,np.nan)
savepath = "/Users/gliu/Downloads/02_Research/01_Projects/04_Predict_AMV/03_Scripts/CESM_data/Masks/"
np.save("%sCESM_PIC_FULL_HMXL_LandMask180.npy" % savepath,mask)

# Apply Mask to TS
ds    = ds * mask[None,:,:]

#%% Select Values

# Select Region
bbox = [-90,0,0,75] # Keep changing till you get the region you want
dsr  = proc.sel_region_xr(ds,bbox)
dsr.TS.isel(time=0).plot()

#%% Load Values

# Load the data to arrays
ts  = dsr.TS.values
lon = dsr.lon.values
lat = dsr.lat.values

# Select a point
lonf      = -30
latf      = 50
klon,klat = proc.find_latlon(lonf,latf,lon,lat)
basets    = ts[:,klat,klon] # Get Base Timeseries


locfn,loctitle =proc.make_locstring(lonf,latf,)

#%% Compute Monthly Autocorrelation (It takes about 5 minutes to run)

lags            = np.arange(0,121,1)
nlags           = len(lags)

ntime,nlat,nlon = ts.shape
nyrs            = int(ntime/12)
tsin            = ts.reshape(ntime,nlon*nlat)
nandict         = proc.find_nan(tsin,0,return_dict=True)

tsclean = nandict['cleaned_data']
okpts   = nandict['ok_indices']
npts    = tsclean.shape[1]


basein   = basets.reshape(nyrs,12).T # Mon x Year

lagcovar = np.zeros((nlags,12,npts)) # [baselag,pt]
for pt in tqdm(range(npts)):
    
    lagts = tsclean[:,pt]
    lagin = lagts.reshape(nyrs,12).T
    
    for im in range(12):
        lagcovar[:,im,pt] = proc.calc_lagcovar(basein,lagin,lags,im+1,1,debug=False)
#% Reshape the file
lagcovar_out = np.zeros((nlags,12,nlat*nlon)) * np.nan
lagcovar_out[:,:,okpts] = lagcovar.copy()
lagcovar_out = lagcovar_out.reshape(nlags,12,nlat,nlon)

#%% Save The Autocorrelation


coords = {
    "lags" : lags,
    "basemonth" : np.arange(12),
    "lat" : lat,
    "lon" : lon,
    }
da = xr.DataArray(lagcovar_out,coords=coords,dims=coords,name="corr")
encoding_dict = {'corr':{'zlib':True}}

savepath   = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/proc/point_correlation/"
savenc = "%sPointAC_%s.nc" % (savepath,locfn)
da.to_netcdf(savenc,encoding=encoding_dict)



#%% Load the autocorrelation

savepath   = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/proc/point_correlation/"
savenc     = "%sPointAC_%s.nc" % (savepath,locfn)

daload       = xr.open_dataset(savenc)
lagcovar_out = daload.corr.values



#%% Test plot


rhocrit = proc.ttest_rho(0.05,2,nyrs)
mons     = proc.get_monstr(nletters=None)
mons3     = proc.get_monstr(nletters=3)

kmonth   = 1
lag      = 10
plotvar  = lagcovar_out[lag,kmonth,:,:]
bboxplot = bbox
fig,ax   = viz.geosubplots(1,1,figsize=(8,5))
ax       = viz.add_coast_grid(ax,bbox=bboxplot,fill_color="k")
pcm      = ax.pcolormesh(lon,lat,plotvar,vmin=-1,vmax=1,cmap='cmo.balance')
cb       = fig.colorbar(pcm,ax=ax,pad=0.01)

cl= ax.contour(lon,lat,plotvar,colors="gray",linewidths=.75,levels=[rhocrit,])
ax.clabel(cl)
#viz.plot_mask(lon,lat,plotvar.T > rhocrit,markersize=.5)

ax.plot(lonf,latf,marker="x",color="yellow",markersize=15)
ax.set_title(r"Correlation at Lag %03i (%s)" % (lag,mons3[(kmonth+lag)%12]),fontsize=16)


figpathpt = figpath+"/%s/month%02i/" % (locfn,im+1)
proc.makedir(figpathpt)
savename = "%slag%03i" % (figpathpt,lag)
plt.savefig(savename,dpi=150,bbox_inches='tight',transparent=True)
#%% Now Loop for lags

contour = False
dotmask = False

cmax   = 1
for lag in tqdm(range(nlags)):
    plotvar  = lagcovar_out[lag,kmonth,:,:]
    fig,ax   = viz.geosubplots(1,1,figsize=(8,5))
    ax       = viz.add_coast_grid(ax,bbox=bboxplot,fill_color="k")
    
    if contour:
        clvls    = np.linspace(-cmax,cmax,20)
        pcm      = ax.contourf(lon,lat,plotvar,levels=clvls,cmap='cmo.balance')
    else:
        pcm      = ax.pcolormesh(lon,lat,plotvar,vmin=-cmax,vmax=cmax,cmap='cmo.balance')
    cb       = fig.colorbar(pcm,ax=ax,pad=0.01)
    
    
    
    ax.plot(lonf,latf,marker="x",color="yellow",markersize=15)
    ax.set_title(r"Correlation at Lag %03i (%s)" % (lag,mons3[(kmonth+lag)%12]),fontsize=16)
    
    cl= ax.contour(lon,lat,plotvar,colors="gray",linewidths=.75,levels=[rhocrit,])
    if dotmask:
        viz.plot_mask(lon,lat,plotvar.T > rhocrit,markersize=.5)
    
    figpathpt = figpath+"/%s/month%02i_contour%i_dotmask%0i_cmax%.2f/" % (locfn,kmonth+1,contour,dotmask,cmax)
    if lag == 0:
        proc.makedir(figpathpt)
    savename = "%slag%03i" % (figpathpt,lag)
    plt.savefig(savename,dpi=150,bbox_inches='tight',transparent=False)
        
    
#%% Do the animation for the autocorrelation function

def plot_base():
    fig,ax = plt.subplots(1,1,figsize=(8,3),constrained_layout=True)
    ax.plot(lags,lagcovar_out[:,kmonth,klat,klon],
            c="cornflowerblue",lw=2.5)
    ax = viz.add_ticks(ax)
    ax.set_xlim([lags[0],lags[-1]])
    ax.axhline([rhocrit],ls='dashed',color='gray',lw=1)
    ax.axhline([0],ls='solid',color='k',lw=1)
    ax.set_xlabel("Lag (Months)")
    ax.set_ylabel("Correlation")
    ax.grid(False)
    
    return fig,ax

for lag in range(nlags):

    #Initialize Plot
    fig,ax = plot_base()
    
    # Plot Marker at Lag
    corrlag = lagcovar_out[lag,kmonth,klat,klon]
    label = "Lag %03i Correlation = %.3f" % (lag,corrlag)
    ax.axvline([lag],color="darkblue",lw=.75,label="label",alpha=0.7)
    ax.axhline([corrlag],color="darkblue",lw=.75,alpha=0.7)
    ax.plot(lag,corrlag,marker="o",mfc='none',color="darkblue",lw=0.75,markersize=5,markeredgewidth=0.5)
    #ax.legend()
    ax.set_title("Autocorrelation Function @ %s \n%s" % (loctitle,label))
    
    # Save it
    figpathpt = figpath+"/%s/month%02i_acf/" % (locfn,kmonth+1,)
    if lag == 0:
        proc.makedir(figpathpt)
    savename = "%slag%03i" % (figpathpt,lag)
    plt.savefig(savename,dpi=150,bbox_inches='tight',transparent=False)
    
    



