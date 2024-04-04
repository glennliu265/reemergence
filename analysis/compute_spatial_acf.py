#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Compute Spatial ACF

Created on Wed Apr  3 10:00:19 2024

@author: gliu

"""



from amv import proc, viz
import scipy.signal as sg
import yo_box as ybx
import amv.loaders as dl
import scm
import reemergence_params as rparams
import numpy as np
import xarray as xr
import sys
import time
import matplotlib.pyplot as plt
import matplotlib as mpl
import cartopy.crs as ccrs
import os
from tqdm import tqdm

# ----------------------------------
# %% Import custom modules and paths
# ----------------------------------

# Indicate the Machine!
machine = "Astraeus"

# First Load the Parameter File
sys.path.append("../")

# Paths and Load Modules
pathdict = rparams.machine_paths[machine]

sys.path.append(pathdict['amvpath'])
sys.path.append(pathdict['scmpath'])

# Set needed paths
figpath = pathdict['figpath']
proc.makedir(figpath)
input_path = pathdict['input_path']
output_path = pathdict['output_path']
procpath = pathdict['procpath']

#%% Import sea level utilities

sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/03_SeaLevel/03_Scripts/cluster_ssh/")
import slutil as sl

# ----------------------
#%% User Edits
# ----------------------

vnames  = ["SST","SSS"]

#PLotting
bboxplot=[-90,0,0,90]
proj    = ccrs.PlateCarree()

def preprocess(ds):
    # Remove the ensemble average
    dsa = ds - ds.mean('ensemble')
    dsa = proc.xrdeseason(dsa)
    return dsa
    

# ----------------------
#%% Load the variable
# ----------------------

# Load the netCDFs
ds_all  = []
datpath = procpath + "CESM1/NATL_proc/"
for vv in range(2):
    nc  = datpath + "CESM1LE_%s_NAtl_19200101_20050101_bilinear.nc" % vnames[vv]
    ds  = xr.open_dataset(nc)[vnames[vv]].load()
    ds_all.append(ds)

# Indicate the DS Anomalies
dsa_all = [preprocess(ds) for ds in ds_all]
var_all = [ds.values for ds in dsa_all]




# ------------------------
#%% Make a consistent mask
# ------------------------

# Check issue withs alinity
sss = var_all[1]
#ssssum = np.nansum(sss,(2,3))
for e in range(nens):
    for t in range(ntime):
        sssmap = sss[e,t,:,:]
        if np.all(np.isnan(sssmap)):
            print("ALl Nan at e=%i,t=%i" % (e,t))
            var_all[1][e,t] = np.zeros((nlat,nlon))
        
#[ix,iy] = np.where(np.isnan(ssssum))
    
masks = [np.sum(v,(0,1)) for v in var_all]
mask  = masks[0] * masks[1]
mask[~np.isnan(mask)] = 1


# ----------------------
#%% Compute the distance matrix
# ----------------------
vv = 1

    
lon    = dsa_all[0].lon.values
lat    = dsa_all[0].lat.values

nens,ntime,nlat,nlon = dsa_all[0].shape

distlev = np.hstack([np.arange(0,3000,200),np.arange(3000,8000,500)])

def efold_dist(rhoin,distin,thres=1/np.exp(1),tol=0.01,distgap=100,return_all=False):
    # Sort by Distance
    sortdist  = np.argsort(distin)
    sortrho   = rhoin[sortdist]
    sortdist1 = distin[sortdist]
    
    # Get indices within range
    idsel    = np.where( (sortrho>= (thres-tol)) * (sortrho<= (thres+tol)) )[0]
    distsel  = sortdist1[idsel]
    rhosel   = sortrho[idsel]
    
    # Filter by distance without grouping
    ddist    = distsel[1:] - distsel[:-1]
    kgap     = np.where(ddist > distgap)[0]
    if len(kgap) > 0:
        distsel  = distsel[:kgap[0]]
        rhosel   = rhosel[:kgap[0]]
    if return_all:
        return np.nanmean(distsel),distsel,rhosel
    return np.nanmean(distsel)
    
    



#rhos = np.zeros(())

debug=False

#decay_scale = np.zeros(())
    
for e in range(nens):
    
    # Take the ensemble member (with or without the mask)
    var_in = var_all[vv][e,:,:,:] * mask[None,:,:] # Time x Lat x Lon
    
    # Compute the disrtance matrix
    matout = sl.calc_matrices(var_in,lon,lat,return_all=True,return_dict=True)
    
    # Get the Matrix Rows
    rhopt,ptlon,ptlat  = sl.extract_point(matout['rho'],matout['okpts'],lon,lat,return_latlon=True)
    distpt,_,_         = sl.extract_point(matout['dist'],matout['okpts'],lon,lat,return_latlon=True)
    
    # For a point, get the distance
    #kk = 2899 # 330,50
    npts = rhopt.shape[0]
    for kk in tqdm(range(npts)):
        rhoin    =   rhopt[kk,:,:].flatten()
        distin   =   distpt[kk,:,:].flatten()
        dexpmean,distsel,rhosel=efold_dist(rhoin,distin,return_all=True)
        
        if (e == 0) and (kk == 0):
            decay_scale = np.zeros((nens,npts)) * np.nan
        
        decay_scale[e,kk] = dexpmean
        
        if debug:
            
            # sortdist = np.argsort(distin)
            
            # thres    = 1/np.exp(1)
            # tol      = 0.01
            # distgap  = 100
            # sortrho  = rhoin[sortdist]
            # sortdist1 = distin[sortdist]
            
            
            # # Get First Set of Indices
            # idsel    = np.where( (sortrho>= (thres-tol)) * (sortrho<= (thres+tol)) )[0]
            # distsel  = sortdist1[idsel]
            # rhosel   = sortrho[idsel]
            
            # # Filter by distance gap to avoid grouping
            # ddist = distsel[1:] - distsel[:-1]
            # kgap  = np.where(ddist > distgap)[0]
            # distsel = distsel[:kgap[0]]
            # rhosel  = rhosel[:kgap[0]]
            
            # Scatter to get the distance
            fig,ax = plt.subplots(1,1,constrained_layout=True)
            locfn,loctitle = proc.make_locstring(ptlon[kk],ptlat[kk],lon360=True)
            ax.set_title("Corr vs Distance @ %s, Ens %02i" % (loctitle,e+1))
            
            ax.scatter(distin[sortdist],rhoin[sortdist],s=12,alpha=.75,marker=".")
            ax.scatter(distsel,rhosel,s=12,alpha=.75,marker="+",c="limegreen")
            
            ax.set_xlabel("Distance (km)")
            ax.set_ylabel("Correlation")
            ax.axhline([1/np.exp(1)],color="k",lw=0.5,label="1/e",ls='dotted')
            ax.axhline([0],color="k",lw=0.5,label="",ls='solid')
            ax.axvline([dexpmean],lw=1,label="d=%.2f" % (dexpmean),c="midnightblue")
            
            ax.legend()
            savename = "%sTest_PtCorrvsScatter_%s_id%04i_%s.png" % (figpath,vnames[vv],kk,locfn)
            plt.savefig(savename,dpi=150,bbox_inches='tight',transparent=True)
    
    # Debug plot
    if debug:
        for kk in range(len(okpts)):
        #kk = 2899
            fig,ax,_ = viz.init_orthomap(1,1,bboxplot)
            ax       = viz.add_coast_grid(ax,bboxplot,line_color='dimgray',fill_color='lightgray')
            
            pcm = ax.pcolormesh(lon,lat,rhopt[kk,:,:],transform=proj,
                                vmin=-1,vmax=1,cmap="cmo.balance")
            cl  = ax.contour(lon,lat,distpt[kk,:,:],transform=proj,colors="k",
                             levels=distlev,linewidths=0.55)
            ax.clabel(cl,distlev[::2])
            ax.plot(ptlon[kk],ptlat[kk],marker="x",transform=proj,markersize=22,lw=2.5,c="cornflowerblue")
            fig.colorbar(pcm,ax=ax)
            locfn,loctitle = proc.make_locstring(ptlon[kk],ptlat[kk],lon360=True)
            ax.set_title("Correlation of %s anomalies @ %s, Ens %02i" % (vnames[vv],loctitle,e+1))
            savename = "%sTest_PtCorr_%s_id%04i_%s.png" % (figpath,vnames[vv],kk,locfn)
            plt.savefig(savename,dpi=150,bbox_inches='tight',transparent=True)

okpts = matout['okpts']
decay_scale_out          = np.zeros((nens,nlat*nlon))
decay_scale_out[:,okpts] = decay_scale
decay_scale_out          = decay_scale_out.reshape(nens,nlat,nlon)
coords                   = dict(ens=np.arange(1,43,1),lat=lat,lon=lon)
daout                    = xr.DataArray(decay_scale_out,coords=coords,dims=coords,name='decay_scale')
edict                    = {'decay_scale': {'zlib':True}}
savename_out             = "%sCESM1_HTR_Decay_SpaceScale_%s.nc" % (datpath,vnames[vv])
daout.to_netcdf(savename_out,encoding=edict)

#%% PLot Variables


fig,ax,_ = viz.init_orthomap(1,1,bboxplot)
ax       = viz.add_coast_grid(ax,bboxplot,line_color='dimgray',fill_color='lightgray')

plotvar = daout.mean('ens')
pcm = ax.pcolormesh(lon,lat,plotvar,transform=proj,
                    vmin=0,vmax=3000,cmap="inferno")


cb = fig.colorbar(pcm,ax=ax,orientation='horizontal',fraction=0.05,pad=0.01)
cb.set_label("Distance (km)")
ax.set_title("%s Anomaly E-folding Distance, Ens. Avg" % (vnames[vv]))

savename = "%sDistanceDecorr_Map_%s.png" % (figpath,vnames[vv])
plt.savefig(savename,dpi=150,bbox_inches='tight',transparent=True)
