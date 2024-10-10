#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Correlate residual components of the forcing as computed by calc_residual_eof


Created on Thu Sep 26 15:21:50 2024

@author: gliu
"""


import numpy as np
import xarray as xr
import sys
import time
import matplotlib.pyplot as plt

from tqdm import tqdm

# ----------------------------------
#%% Import custom modules and paths
# ----------------------------------

# Indicate the Machine!
machine = "Astraeus"

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
rawpath   = pathdict['raw_path']


#%% Random Functions

def reload_cc_matrix(rawpath):
    
    
    vnames = ["Fprime","LHFLX","PRECTOT","Qek_SST","Qek_SSS"]
    iiskip = 0
    ds_cc_all = []
    
    for vv in range(5):
        
        cc_inner = []
        for uu in range(5):
            
            if uu < iiskip:
                cc_inner.append(None)
                continue
            
            savename = "%sCESM1_EOF_Projection_concatEns_Residual_crosscorr_%s_%s.nc" % (rawpath,vnames[vv],vnames[uu])
            
            ds = xr.open_dataset(savename).load().__xarray_dataarray_variable__
            cc_inner.append(ds)
            
            
        iiskip += 1
        ds_cc_all.append(cc_inner)
    return ds_cc_all


def reshape_monyr_ds(ds,startyr):
    
    ds              = ds.transpose('yr','mon','lat','lon')
    nyr,_,nlat,nlon = ds.shape
    
    timedim         = xr.cftime_range(start=str(startyr),periods=nyr*12,freq="MS",calendar='noleap')
    
    ds_rs           = ds.data.reshape(nyr*12,nlat,nlon)
    coords          = dict(time=timedim,lat=ds.lat,lon=ds.lon)
    ds_rs           = xr.DataArray(ds_rs,coords=coords,dims=coords)
    return ds_rs

# Compute the pointwise crosscorrelation
def ds_crosscorr(ds1,ds2,dim='time'):
    return xr.cov(ds1,ds2,dim=dim) / (ds1.std(dim) * ds2.std(dim))

#%% Load the residual files
recalc = False # Set to True to alculate on stormtracl

if recalc:
    vnames  = ["Fprime","LHFLX","PRECTOT","Qek_SST","Qek_SSS"]
    nvars   = len(vnames)
    dsres   = []
    for vv in tqdm(range(nvars)):
        ncname = "%sCESM1_%s_EOF_Projection_concatEns.nc" % (rawpath,vnames[vv])
        ds     = xr.open_dataset(ncname).load()[vnames[vv]]
        dsres.append(ds)

    #dsres      = [dsres[vv][vnames[vv]] for vv in range(nvars)]
    
    dsres_time = [reshape_monyr_ds(ds,1920) for ds in dsres]
    
    #dsres = [print(ds.dims) for ds in dsres]
    #% Idea: Do a residual correlation matrix
    
    iiskip = 0
    ds_cc_all = []
    for vv in tqdm(range(nvars)):
        
        dsbase   = dsres_time[vv]
        
        cc_inner = []
        
        for uu in range(nvars):
            
            if uu < iiskip:
                cc_inner.append(None)
                continue
            
            dstarg = dsres_time[uu]
            
            cc = ds_crosscorr(dsbase,dstarg,dim='time')
            cc_inner.append(cc)
            
        iiskip += 1
        
        ds_cc_all.append(cc_inner)
        
    #%% Save the Output
    
    for vv in range(5):
        vn1 = vnames[vv]
        
        for uu in range(5):
            vn2 = vnames[uu]
            
            dsout = ds_cc_all[vv][uu]
            if dsout is None:
                continue
            
            
            savename = "%sCESM1_EOF_Projection_concatEns_Residual_crosscorr_%s_%s.nc" % (rawpath,vnames[vv],vnames[uu])
            dsout.to_netcdf(savename)
            print(savename)
else:
    
    ds_cc_all = reload_cc_matrix(rawpath)
    
    
    #%%

        
            

    
#%%

import cartopy.crs as ccrs

fsz_tick = 14
bboxplot = [-80,0,20,60]
fsz_axis = 24

proj     = ccrs.PlateCarree()



#fig,axs,_   = viz.init_orthomap(5,5,bboxplot,figsize=(28,26))
fig,axs = plt.subplots(5,5,figsize=(46,28),subplot_kw=dict(projection=proj),
                        constrained_layout=True)
ii          = 0

vnames_plot = ["$F'$","$q_L'$","$P'$","$Q_{ek} SST$","$Q_{ek} SSS$"]

for ax in axs.flatten():
    ax           = viz.add_coast_grid(ax,bboxplot,fill_color="lightgray",fontsize=fsz_tick,
                                grid_color="k")


cints = np.arange(-1,1.05,0.05)
for vv in range(5):
    
    for uu in range(5):
        ax = axs[vv,uu]
        
        if vv == 0:
            ax.set_title(vnames_plot[uu],fontsize=fsz_axis)
        
        if uu == 0:
            viz.add_ylabel(vnames_plot[vv],fontsize=fsz_axis,ax=ax)
        
        
        plotvar = ds_cc_all[vv][uu]
        
        if plotvar is None:
            #ax.clear()
            #ax.axis('off')
            continue
        
        pcm = ax.contourf(plotvar.lon,plotvar.lat,plotvar,
                                levels=cints,
                                cmap='cmo.balance',transform=proj)
        cl = ax.contour(plotvar.lon,plotvar.lat,plotvar,
                                levels=cints[::2],
                                linewidths=0.75,colors='k',transform=proj)
        ax.clabel(cl,fontsize=fsz_tick)
        # pcm     = ax.pcolormesh(plotvar.lon,plotvar.lat,plotvar,
        #                         vmin=-1,vmax=1,
        #                         cmap='cmo.balance',transform=proj)
            
cb = viz.hcbar(pcm,ax=axs.flatten(),fraction=0.035,pad=0.01)
cb.ax.tick_params(labelsize=fsz_tick)
cb.set_label("Cross-correlation",fontsize=fsz_axis)

figname = "%sCESM1_Residual_Correlations_EOF.png" % (figpath,)
plt.savefig(figname,dpi=200,bbox_inches='tight')



plt.show()

#%%


    
import itertools as it
        
[print(ii) for ii in it.permutations(vnames)]
        



def check_latlon_ds(ds_list,refid=0):
    # Checks "lat" and "lon" in list of reference datasets/dataarrays
    # that are of the same size. Compares it to the lat from the reference
    # ds (whose index/position in the list is indicated by refid)
    lats = [ds.lat.data for ds in ds_list]
    lons = [ds.lon.data for ds in ds_list]
    latref = lats[refid]
    lonref = lons[refid]
    nds    = len(ds_list)
    for dd in range(nds):
        if ~np.all(latref==lats[dd]):
            print("Warning: lat for ds %02i is not matching! Reassigning...")
            ds_list[dd]['lat'] = latref
        if ~np.all(lonref==lons[dd]):
            print("Warning: lon for ds %02i is not matching! Reassigning...")
            ds_list[dd]['lon'] = lonref
    return ds_list
            
            
            
        
    
    
    



