#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Correct EOF Forcing

Perform EOF filtering based on a variance threshold.
Compute the Required Variance needed to correct back to 100% (std(F'')) at each month.


Inputs:
------------------------



Outputs: 
------------------------

    varname             : dims                              - units                 - Full Name
    correction_factor   : (mon, lat, lon)                   [W/m2]
    eofs                : (mode, mon, lat, lon)             [W/m2 per std(pc)]
    
    

What does this script do?
------------------------
(1) Load in EOF Output and Fprime and take ensemble mean of variance explained, patterns, and std(F')
(2) Apply EOF Filtering, retaining only modes explaining up to N %
(3) Take difference of std(F') and std(EOF_filtered) to get pointwise variance correction factor
                                                                                                 
                                                                                                 

Note that correct ion is performed on the ENSEMBLE MEAN forcing and Fstd!!


Created on Tue Feb 13 20:21:00 2024

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

figpath   = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/02_Figures/20240217/"
datpath   = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/proc/CESM1/NATL_proc/"
outpath   = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/proc/model_input/forcing/"


# Indicate Filtering OPtions
eof_thres = 0.90

# Indicate Forcing Options
dampstr   = "nomasklag1"
rollstr   = "nroll0"

# Load EOF results
nceof     = "EOF_Monthly_NAO_EAP_Fprime_%s_%s_NAtl.nc" % (dampstr,rollstr)

# Load Fprime
ncfprime  = "CESM1_HTR_FULL_Fprime_timeseries_%s_%s_NAtl.nc" % (dampstr,rollstr)

# Load Ekman Forcing

#fp1  = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/proc/model_input/forcing/"
#ncek = "CESM1_HTR_FULL_Qek_SSS_NAO_nomasklag1_nroll0_NAtl_EnsAvg.nc"
#dsek = xr.open_dataset(fp1+ncek)

# Other things

bbox = [-80,0,0,65]
debug    = True
#%% Procedure

# (1) Load EOF Results, compute variance explained
dseof    = xr.open_dataset(datpath+nceof).load()
eofs     = dseof.eofs.mean('ens')   # (mode: 86, mon: 12, lat: 96, lon: 89)
varexp   = dseof.varexp.mean('ens') # (mode: 86, mon: 12)

# (2) Load Fprime, compute std(F') at each point
dsfp     = xr.open_dataset(datpath+ncfprime).load()
monvarfp = dsfp.Fprime.groupby('time.month').std('time').mean('ens') # (month: 12, lat: 96, lon: 89)


#%% 3. Perform EOF filtering (retain enough modes to explain [eof_thres]% of variance for each month)

eofs_std = dseof.eofs

# Eventualyl move this function to proc or some other package
# Ok I moved this to proc, need to try rerunning this at some point
def eof_filter(eofs,varexp,eof_thres,axis=0,return_all=False):
    # varexp    : [mode x mon]
    # eofs      : [mode x mon x lat x lon]
    # eof_thres : the percentange threshold (0.90=90%)
    # axis      : axis of the mode dimension
    
    varexp_cumu   = np.cumsum(varexp,axis=0) # Cumulative sum of variance
    above_thres   = varexp_cumu >= eof_thres        # Check exceedances
    nmodes_needed = np.argmax(above_thres,0)        # Get first exceedance
    
    eofs_filtered = eofs.copy()
    varexps_filt  = varexp.copy()
    for im in range(12):
        eofs_filtered[nmodes_needed[im]:,im,:,:] = 0 # Set modes above exceedence to zero
        varexps_filt[nmodes_needed[im]:,im] = 0
    if return_all:
        return eofs_filtered,varexp_cumu,nmodes_needed,varexps_filt
    # Here's a check"
    # print(np.sum(varexps_filt,0)) # Should be all below the variance threshold
    return eofs_filtered
        
# Perform Filtering
eofs_filtered,varexp_cumu,nmodes_needed,varexps_filt=proc.eof_filter(eofs.values,varexp.values,
                                                   eof_thres,axis=0,return_all=True)

# Check values
if debug:
    mons3   = proc.get_monstr()
    fig,ax  = viz.init_monplot(1,1)
    ax.bar(mons3,nmodes_needed,alpha=0.5,color='darkred')
    ax.set_xlim([-1,12])
    ax.set_title("Number of Modes Needed to Explain %.2f" % (eof_thres*100) + "% of Variance")
    ax.set_ylabel("Number of Modes")
    savename = "%sNAO_EAP_Fprime_Forcing_NumModes_thres%03i.png" % (figpath,eof_thres*100)
    plt.savefig(savename,dpi=150,bbox_inches='tight')
    
    fig,ax  = viz.init_monplot(1,1)
    ax.plot(np.sum(varexp,0),label="Raw")
    ax.plot(np.sum(varexps_filt,0),label="Post-Filtering")
    ax.set_ylabel("Total Variance Explained")
    ax.legend()
    
#%% 4. Compute the needed pointwise corrections 

# Compute Stdev of EOFs
eofs_std = np.sqrt(np.sum(eofs_filtered**2,0)) # [Mon x Lat x Lon]

if debug:
    
    vplot = "Fprime"
    if vplot == "EOFs":
        
        invar = eofs_std
        cmap  = 'inferno'
        vmax = 40
    elif vplot == "Fprime":
        invar = monvarfp
        cmap  = 'cmo.thermal'
        vmax = 80
    
    lon = dseof.lon.values
    lat = dseof.lat.values
    
    fig,axs = viz.geosubplots(4,3,constrained_layout=True,figsize=(12,15))
    
    for im in range(12):
        
        ax = axs.flatten()[im-1]
        ax = viz.add_coast_grid(ax,bbox=bbox,fill_color='lightgray')
        ax.set_title(mons3[im])
        
        plotvar = invar[im,:,:]
        pcm = ax.pcolormesh(lon,lat,plotvar,vmin=0,vmax=vmax,cmap=cmap)
        if vmax is None:
            fig.colorbar(pcm,ax=ax,orientation='horizontal',fraction=0.015,pad=0.01)
            pcm = ax.pcolormesh(lon,lat,plotvar,cmap='inferno')
        else:
            pcm = ax.pcolormesh(lon,lat,plotvar,vmin=0,vmax=vmax,cmap=cmap)
            
    if vmax is not None:
        cb = fig.colorbar(pcm,ax=axs.flatten(),orientation='horizontal',fraction=0.015,pad=0.01)
        cb.set_label("F' EOF Forcing ($W/m^2$)")
        
        savename = "%sNAO_EAP_%s_Forcing_Stdev.png" % (figpath,vplot)
        plt.savefig(savename,dpi=150,bbox_inches='tight')
        

# Compute Ratio with Fstd
correction = 1/(eofs_std/monvarfp)

if debug:
    
    vmax = 10
    invar = correction


    
    lon = dseof.lon.values
    lat = dseof.lat.values
    
    fig,axs = viz.geosubplots(4,3,constrained_layout=True,figsize=(12,15))
    
    for im in range(12):
        
        ax = axs.flatten()[im-1]
        ax = viz.add_coast_grid(ax,bbox=bbox,fill_color='lightgray')
        ax.set_title(mons3[im])
        
        plotvar = invar[im,:,:]
        pcm = ax.pcolormesh(lon,lat,plotvar,vmin=0,vmax=vmax,cmap=cmap)
        if vmax is None:
            fig.colorbar(pcm,ax=ax,orientation='horizontal',fraction=0.015,pad=0.01)
            pcm = ax.pcolormesh(lon,lat,plotvar,cmap='inferno')
        else:
            pcm = ax.pcolormesh(lon,lat,plotvar,vmin=0,vmax=vmax,cmap=cmap)
            
    if vmax is not None:
        cb = fig.colorbar(pcm,ax=axs.flatten(),orientation='horizontal',fraction=0.015,pad=0.01)
        cb.set_label("Correction Factor")
        
        savename = "%sNAO_EAP_EOF_Forcing_Correction.png" % (figpath)
        plt.savefig(savename,dpi=150,bbox_inches='tight')
#%% Compute correction via the differences

correction_diff = monvarfp - eofs_std

if debug:
    
    vmax = None
    invar = correction_diff


    vmax = 40
    lon = dseof.lon.values
    lat = dseof.lat.values
    
    fig,axs = viz.geosubplots(4,3,constrained_layout=True,figsize=(12,15))
    
    for im in range(12):
        
        ax = axs.flatten()[im-1]
        ax = viz.add_coast_grid(ax,bbox=bbox,fill_color='lightgray')
        ax.set_title(mons3[im])
        
        plotvar = invar[im,:,:]
        pcm = ax.pcolormesh(lon,lat,plotvar,vmin=0,vmax=vmax,cmap=cmap)
        if vmax is None:
            fig.colorbar(pcm,ax=ax,orientation='horizontal',fraction=0.015,pad=0.01)
            pcm = ax.pcolormesh(lon,lat,plotvar,cmap='inferno')
        else:
            pcm = ax.pcolormesh(lon,lat,plotvar,vmin=0,vmax=vmax,cmap=cmap)
            
    if vmax is not None:
        cb = fig.colorbar(pcm,ax=axs.flatten(),orientation='horizontal',fraction=0.015,pad=0.01)
        cb.set_label("Correction Factor")
        
        savename = "%sNAO_EAP_EOF_Forcing_Correction_Diff.png" % (figpath)
        plt.savefig(savename,dpi=150,bbox_inches='tight')

#%% Save Output

corcoords     = dict(mon=np.arange(1,13,1),lat=dseof.lat,lon=dseof.lon)
eofcoords     = dict(mode=dseof.mode,mon=np.arange(1,13,1),lat=dseof.lat,lon=dseof.lon)

da_correction = xr.DataArray(correction_diff,coords=corcoords,dims=corcoords,name="correction_factor")
da_eofs_filt  = xr.DataArray(eofs_filtered,coords=eofcoords,dims=eofcoords  ,name="eofs")

ds_out        = xr.merge([da_correction,da_eofs_filt])
edict         = proc.make_encoding_dict(ds_out)

savename      = "%sCESM1_HTR_FULL_Fprime_EOF_corrected_%s_%s_perc%03i_NAtl_EnsAvg.nc"  % (outpath,dampstr,rollstr,eof_thres*100)
#"EOF_Monthly_NAO_EAP_Fprime_%s_%s_NAtl.nc" % (dampstr,rollstr)

ds_out.to_netcdf(savename,encoding=edict)
