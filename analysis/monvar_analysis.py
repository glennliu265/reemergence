#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Monthly Variability Analysis

Analyze the monthly variability of terms in the Stochastic Model to see if there is any insight...
Copied upper section of viz_CESM1_HTR_meanstates


Created on Tue Sep 10 09:13:19 2024

@author: gliu
"""



import xarray as xr
import numpy as np
import matplotlib as mpl

import matplotlib.pyplot as plt
import sys
import glob
import os

import tqdm
import time

import cartopy.crs as ccrs
import matplotlib.patheffects as pe
# ----------------------------------
# %% Import custom modules and paths
# ----------------------------------

# Import re-eergemce parameters

# Indicate the Machine!
machine = "Astraeus"

# First Load the Parameter File
cwd = os.getcwd()
sys.path.append(cwd+ "/..")
import reemergence_params as rparams

# Paths and Load Modules
pathdict = rparams.machine_paths[machine]

sys.path.append(pathdict['amvpath'])
sys.path.append(pathdict['scmpath'])

# Set needed paths
figpath     = pathdict['figpath']
input_path  = pathdict['input_path']
output_path = pathdict['output_path']
procpath    = pathdict['procpath']
rawpath     = pathdict['raw_path']

#%% Import Custom Modules

# Import AMV Calculation
from amv import proc,viz
import amv.loaders as dl

# Import stochastic model scripts
import scm

#%% Plotting Information

bboxplot                    = [-80,0,20,65]
mpl.rcParams['font.family'] = 'Avenir'
mons3                       = proc.get_monstr(nletters=3)
fsz_tick                    = 18
fsz_axis                    = 22
fsz_title                   = 28

proj                        = ccrs.PlateCarree()

#%% Load necessary data

ds_uvel,ds_vvel = dl.load_current()
ds_bsf          = dl.load_bsf(ensavg=False)
ds_ssh          = dl.load_bsf(ensavg=False,ssh=True)

# Convert Currents to m/sec instead of cmsec
ds_uvel = ds_uvel/100
ds_vvel = ds_vvel/100

# Load data processed by [calc_monmean_CESM1.py]
ds_sss          = dl.load_monmean('SSS')
ds_sst          = dl.load_monmean('SST')


tlon  = ds_uvel.TLONG.mean('ens').values
tlat  = ds_uvel.TLAT.mean('ens').values


# Load Mixed-Layer Depth
mldpath = input_path + "mld/"
mldnc   = "CESM1_HTR_FULL_HMXL_NAtl_EnsAvg.nc"
ds_mld  = xr.open_dataset(mldpath+mldnc).h.load()

#ds_h          = dl.load_monmean('HMXL')

#%% Compute the velocity

ds_umod = (ds_uvel.UVEL ** 2 + ds_vvel.VVEL ** 2)**(0.5)


#%% Load Masks

# Load Land Ice Mask
icemask     = xr.open_dataset(input_path + "masks/CESM1LE_HTR_limask_pacificmask_enssum_lon-90to20_lat0to90.nc")


mask        = icemask.MASK.squeeze()
mask_plot   = xr.where(np.isnan(mask),0,mask)#mask.copy()


mask_reg_sub    = proc.sel_region_xr(mask,bboxplot)
mask_reg_ori    = xr.ones_like(mask) * 0
mask_reg        = mask_reg_ori + mask_reg_sub


mask_apply  = icemask.MASK.squeeze().values
#mask_plot[np.isnan(mask)] = 0

# Load Gulf Stream
ds_gs   = dl.load_gs()
ds_gs   = ds_gs.sel(lon=slice(-90,-50))
ds_gs2  = dl.load_gs(load_u2=True)

#%% Load REI From a specific run

rei_nc   = "REI_Pointwise.nc"
rei_path = output_path + "SSS_CESM/Metrics/"
ds_rei   = xr.open_dataset(rei_path + rei_nc).load().rei
reiplot  = ds_rei.isel(mon=[1,2],yr=0).mean('mon').mean('ens')

reiplot_sss= reiplot.copy()


rei_nc   = "REI_Pointwise.nc"
rei_path = output_path + "SST_CESM/Metrics/"
ds_rei_sst   = xr.open_dataset(rei_path + rei_nc).load().rei
reiplot_sst  = ds_rei_sst.isel(mon=[1,2],yr=0).mean('mon').mean('ens')

# ===============================================
#%% Part (1), Monthly Variance of SST and SSS
# ===============================================

# Load the CESM1 Datasets
vnames = ["SST","SSS"]
ds_all = []
for vv in range(2):
    
    vname       = vnames[vv]
    exppath     = rawpath #"%s%s_CESM/Output/" % (output_path,vname)
    monvar_nc   = exppath + "monthly_variance/CESM1LE_%s_NAtl_19200101_20050101_bilinear_stdev.nc" % vname #exppath + "%s_runid00.nc" % vname
    ds          = xr.open_dataset(monvar_nc).load()[vname]#.mean('run')
    ds_all.append(ds)
    

# def nanargmaxds(ds,dimname):
#     mask   = xr.where(~np.isnan(ds),1,np.nan)
#     tempds = xr.where(np.isnan(ds),0,ds)
#     argmax = tempds.argmax(dimname)
#     return argmax * mask

ensavg = [ds.mean('ens') for ds in ds_all]
monmax = [proc.nanargmaxds(ds,'mon')+1 for ds in ensavg]
monmin = [proc.nanargminds(ds,'mon')+1 for ds in ensavg]

#%% Plot Month of Maximum /Minimum SST/SSS Variance
vv          = 1
plotmax     = True

vname       = vnames[vv]

# Initialize Plot and Map
fig,ax,_    = viz.init_orthomap(1,1,bboxplot,figsize=(28,16))
ax          = viz.add_coast_grid(ax,bbox=bboxplot,fill_color="lightgray",fontsize=24)

if plotmax:
    plotvar     = monmax[vv] * mask_reg
    maxstr      = "Max"
else:
    plotvar     = monmin[vv] * mask_reg
    maxstr      = "Min"
    
    
pcm         = ax.pcolormesh(plotvar.lon,plotvar.lat,plotvar,cmap='twilight',transform=proj,vmin=0,vmax=12)
cb          = viz.hcbar(pcm,ax=ax)
cb.ax.tick_params(labelsize=24)

ax.set_title("Month of %s Interannual %s Variance" % (maxstr,vnames[vv]),fontsize=42)

# Plot Labels for points
for (i, j), z in np.ndenumerate(plotvar):
    try:
        ax.text(plotvar.lon.data[j], plotvar.lat.data[i], '%i' % (z),
                ha='center', va='center',transform=proj,fontsize=14,color='k',zorder=4,)#path_effects=[pe.withStroke(linewidth=1.5, foreground="w")])
    except:
        print(None)
        #print("Nan Point")
        
        
# Plot Other Features (Ice Edge, Gulf Stream)
# Plot Ice Mask
ax.contour(icemask.lon,icemask.lat,mask_plot,colors="cyan",linewidths=4,
           transform=proj,levels=[0,1],zorder=-1)
# Plot Gulf Stream Position
ax.plot(ds_gs2.lon.mean('mon'),ds_gs2.lat.mean('mon'),transform=proj,lw=4,c='cornflowerblue',ls='dashdot')

# plot contours of max MLD
mldplot = ds_mld.max('mon')
ax.contour(mldplot.lon,mldplot.lat,mldplot,colors="w",transform=proj,
           levels=np.arange(0,1300,50),linewidths=0.75,zorder=3)

savename = "%sMonth%s_%s_Variance.png" % (figpath,maxstr,vname)
plt.savefig(savename,dpi=150,bbox_inches='tight')

# =========================================
#%% Part (2): Check Mon of Maximum Forcing
# =========================================

mvpath      = "monthly_variance/" # New Monthly variance Folder
saveprec    = True
recalc      = True

def anomalize(ds):
    dsanom = ds - ds.mean('ens')
    dsanom = proc.xrdeseason(dsanom)
    return dsanom

# Load and Process Precip ----------------------------------------------
savename = "%s%sCESM1LE_PRECTOT_NAtl_19200101_20050101_stdev.nc" % (rawpath,mvpath)
if len(glob.glob(savename)) < 1 or recalc:
    print("Warning, file not found. Recalculating")
    

    ncprec          = rawpath + "PRECTOT_HTR_FULL.nc"
    dsprec         = xr.open_dataset(ncprec).load().PRECTOT
    
    dsprec         = proc.fix_febstart(dsprec)
    dsprec_anom    = anomalize(dsprec)
    prec_monvar    = dsprec_anom.groupby('time.month').var('time')#.mean('ens')
    
    # Save Output
    if saveprec:
        edict    = proc.make_encoding_dict(prec_monvar)
        prec_monvar.to_netcdf(savename,encoding=edict)
    
    
else:
    print("File for PRECTOT was found")
    prec_monvar = xr.open_dataset(savename).PRECTOT.load()
prec_monvarmax = proc.nanargmaxds(prec_monvar,'month')
prec_monvarmin = proc.nanargminds(prec_monvar,'month')

#%% Load and Process Stochastic Evaporation -------------------------------
monvar_name = "LHFLX"
recalc      = True
ncname      = "CESM1_HTR_FULL_Eprime_timeseries_LHFLXnomasklag1_nroll0_NAtl.nc"

savename    = rawpath + mvpath + proc.addstrtoext(ncname,"_stdev",adjust=-1)#"%s%s%s_stdev.nc" % (rawpath,mvpath,ncname)
if len(glob.glob(savename)) < 1 or recalc:
    print("Warning, file  for %s not found. Recalculating" % monvar_name)
    
    ncprec          = rawpath + ncname
    dsprec         = xr.open_dataset(ncprec).load()[monvar_name]
    
    dsprec         = proc.format_ds_dims(dsprec)
    dsprec         = proc.fix_febstart(dsprec)
    dsprec_anom    = anomalize(dsprec)
    prec_monvar    = dsprec_anom.groupby('time.month').var('time')#.mean('ens')
    
    # Save Output
    if saveprec:
        edict    = proc.make_encoding_dict(prec_monvar)
        prec_monvar.to_netcdf(savename,encoding=edict)
        
    qL_monvar = prec_monvar.copy()
    
else:
    print("File for %s %s was found" % (monvar_name,vname))
    qL_monvar = xr.open_dataset(savename)[monvar_name].load()
    
#%% Load the Process Fprim

monvar_name = "Fprime"
recalc      = True
ncname      = "CESM1_HTR_FULL_Fprime_timeseries_nomasklag1_nroll0_NAtl.nc"


savename    = rawpath + mvpath + proc.addstrtoext(ncname,"_stdev",adjust=-1)#"%s%s%s_stdev.nc" % (rawpath,mvpath,ncname)
if len(glob.glob(savename)) < 1 or recalc:
    print("Warning, file  for %s not found. Recalculating" % monvar_name)
    
    ncprec          = rawpath + ncname
    dsprec         = xr.open_dataset(ncprec).load()[monvar_name]
    
    dsprec         = proc.format_ds_dims(dsprec)
    dsprec         = proc.fix_febstart(dsprec)
    dsprec_anom    = anomalize(dsprec)
    prec_monvar    = dsprec_anom.groupby('time.month').var('time')#.mean('ens')
    
    # Save Output
    if saveprec:
        edict    = proc.make_encoding_dict(prec_monvar)
        prec_monvar.to_netcdf(savename,encoding=edict)
        
    Fprime_monvar = prec_monvar.copy()
    
else:
    print("File for %s was found" % (monvar_name))
    qL_monvar = xr.open_dataset(savename)[monvar_name].load()

#%% Load and Process Qek Forcings
Qek_monvars = []
monvar_name = "Qek"
recalc=True
for vv in range(2):
    
    vname       = vnames[vv]
    savename    = "%s%sCESM1LE_Qek_%s_NAtl_19200101_20050101_bilinear_stdev.nc" % (rawpath,mvpath,vname)
    if len(glob.glob(savename)) < 1 or recalc:
        print("Warning, file not found. Recalculating")
        
        print(vname)
        ncprec          = rawpath + "CESM1LE_Qek_%s_NAtl_19200101_20050101_bilinear.nc" % vname
        dsprec         = xr.open_dataset(ncprec).load().Qek
        
        dsprec         = proc.format_ds_dims(dsprec)
        dsprec         = proc.fix_febstart(dsprec)
        dsprec_anom    = anomalize(dsprec)
        prec_monvar    = dsprec_anom.groupby('time.month').var('time')#.mean('ens')
        
        # Save Output
        if saveprec:
            edict    = proc.make_encoding_dict(prec_monvar)
            prec_monvar.to_netcdf(savename,encoding=edict)
        
    else:
        print("File for %s %s was found" % (monvar_name,vname))
        prec_monvar = xr.open_dataset(savename)[monvar_name].load()
    Qek_monvars.append(prec_monvar)

#%% Make Plot of Precipitation

plotmax     = False

# vname       = "PRECIP"
# inmonvarmax = prec_monvarmax * mask_reg
# inmonvarmin = prec_monvarmin * mask_reg

# # vname       = "Qek_SSS"
# # monvarin    = Qek_monvars[1]
# vname       = "Qek_SST"
# monvarin    = Qek_monvars[0]
# inmonvarmax = proc.nanargmaxds(monvarin,'month') * mask_reg
# inmonvarmin = proc.nanargminds(monvarin,'month') * mask_reg

vname       = "qL"
monvarin    = qL_monvar
inmonvarmax = proc.sel_region_xr(proc.nanargmaxds(monvarin,'month'),bboxplot) #* mask_reg
inmonvarmin = proc.sel_region_xr(proc.nanargminds(monvarin,'month'),bboxplot) #* mask_reg

#plotvar     = [monvarmin[1],monvarmax[1]]

# Initialize Plot and Map
fig,ax,_    = viz.init_orthomap(1,1,bboxplot,figsize=(28,16))
ax          = viz.add_coast_grid(ax,bbox=bboxplot,fill_color="lightgray",fontsize=24)

if plotmax:
    plotvar     = inmonvarmax
    maxstr      = "Max"
    plotcont    = monmax[1]
    ccol = "k"
else:
    plotvar     = inmonvarmin
    maxstr      = "Min"
    plotcont    = monmin[1]
    ccol = "w"
    
    
pcm         = ax.pcolormesh(plotvar.lon,plotvar.lat,plotvar,cmap='twilight',transform=proj,vmin=0,vmax=12)
cb          = viz.hcbar(pcm,ax=ax)
cb.ax.tick_params(labelsize=24)

ax.set_title("Month of %s Interannual %s Variance" % (maxstr,vname),fontsize=42)

# Plot Labels for points
for (i, j), z in np.ndenumerate(plotvar):
    try:
        ax.text(plotvar.lon.data[j], plotvar.lat.data[i], '%i' % (z),
                ha='center', va='center',transform=proj,fontsize=14,color='k',zorder=4,)#path_effects=[pe.withStroke(linewidth=1.5, foreground="w")])
    except:
        print(None)
        #print("Nan Point")
        
        
# Plot Other Features (Ice Edge, Gulf Stream)
# Plot Ice Mask
ax.contour(icemask.lon,icemask.lat,mask_plot,colors="cyan",linewidths=4,
           transform=proj,levels=[0,1],zorder=-1)
# Plot Gulf Stream Position
ax.plot(ds_gs2.lon.mean('mon'),ds_gs2.lat.mean('mon'),transform=proj,lw=4,c='cornflowerblue',ls='dashdot')

# plot contours of when Salinity is max or min
mldplot = plotcont#ds_mld.max('mon')
cl = ax.contour(mldplot.lon,mldplot.lat,mldplot,colors=ccol,transform=proj,
           levels=np.arange(1,13,1),linewidths=0.75,zorder=3)
ax.clabel(cl,fontsize=24)

savename = "%sMonth%s_%s_Variance.png" % (figpath,maxstr,vname)
plt.savefig(savename,dpi=150,bbox_inches='tight')

# =============================================================================
#%%
# =============================================================================


