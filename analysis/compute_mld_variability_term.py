#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Calculate and investigate the contribution of the mixed-layer depth variability term
A rough calculation...

General Procedure
- Load SST and SSS
- Compute Sbar and Tbar
- Compute dSbar dTbar...
- Load HMXL
- Compute Hbar and Hprime
- Check the contributions of the term...


Note... Ensemble average was not removed, so this is still there...

Created on Tue Sep 24 13:31:55 2024

@author: gliu
"""

import xarray as xr
import numpy as np
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import cartopy.crs as ccrs
import scipy as sp

import matplotlib.pyplot as plt
import sys
import glob
import os

import tqdm
import time

from cmcrameri import cm

# ----------------------------------
# %% Import custom modules and paths
# ----------------------------------

# Import re-eergemce parameters

# Indicate the Machine!
machine = "Astraeus"

# First Load the Parameter File
cwd             = os.getcwd()
sys.path.append(cwd + "/..")

# Paths and Load Modules
import reemergence_params as rparams
pathdict        = rparams.machine_paths[machine]

sys.path.append(pathdict['amvpath'])
sys.path.append(pathdict['scmpath'])

# Set needed paths
figpath         = pathdict['figpath']
input_path      = pathdict['input_path']
output_path     = pathdict['output_path']
procpath        = pathdict['procpath']
rawpath         = pathdict['raw_path']

# %% Import Custom Modules

from amv import proc, viz
import scm
import amv.xrfunc as xrf
import amv.loaders as dl

# Import stochastic model scripts
proc.makedir(figpath)

#%% Define some functions

# From viz_inputs_basinwide
def init_monplot(bboxplot=[-80,0,20,60],fsz_axis=24):
    mons3         = proc.get_monstr()
    plotmon       = np.roll(np.arange(12),1)
    fig,axs,mdict = viz.init_orthomap(4,3,bboxplot=bboxplot,figsize=(18,18))
    for a,ax in enumerate(axs.flatten()):
        im = plotmon[a]
        ax = viz.add_coast_grid(ax,bbox=bboxplot,fill_color="lightgray")
        ax.set_title(mons3[im],fontsize=fsz_axis)
    return fig,axs

#%% Load Plotting Parameters
bboxplot    = [-80,0,20,60]

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

#%% Load more infomation on the points

# Get Point Info
pointset        = "PaperDraft02"
ptdict          = rparams.point_sets[pointset]
ptcoords        = ptdict['bboxes']
ptnames         = ptdict['regions']
ptnames_long    = ptdict['regions_long']
ptcols          = ptdict['rcols']
ptsty           = ptdict['rsty']

#%% Indicate Dataset to Filter

vnames = ["SST","SSS","HMXL"]
ds_all = []
for vv in range(3):
    st = time.time()
    
    ncname  = rawpath + "CESM1LE_%s_NAtl_19200101_20050101_bilinear.nc" % vnames[vv]
    ds      = xr.open_dataset(ncname)[vnames[vv]].load()
    ds_all.append(ds)
    print("Loaded %s in %.2fs" % (vnames[vv],time.time()-st))

#%% Compute the monthly mean

print("Computing mean values and tendencies")
ds_mean = [ds.groupby('time.month').mean('time') for ds in ds_all]
#[print(ds.dims) for ds in ds_mean]

Tbar,Sbar,Hbar = ds_mean

# Compute dTbar dt [ for dT/dt(Jan) = T(Feb) - T(Jan) ], units are [degC/mon] and [psu/mon]
dTbar_dt       = Tbar.differentiate('month',datetime_unit='m') # (42, 12, 96, 89)
dSbar_dt       = Sbar.differentiate('month',datetime_unit='m')


#% Read out to Arrays
dTbar_dt = dTbar_dt.data[:,None,:,:,:] # [Ens x Year x Mon x Lat x Lon]
dSbar_dt = dSbar_dt.data[:,None,:,:,:] # [Ens x Year x Mon x Lat x Lon]

#%% Compute hrpime / hbar

print("Computing h' and hbar")
ds_h                 =  ds_all[2]
nens,ntime,nlat,nlon = ds_h.shape
h                    = ds_h.data.reshape(nens,int(ntime/12),12,nlat,nlon)
hbar                 = Hbar.data[:,None,:,:]

hprime               = h - hbar





#%% Compute the full term, place back in dataarray

print("Computing full term h'/hbar * dTbar/dt...")
hterm_SSS    = (hprime/hbar * dSbar_dt).reshape(nens,ntime,nlat,nlon)
hterm_SST    = (hprime/hbar * dTbar_dt).reshape(nens,ntime,nlat,nlon)

coords       = dict(ens=np.arange(1,nens+1,1),time=ds_h.time,lat=ds_h.lat,lon=ds_h.lon)
da_hterm_SSS = xr.DataArray(hterm_SSS,coords=coords,dims=coords,name="SSS")
da_hterm_SST = xr.DataArray(hterm_SST,coords=coords,dims=coords,name="SST")

hratio    = hprime.std(1) / hbar.squeeze()#.groupby('time.month').std('time')/hbar
coords2  = dict(ens=np.arange(1,nens+1,1),mon=np.arange(1,13,1),lat=ds_h.lat,lon=ds_h.lon)
da_hratio = xr.DataArray(hratio,coords=coords2,dims=coords2,name="h")


ncname      = "%sCESM1LE_HMXL_hratio_NAtl_19200101_20050101_bilinear.nc" % (rawpath)
edict       = proc.make_encoding_dict(da_hratio)
da_hratio.to_netcdf(ncname,encoding=edict)

#%% Save the Output

vnames   = ["SST","SSS"]
savevars = [da_hterm_SST,da_hterm_SSS,]

for vv in range(2):
    savename = "%sCESM1LE_MLDvarTerm_%s_NAtl_19200101_20050101_bilinear.nc" % (rawpath,vnames[vv])
    dain     = savevars[vv]
    edict    = proc.make_encoding_dict(dain)
    dain.to_netcdf(savename,encoding=edict)
    
    
#%% Also compute the total ratio


hprime_std_all  = np.nanstd(hprime.reshape(nens,ntime,nlat,nlon),1)
hbar_std_all    = np.nanstd(h.reshape(nens,ntime,nlat,nlon),1)
hratio_std_all  = hprime_std_all/hbar_std_all


coords_all    = dict(ens=np.arange(1,nens+1,1),lat=ds_h.lat,lon=ds_h.lon)
da_hprime_all = xr.DataArray(hprime_std_all,coords=coords_all,dims=coords_all,name="hprime")
da_hbar_all   = xr.DataArray(hbar_std_all,coords=coords_all,dims=coords_all,name="hbar")
da_hratio_all = xr.DataArray(hratio_std_all,coords=coords_all,dims=coords_all,name="hratio")

da_hout = xr.merge([da_hprime_all,da_hbar_all,da_hratio_all])

ncname        = "%sCESM1LE_HMXL_hratio_totalstd_NAtl_19200101_20050101_bilinear.nc" % (rawpath)
edict         = proc.make_encoding_dict(da_hout)
da_hout.to_netcdf(ncname,encoding=edict)

# ========================
#%% Load the Output
# ========================

vnames   = ["SST","SSS"]
savevars = []
for vv in range(2):
    savename = "%sCESM1LE_MLDvarTerm_%s_NAtl_19200101_20050101_bilinear.nc" % (rawpath,vnames[vv])
    ds = xr.open_dataset(savename)[vnames[vv]].load()
    savevars.append(ds)

da_hterm_SST,da_hterm_SSS = savevars

ncname      = "%sCESM1LE_HMXL_hratio_NAtl_19200101_20050101_bilinear.nc" % (rawpath)
da_hratio   = xr.open_dataset(ncname).h.load()

ncname      = "%sCESM1LE_HMXL_hratio_totalstd_NAtl_19200101_20050101_bilinear.nc" % (rawpath)
da_hratio_total = xr.open_dataset(ncname).hratio.load()


#%% Look at monthly variance contributions

hSSS_monvar = da_hterm_SSS.groupby('time.month').std("time")
hSST_monvar = da_hterm_SST.groupby('time.month').std("time")

#%% Plot the monthly patterns of contribution
plotmons = np.roll(np.arange(1,13,1),1)
proj     = ccrs.PlateCarree()
fsz_tick = 14
fsz_axis = 20

vv       = 1
 
vmax     = 0.015
cints    = np.arange(0.01,0.2,0.01)

fig,axs = init_monplot()

for ii in range(12):
    
    
    kmonth = plotmons[ii] - 1
    ax     = axs.flatten()[ii]
    
    if vv == 0:
        mvin = hSSS_monvar
        vmax     = 0.015
        cints    = np.arange(0.01,0.2,0.01)
        cmap     = 'cmo.haline'
        vunit    = 'psu/mon'
        vname    = "SSS"
    else:
        mvin     = hSST_monvar
        vmax     = 0.5
        cints    = np.arange(0.05,1.05,0.05)
        cmap     = 'cmo.thermal'
        vunit    = '$\degree$C/mon'
        vname    = "SST"
        
    
    plotvar = mvin.isel(month=kmonth).mean('ens') * mask
    
    
    pcm     = ax.pcolormesh(plotvar.lon,plotvar.lat,plotvar,
                            transform=proj,vmax=vmax,vmin=0.,cmap=cmap)
    
    cl      = ax.contour(plotvar.lon,plotvar.lat,plotvar,
                             levels=cints,transform=proj,
                             colors="k",linewidths=0.75)
    ax.clabel(cl,fontsize=fsz_tick)
    
    # Plot Other Features
    ax.plot(ds_gs2.lon.isel(mon=kmonth),ds_gs2.lat.isel(mon=kmonth),transform=proj,lw=1.75,c='cornflowerblue',ls='dashdot')

    # Plot Ice Edge
    ax.contour(icemask.lon,icemask.lat,mask_plot,colors="cyan",linewidths=2.5,
               transform=proj,levels=[0,1],zorder=-1)
    
cb = fig.colorbar(pcm,ax=axs.flatten(),fraction=0.045,pad=0.01)

termname = r"$\frac{h'}{\overline{h}} \, \frac{ \partial \overline{%s}}{\partial t}$" % vname[0]
cb.set_label("Monthly Stdev. of MLD Term (%s, Units: %s)" % (termname,vunit),fontsize=fsz_axis)
cb.ax.tick_params(labelsize=fsz_tick)

savename = "%sMLD_Term_Stdev_Monthly_%s.png" % (figpath,vname)
plt.savefig(savename,dpi=150,bbox_inches='tight')
#plt.suptitle(termname,fontsize=fsz_axis)


#%% Save as Above, but make seasonally averaged plots

#%% Remake Monthly Variance Plot, but for the individual locations

mons3     = proc.get_monstr()

# Compute Monthly variance
inmonvars = [ds.groupby('time.month').var('time') for ds in savevars]
inmonstds = [ds.groupby('time.month').std('time') for ds in savevars]

# Compute the monthly averages
inmonvars_savg = [proc.calc_savg(ds.rename(dict(month='mon')),ds=True) for ds in inmonvars]
inmonstds_savg = [proc.calc_savg(ds.rename(dict(month='mon')),ds=True) for ds in inmonstds]


# Compute monthly average hratio
#da_monmean  = da_hratio.groupby('time.month').mean('time')
hratio_savg = proc.calc_savg(da_hratio,ds=True)

fig,axs = viz.init_monplot(2,3,figsize=(12,6.5))

for vv in range(2):
    for ip in range(3):
        ax  = axs[vv,ip]
        lonf,latf = ptcoords[ip]
        locfn,loctitle=proc.make_locstring(lonf,latf)
        plotvar = proc.selpt_ds(inmonvars[vv],lonf,latf).mean('ens')
        ax.plot(mons3,plotvar)
    ax.set_title(loctitle)
        
    
#%% Plot The Seasonally Averaged Plots, borrowing the format from the hFFF plots
# (Draft 04)

fsz_tick  = 16
fsz_title = 26

fig,axs,mdict = viz.init_orthomap(3,4,bboxplot=bboxplot,figsize=(28,15))

for ax in axs.flatten():
    ax = viz.add_coast_grid(ax,bbox=bboxplot,fill_color="lightgray")
    
ii = 0
# Plot hprime
vv = 0
for sid in range(4):
    ax = axs[vv,sid]
    plotvar = hratio_savg.isel(season=sid).mean('ens') #* mask
    
    ax.set_title(plotvar.season.item(),fontsize=fsz_axis)
    if sid == 0:
        viz.add_ylabel("Mixed-Layer Depth Ratio",fontsize=fsz_axis,ax=ax)
    
    cmap    = 'cmo.deep_r'
    cints   = np.arange(0,1.1,0.1)
    lc      = 'lightgray'
    
    pcm     = ax.contourf(plotvar.lon,plotvar.lat,plotvar,
                            transform=proj,levels=cints,cmap=cmap)
    
    cl      = ax.contour(plotvar.lon,plotvar.lat,plotvar,
                             levels=cints,transform=proj,
                             colors=lc,linewidths=0.75)
    
    
    # Plot Other Features
    ax.plot(ds_gs2.lon.isel(mon=kmonth),ds_gs2.lat.isel(mon=kmonth),transform=proj,lw=1.75,c='cornflowerblue',ls='dashdot')

    # Plot Ice Edge
    ax.contour(icemask.lon,icemask.lat,mask_plot,colors="cyan",linewidths=2.5,
               transform=proj,levels=[0,1],zorder=-1)
    
    
    viz.label_sp(ii,alpha=0.75,ax=ax,fontsize=fsz_title,y=1.08,x=-.02)
    ii += 1
    
    
    ax.clabel(cl,fontsize=fsz_tick)
cb = fig.colorbar(pcm,ax=axs[0,:].flatten(),fraction=0.0075,pad=0.01)
cb.ax.tick_params(labelsize=fsz_tick)
cb.set_label(r"$\frac{\sigma (h')}{\overline{h}}$",fontsize=fsz_axis)

# Plot the MLD Terms
for vv in range(2):
    
        
    if vv == 0:
        vmax     = 0.5
        cints    = np.arange(0.05,1.05,0.05)
        cmap     = 'cmo.thermal'
        vunit    = '$\degree$C/mon'
        vname    = "SST"
        lc       = "k"
    elif vv == 1:
        vmax     = 0.015
        cints    = np.arange(0.005,0.25,0.005)
        cmap     = 'cmo.rain'
        vunit    = 'psu/mon'
        vname    = "SSS"
        lc       = "cyan"
    
    for sid in range(4):
        ax      = axs[vv+1,sid]
        
        
        plotvar = inmonstds_savg[vv].isel(season=sid).mean('ens') * mask
        

        if sid == 0:
            viz.add_ylabel(vnames[vv],fontsize=fsz_axis,ax=ax)
            
        
        pcm     = ax.pcolormesh(plotvar.lon,plotvar.lat,plotvar,
                                transform=proj,vmax=vmax,vmin=0.,cmap=cmap)
        
        cl      = ax.contour(plotvar.lon,plotvar.lat,plotvar,
                                 levels=cints,transform=proj,
                                 colors=lc,linewidths=0.75)
        ax.clabel(cl,fontsize=fsz_tick)
            
        # Plot Other Features
        ax.plot(ds_gs2.lon.isel(mon=kmonth),ds_gs2.lat.isel(mon=kmonth),transform=proj,lw=1.75,c='cornflowerblue',ls='dashdot')

        # Plot Ice Edge
        ax.contour(icemask.lon,icemask.lat,mask_plot,colors="cyan",linewidths=2.5,
                   transform=proj,levels=[0,1],zorder=-1)
        
        viz.label_sp(ii,alpha=0.75,ax=ax,fontsize=fsz_title,y=1.08,x=-.02)
        ii += 1
    
    cb = fig.colorbar(pcm,ax=axs[vv+1,:].flatten(),fraction=0.0075,pad=0.01)
    cb.ax.tick_params(labelsize=fsz_tick)
    termname = r"$\sigma (\frac{h'}{\overline{h}} \, \frac{ \partial \overline{%s}}{\partial t})$" % vname[0]
    cb.set_label("%s (%s)" % (termname,vunit),fontsize=fsz_axis)
    
            
savename = "%sMLD_Ratio_Draft03.png" % (figpath)
plt.savefig(savename,dpi=150,bbox_inches='tight')           
    

#%% Plot the same thing but for ann mean intearnnual variance

fig,axs,mdict = viz.init_orthomap(1,3,bboxplot=bboxplot,figsize=(24,8))

for ax in axs.flatten():
    ax = viz.add_coast_grid(ax,bbox=bboxplot,
                            fill_color="lightgray",fontsize=fsz_tick)

ii = 0

axisorders = [1,2,0]

for vv in range(3):
    
    ax      = axs[axisorders[vv]]
    
    if vv == 0:
        vmax     = 0.5
        cints    = np.arange(0.05,0.32,0.04)
        cmap     = cm.lajolla_r#'cmo.thermal'
        vunit    = '$\degree$C/mon'
        vname    = "SST"
        lc       = "k"
    elif vv == 1:
        vmax     = 0.015
        cints    = np.arange(0.005,0.25,0.005)
        cmap     = cm.acton_r#'cmo.rain'
        vunit    = 'psu/mon'
        vname    = "SSS"
        lc       = "lightgray"#"cyan"
    
    if vv < 2:
        plotvar = inmonstds[vv].mean('ens').mean('month') * mask
        
        pcm     = ax.pcolormesh(plotvar.lon,plotvar.lat,plotvar,
                                transform=proj,vmax=vmax,vmin=0.,cmap=cmap)
        
        cl      = ax.contour(plotvar.lon,plotvar.lat,plotvar,
                                 levels=cints,transform=proj,
                                 colors=lc,linewidths=0.75)
        
    else:
        plotvar = da_hratio[vv].mean('mon') * mask
        
        cmap    = 'cmo.deep_r'
        cints   = np.arange(0,0.64,0.04)
        lc      = 'lightgray'
        
        pcm     = ax.contourf(plotvar.lon,plotvar.lat,plotvar,
                                transform=proj,levels=cints,cmap=cmap)
        
        cl      = ax.contour(plotvar.lon,plotvar.lat,plotvar,
                                 levels=cints,transform=proj,
                                 colors=lc,linewidths=0.75)
        
    
    ax.clabel(cl,fontsize=fsz_tick)
    cb = viz.hcbar(pcm,ax=ax,fraction=0.045,pad=0.01)
    cb.ax.tick_params(labelsize=fsz_tick)
    if vv <2:
        
        termname = r"$\sigma (\frac{h'}{\overline{h}} \, \frac{ \partial \overline{%s}}{\partial t})$" % vname[-1]
    else:
        termname = r"$\frac{\sigma(h')}{\overline{h}}$"
        vunit    = "meters"
    cb.set_label("%s (%s)" % (termname,vunit),fontsize=fsz_axis)
    
    
    
    # Plot Other Features
    ax.plot(ds_gs2.lon.isel(mon=kmonth),ds_gs2.lat.isel(mon=kmonth),transform=proj,lw=1.75,c='cornflowerblue',ls='dashdot')

    # Plot Ice Edge
    ax.contour(icemask.lon,icemask.lat,mask_plot,colors="cyan",linewidths=2.5,
               transform=proj,levels=[0,1],zorder=-1)
    
    viz.label_sp(axisorders[vv],alpha=0.75,ax=ax,fontsize=fsz_title,y=1.08,x=-.02)
    ii += 1
    
    
savename = "%sMLD_Ratio_Draft04_InterannStd.png" % (figpath)
plt.savefig(savename,dpi=150,bbox_inches='tight')   
    
#%% Replot above but for annual variance

inmonstds_total = [ds.std('time') for ds in savevars]

fig,axs,mdict = viz.init_orthomap(1,3,bboxplot=bboxplot,figsize=(24,8))

for ax in axs.flatten():
    ax = viz.add_coast_grid(ax,bbox=bboxplot,
                            fill_color="lightgray",fontsize=fsz_tick)

ii = 0

axisorders = [1,2,0]

for vv in range(3):
    
    ax      = axs[axisorders[vv]]
    
    if vv == 0:
        vmax     = 0.5
        cints    = np.arange(0.05,1.05,0.05)
        cmap     = 'cmo.thermal'
        vunit    = '$\degree$C/mon'
        vname    = "SST"
        lc       = "k"
    elif vv == 1:
        vmax     = 0.015
        cints    = np.arange(0.005,0.25,0.005)
        cmap     = 'cmo.rain'
        vunit    = 'psu/mon'
        vname    = "SSS"
        lc       = "cyan"
    
    if vv < 2:
        plotvar = inmonstds_total[vv].mean('ens') * mask
        
        pcm     = ax.pcolormesh(plotvar.lon,plotvar.lat,plotvar,
                                transform=proj,vmax=vmax,vmin=0.,cmap=cmap)
        
        cl      = ax.contour(plotvar.lon,plotvar.lat,plotvar,
                                 levels=cints,transform=proj,
                                 colors=lc,linewidths=0.75)
        
    else:
        plotvar = da_hratio_total[vv] * mask
        
        cmap    = 'cmo.deep_r'
        cints   = np.arange(0,1.05,0.05)
        lc      = 'lightgray'
        
        pcm     = ax.contourf(plotvar.lon,plotvar.lat,plotvar,
                                transform=proj,levels=cints,cmap=cmap)
        
        cl      = ax.contour(plotvar.lon,plotvar.lat,plotvar,
                                 levels=cints,transform=proj,
                                 colors=lc,linewidths=0.75)
        
    
    ax.clabel(cl,fontsize=fsz_tick)
    cb = viz.hcbar(pcm,ax=ax,fraction=0.025,pad=0.01)
    cb.ax.tick_params(labelsize=fsz_tick)
    if vv <2:
        
        termname = r"$\sigma (\frac{h'}{\overline{h}} \, \frac{ \partial \overline{%s}}{\partial t})$" % vname[-1]
    else:
        termname = r"$\frac{\sigma(h')}{\overline{h}}$"
        vunit    = "meters"
    cb.set_label("%s (%s)" % (termname,vunit),fontsize=fsz_axis)
    
    
    
    # Plot Other Features
    ax.plot(ds_gs2.lon.isel(mon=kmonth),ds_gs2.lat.isel(mon=kmonth),transform=proj,lw=1.75,c='cornflowerblue',ls='dashdot')

    # Plot Ice Edge
    ax.contour(icemask.lon,icemask.lat,mask_plot,colors="cyan",linewidths=2.5,
               transform=proj,levels=[0,1],zorder=-1)
    
    viz.label_sp(axisorders[vv],alpha=0.75,ax=ax,fontsize=fsz_title,y=1.08,x=-.02)
    ii += 1
    
savename = "%sMLD_Ratio_Draft04_TotalStd.png" % (figpath)
plt.savefig(savename,dpi=150,bbox_inches='tight')   
        

#%%

lonf = -30
latf = 50
proc.selpt_ds(dTbar_dt,lonf,latf).isel(ensemble=0)


proc.selpt_ds(Tbar,lonf,latf).isel(ensemble=0)


#%% Retrieve the components


