#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Compare detrending effect on detrainment damping (linear vs. ens mean)


Created on Wed Feb 28 10:36:58 2024

@author: gliu
"""


import time
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import xarray as xr
import sys
from tqdm import tqdm
import scipy as sp
import cartopy.crs as ccrs

import matplotlib as mpl

#%%

stormtrack = 0
if stormtrack:
    amvpath = "/home/glliu/00_Scripts/01_Projects/00_Commons/" # amv module
    scmpath = "/home/glliu/00_Scripts/01_Projects/01_AMV/02_stochmod/stochmod/model/" # scm module
else:
    amvpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/03_Scripts/" # amv module
    scmpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/03_Scripts/stochmod/model/"

sys.path.append(amvpath)
sys.path.append(scmpath)

from amv import proc,viz
import amv.loaders as dl

#%% Indicate the files

# compare_name = 'compareDT_TEMP_ens01'
# fns          = ["CESM1_HTR_FULL_lbd_d_params_TEMP_detrendlinear_lagmax3_ens01_regridNN.nc",
#        "CESM1_HTR_FULL_lbd_d_params_TEMP_detrendensmean_lagmax3_ens01_regridNN.nc"]
# fnnames      = ["Detrend (Linear)","Detrend (Ens. Mean)"]
#outpath      = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/proc/CESM1/NATL_proc/ocn_var_3d/"
#mult = -1 # Use this if the data is raw, not flipped through preprocess_SSS Input

# compare_name = 'compare_EnsAvg_SSS'
# fns          = ["CESM1_HTR_FULL_SSS_Expfit_lbdd_monvar_detrendensmean_lagmax3_EnsAvg.nc",
#                 "CESM1_HTR_FULL_SSS_Expfit_lbdd_monvar_detrendensmean_lagmax3_Ens01.nc"]
# fnnames      = ["Ens Avg.","Ens. 01"]
# outpath      = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/proc/model_input/damping/"
# mult         = 1

# Compare correlation based detrainment damping
compare_name = 'compare_Detrainment_0424_SST_SSS'
fns          = ["CESM1_HTR_FULL_corr_d_TEMP_detrendensmean_lagmax3_interp1_imshift1_dtdepth1_EnsAvg.nc",
                "CESM1_HTR_FULL_corr_d_SALT_detrendensmean_lagmax3_interp1_imshift1_dtdepth1_EnsAvg.nc",
                "CESM1_HTR_FULL_SST_Expfit_lbdd_monvar_detrendensmean_lagmax3_EnsAvg.nc",
                "CESM1_HTR_FULL_SSS_Expfit_lbdd_monvar_detrendensmean_lagmax3_EnsAvg.nc"]
fnnames      = ["Corr (SST)","Corr (SSS)","Exp Fit (SST)","Exp Fit (SSS)"]
outpath      = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/proc/model_input/damping/"
mult         = 1

# Other PAths
rawpath      = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/proc/CESM1/NATL_proc/"
figpath      = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/02_Figures/20240425/"
proc.makedir(figpath)

def init_monplot():
    plotmon       = np.roll(np.arange(12),1)
    fig,axs,mdict = viz.init_orthomap(4,3,bboxplot=bboxplot,figsize=(18,18))
    for a,ax in enumerate(axs.flatten()):
        im = plotmon[a]
        ax = viz.add_coast_grid(ax,bbox=bboxplot,fill_color="lightgray")
        ax.set_title(mons3[im],fontsize=fsz_axis)
    return fig,axs


#%% Load the files

nf = len(fns)
ds_all = []
for ff in range(nf):
    ncname=outpath + fns[ff]
    ds = xr.open_dataset(ncname).load()
    ds_all.append(ds)
    
    
#%% Do some conversions

dt = 3600*24*30
corr_ori = []
for ff in range(nf):
    if "corr_d" in fns[ff]:
        print("Applying Correction to detrainment damping for %s" % fnnames[ff])
        
        corr_ori.append(ds_all[ff].copy())
        
        ds_all[ff] = np.log(ds_all[ff])
        
        

    
#if compare_name == 'compare_Detrainment_0424_SST_SSS':
    


#%% Visualize lbd_d for a particular month

# Plotting Params
mpl.rcParams['font.family'] = 'JetBrains Mono'
bboxplot                    = [-80,0,20,65]
proj                        = ccrs.PlateCarree()
lon                         = ds.lon.values
lat                         = ds.lat.values
mons3                       = proc.get_monstr()



vlms_base = [0,.3]
vlms_diff = [-.2,.2]
im        = 0

for im in range(12):
    cints_base     = [12,24,36,48,60,72]
    cints_diff     = [-12,-6,-3,0,3,6,12]
    
    
    fig,axs,mdict                = viz.init_orthomap(1,3,bboxplot,figsize=(14,8),constrained_layout=True,)
    for a,ax in enumerate(axs):
        
        ax                          = viz.add_coast_grid(ax,bbox=bboxplot,fill_color="lightgray")
        
        if a <2:
            plotvar = ds_all[a].lbd_d.isel(mon=im) * -mult
            cmap    = "inferno"
            title   = fnnames[a]
            vlms    = vlms_base
            cints   = cints_base
            linecol = "w"
            
        else:
            plotvar = (-ds_all[1].lbd_d.isel(mon=im) - -ds_all[0].lbd_d.isel(mon=im))
            cmap    = 'cmo.balance'
            title   =  "%s - %s" % (fnnames[1],fnnames[0])
            vlms    = vlms_diff
            cints   = cints_diff
            linecol = "k"
            
        ax.set_title(title)
        pcm     = ax.pcolormesh(lon,lat,plotvar,transform=proj,vmin=vlms[0],vmax=vlms[1],cmap=cmaps[vv])
        
        # Contours
        if a <2:
            cl      = ax.contour(lon,lat,1/plotvar,levels=cints,colors=linecol,linewidths=.55,transform=proj)
            ax.clabel(cl,fontsize=8)
            
        # Colorbar
        cb      = fig.colorbar(pcm,ax=ax,orientation='horizontal',fraction=0.02)
        cb.set_label("Detrainment Damping (1/mon)")
        
    plt.suptitle("%s. Detrainment Damping Comparison" % (mons3[im]),y=0.60,fontsize=16)
    
    savename = "%sDetrainment_Damping_Detrend_Comparison_%s_Ens01_mon%02i.png" % (figpath,compare_name,im+1)
    plt.savefig(savename,dpi=150,bbox_inches='tight')

#%% Copy Above, but for 1 month



vlms_base = None#,[0,.3]
vlms_diff = None#,[-.2,.2]
im        = 0


cints_base     = [12,24,36,48,60,72]
cints_diff     = [-12,-6,-3,0,3,6,12]


fig,axs,mdict                   = viz.init_orthomap(1,3,bboxplot,figsize=(14,8),constrained_layout=True,)
for a,ax in enumerate(axs):
    
    ax                          = viz.add_coast_grid(ax,bbox=bboxplot,fill_color="lightgray")
    
    if a <2:
        plotvar = ds_all[a].lbd_d.isel(mon=im) 
        cmap    = "inferno"
        title   = fnnames[a]
        vlms    = vlms_base
        cints   = cints_base
        linecol = "w"
        
    else:
        plotvar = (-ds_all[1].lbd_d.isel(mon=im) - -ds_all[0].lbd_d.isel(mon=im))
        cmap    = 'cmo.balance'
        title   =  "%s - %s" % (fnnames[1],fnnames[0])
        vlms    = vlms_diff
        cints   = cints_diff
        linecol = "k"
    
    if a == 0:
        vlms = [-.5,0]
    elif a == 1:
        vlms = [0,.2]
    
    ax.set_title(title)
    if vlms is None:
        pcm     = ax.pcolormesh(lon,lat,plotvar,transform=proj,cmap=cmap)
    else:
        pcm     = ax.pcolormesh(lon,lat,plotvar,transform=proj,vmin=vlms[0],vmax=vlms[1],cmap=cmap)
    
    # Contours
    if a <2:
        cl      = ax.contour(lon,lat,1/plotvar,levels=cints,colors=linecol,linewidths=.55,transform=proj)
        ax.clabel(cl,fontsize=8)
        
    # Colorbar
    cb      = fig.colorbar(pcm,ax=ax,orientation='horizontal',fraction=0.02)
    cb.set_label("Detrainment Damping (1/mon)")
    
plt.suptitle("%s. Detrainment Damping Comparison" % (mons3[im]),y=0.60,fontsize=16)

#%% Updated Plots Here (for the four comparisons)


# Settings and load
bsf      = dl.load_bsf()

# Load Land Ice Mask
icemask  = xr.open_dataset(outpath + "../masks/CESM1LE_HTR_limask_pacificmask_enssum_lon-90to20_lat0to90.nc")

# Resize
bsf,icemask,_    = proc.resize_ds([bsf,icemask,ds_all[0]])
bsf_savg = proc.calc_savg_mon(bsf)

#
mask = icemask.MASK.squeeze()
mask_plot = xr.where(np.isnan(mask),0,mask)#mask.copy()

#mask_plot[np.isnan(mask)] = 0

#%% Visualize Detrianment Damping (Most recent version as of 2024.04)
# Note that running the "correction" leads to erroneous visualizations
# Need to check deeper for potential differences

fsz_title= 25
fsz_axis = 20
fsz_ticks= 18

# Note: Just works for correlation case

# Take just the correlation values
ds_corr   = [ds_all[0],ds_all[1]]
ds_names  = ["SST","SSS"]

plot_sids = [1,3,0]

cints_bsf = np.arange(-50,60,10)

vlms      = [0.25,1]#None

# Compute seasonal averages
ds_savg   = [proc.calc_savg_mon(ds.lbd_d) for ds in ds_corr]

cmap      = 'inferno'

# Make the plots
fig,axs,mdict                = viz.init_orthomap(2,3,bboxplot,figsize=(22,9),constrained_layout=True,)
for ff in range(2):
    
    
    for ss in range(3):
        sid    = plot_sids[ss]
        dsplot = ds_savg[ff].isel(season=sid)
        
        ax     = axs[ff,ss]
        ax     = viz.add_coast_grid(ax,bbox=bboxplot,fill_color="lightgray")
        
        if ff == 0:
            ax.set_title(dsplot.season.values,fontsize=fsz_title)
        
        
        if vlms is None:
            pcm     = ax.pcolormesh(lon,lat,dsplot,transform=proj,cmap=cmap)
        else:
            pcm     = ax.pcolormesh(lon,lat,dsplot,transform=proj,vmin=vlms[0],vmax=vlms[1],cmap=cmap)
        
        # Colorbar
        if vlms is None:
            cb      = fig.colorbar(pcm,ax=ax,orientation='horizontal',fraction=0.02)
            cb.set_label("Detrainment Damping (Correlation)")
            
        if ss == 0:
            viz.add_ylabel(ds_names[ff],ax=ax,x=-0.15,y=0.5,fontsize=fsz_title,rotation='horizontal')
        
        
        # Plot BSF
        plotbsf = bsf_savg.isel(season=sid).BSF * mask
        cl      = ax.contour(plotbsf.lon,plotbsf.lat,plotbsf,
                             levels=cints_bsf,colors="k",linewidths=0.75,transform=proj)
        
        
        # Plot Mask
        cl2      = ax.contour(mask.lon,mask.lat,mask_plot,
                             levels=[0,1,2],colors="w",linewidths=2,transform=proj)

if vlms is not None:
    cb = fig.colorbar(pcm,ax=axs.flatten(),orientation='vertical',fraction=0.02,pad=0.01)
    cb.set_label("Detrainment Damping (Correlation)",fontsize=fsz_axis)
    
#plt.suptitle("%s. Detrainment Damping Comparison" % (mons3[im]),y=0.60,fontsize=16)
savename = "%sDetrainment_Damping_Corrd_SeasonalAvg_EnsAvg_%s.png" % (figpath,compare_name)
plt.savefig(savename,dpi=150,bbox_inches='tight')
#ds_savg = [ds.lbd_d.assign_coords(mon=proc.get_xryear()).rename({'mon':'time'}) for ds in ds_corr]
#ds_savg = [ds.groupby('time.season').mean('time') for ds in ds_savg]

#%% Plot Seasonal Cycle in Differences

fig,axs,_ = viz.init_orthomap(4,3,bboxplot,figsize=(16,16))

plotmon = np.roll(np.arange(12),1)
for a in range(12):
    
    ax = axs.flatten()[a]
    im = plotmon[a]
    ax = viz.add_coast_grid(ax,bbox=bboxplot,fill_color="lightgray")
    
    diff = (ds_corr[1] - ds_corr[0]).lbd_d.isel(mon=im)
    ax.set_title(mons3[im],fontsize=fsz_axis)
    pcm  = ax.pcolormesh(diff.lon,diff.lat,diff,transform=proj,
                         cmap="RdBu_r",vmin=-.5,vmax=.5)
    
    #viz.hcbar(pcm,ax=ax)
    
cb = viz.hcbar(pcm,ax=axs.flatten(),fraction=0.025)
cb.set_label("Correlation ($\lambda^d_{SALT}$ - $\lambda^d_{TEMP}$)",fontsize=fsz_axis)
plt.suptitle("Difference in Detrainment Damping",fontsize=fsz_title)

#fig,axs = plt.subplots(4,3,constrained_layout=True,subplot_kw={'projection':proj})

savename = "%sDetrainment_Damping_Corrd_MonthlyDiff_EnsAvg_%s.png" % (figpath,compare_name)
plt.savefig(savename,dpi=150,bbox_inches='tight')

#%% Plot Differences over key months (as in the first plot above)

vlms = [-.5,.5]
cmap = 'cmo.balance'

# Make the plots
fig,axs,mdict                = viz.init_orthomap(1,3,bboxplot,figsize=(22,5),constrained_layout=True,)

for ss in range(3):
    sid    = plot_sids[ss]
    dsplot = (ds_savg[1] - ds_savg[0]).isel(season=sid)
    
    ax     = axs[ss]
    ax     = viz.add_coast_grid(ax,bbox=bboxplot,fill_color="lightgray")
    ax.set_title(dsplot.season.values,fontsize=fsz_title)
    
    if ff == 0:
        ax.set_title(dsplot.season.values,fontsize=fsz_title)
    
    
    if vlms is None:
        pcm     = ax.pcolormesh(lon,lat,dsplot,transform=proj,cmap=cmap)
    else:
        pcm     = ax.pcolormesh(lon,lat,dsplot,transform=proj,vmin=vlms[0],vmax=vlms[1],cmap=cmap)
    
    # Colorbar
    if vlms is None:
        cb      = fig.colorbar(pcm,ax=ax,orientation='horizontal',fraction=0.02)
        cb.set_label("Detrainment Damping (Correlation)")
        

    
    
    # Plot BSF
    plotbsf = bsf_savg.isel(season=sid).BSF * mask
    cl      = ax.contour(plotbsf.lon,plotbsf.lat,plotbsf,
                         levels=cints_bsf,colors="k",linewidths=0.75,transform=proj)
    
    
    # Plot Mask
    cl2      = ax.contour(mask.lon,mask.lat,mask_plot,
                         levels=[0,1,2],colors="w",linewidths=2,transform=proj)

if vlms is not None:
    cb = fig.colorbar(pcm,ax=axs.flatten(),orientation='vertical',fraction=0.02,pad=0.01)
    cb.set_label("Corr(Detrain,Entrain)\n$SALT$ - $TEMP$",fontsize=fsz_axis)
    
#plt.suptitle("Seasonal Mean Differences in Detrainment Correlation",y=1.1,fontsize=32)
savename = "%sDetrainment_Damping_Corrd_SeasonalDiff_EnsAvg_%s.png" % (figpath,compare_name)
plt.savefig(savename,dpi=150,bbox_inches='tight')


#%% Examine Seasonal Cycle in Detrainment Damping (SALT and TEMP) at specific locations

lonf           = -47
latf           = 47
ds_cols        = ["hotpink","navy"]

locfn,loctitle = proc.make_locstring(lonf,latf)

ytks           = np.arange(0,1.1,0.1)
fig,ax         = viz.init_monplot(1,1)

for vv in range(2):
    plotvar = ds_corr[vv].sel(lon=lonf,lat=latf,method='nearest').lbd_d
    
    ax.plot(mons3,plotvar,label=ds_names[vv],lw=2.5,marker="o",c=ds_cols[vv])
ax.legend()

ax.set_yticks(ytks)
ax.set_ylabel("Corr(Detrain Month,Entrain Month-1)")
ax.set_xlabel("Entraining Month")
ax.set_title("Detrainment Damping @ %s" % loctitle)
savename = "%sDetrainment_Damping_SALT_v_TEMP_%s_EnsAvg.png" % (figpath,locfn)
plt.savefig(savename,dpi=150,bbox_inches='tight')

#%% 2024.05.24: Look at the Vertical Gradients

ocnpath        = rawpath + "ocn_var_3d/"
dstemp_dtgrad  = xr.open_dataset(ocnpath + "CESM1_HTR_TEMP_Detrain_Gradients.nc").load()
dssalt_dtgrad  = xr.open_dataset(ocnpath + "CESM1_HTR_SALT_Detrain_Gradients.nc").load()

dstemp_dtgrad2 = xr.open_dataset(ocnpath + "CESM1_HTR_TEMP_Detrain_Gradients_dz2.nc").load()
dssalt_dtgrad2 = xr.open_dataset(ocnpath + "CESM1_HTR_SALT_Detrain_Gradients_dz2.nc").load()

    
#%% Compute Normalized Forms (copied from box below, eventually replace this)

nfactor_names = ['spatial','seasonal']


ds_norm_all     = []
nfactors_all    = []

for nn in range(2): # Loop by normfactor

    ds_norm     = []
    nfactors    = []
    
    for ff in range(2):  # Loop by variable
        
        dsplot = ds_savg[ff]#.isel(season=sid)
        
        if nn == 0: # Spatial Mean
            nfactor  = np.abs(dsplot*mask_plot).mean('lat').mean('lon')
        elif nn == 1: # Seasonal Mean (at each point)
            nfactor  = np.abs(ds_corr[ff].mean('mon').mean('ens')).mean('entrain_mon').grad
        
        nfactors.append(nfactor)
        ds_norm.append((dsplot/nfactor).copy())
    
    nfactors_all.append(nfactors)
    ds_norm_all.append(ds_norm)
    
        
            
        


#%% Plot Vertical gradients (copied from SALT and TEMP detrainment damping above) (seasonal Mean)

ds_corr   = [dstemp_dtgrad,dssalt_dtgrad]
ds_names  = ["SST","SSS"]
vunits    = [r"$\frac{dT}{dz}$ ($\degree C$/meter)",r"$\frac{dS}{dz}$ (psu/meter)"]
vnames    = ["TEMP","SALT"]
vcolors   = ["hotpink","navy"]
plot_sids = [1,3,0]

#% Set the intervals for the barotropic streamfunction
cints_bsf = np.arange(-50,60,10)

# First, take mean over year
ds_corr_in= [ds.mean('mon').rename({'entrain_mon':'mon'}).mean('ens') for ds in ds_corr]

# Compute seasonal averages
ds_savg   = [proc.calc_savg_mon(ds.grad) for ds in ds_corr_in]

# Set the Colormap
cmap      = 'cmo.balance'

# Normfactor [None,spatial,seasonal]
normfactor = 'seasonal'

# Colorbar limits (if there is not normalization factor)
if normfactor is None:
    vlms_all  = ([-.08,.08],[-.008,.008])#
else:
    vlms_all = ([-2,2],[-2,2])

# Make the plots
fig,axs,mdict                = viz.init_orthomap(2,3,bboxplot,figsize=(22,9),constrained_layout=True,)
for ff in range(2):
    
    vlms = vlms_all[ff]
    for ss in range(3):
        
        sid    = plot_sids[ss]
        
        dsplot = ds_savg[ff].isel(season=sid)
        
        if normfactor is not None:
            
            if normfactor == "spatial":
                dsplot = ds_norm_all[0][ff].isel(season=sid)#nfactor  = np.nanmean(np.abs(dsplot*mask_plot)) # Single Value
            elif normfactor == "seasonal":
                dsplot = ds_norm_all[1][ff].isel(season=sid)
                #nfactor = np.abs(ds_corr[ff].mean('mon').mean('ens')).mean('entrain_mon').grad
                #dsplot   = dsplot.grad
            #dsplot =dsplot / nfactor
        
        ax     = axs[ff,ss]
        ax     = viz.add_coast_grid(ax,bbox=bboxplot,fill_color="lightgray")
        
        if ff == 0:
            ax.set_title(dsplot.season.values,fontsize=fsz_title)
        
        
        if vlms is None:
            pcm     = ax.pcolormesh(lon,lat,dsplot,transform=proj,cmap=cmap)
        else:
            pcm     = ax.pcolormesh(lon,lat,dsplot,transform=proj,vmin=vlms[0],vmax=vlms[1],cmap=cmap)
        
        # Colorbar
        if vlms is None:
            cb      = fig.colorbar(pcm,ax=ax,orientation='horizontal',fraction=0.02)
            cb.set_label("Detrainment Damping (Correlation)")
            
        if ss == 0:
            viz.add_ylabel(ds_names[ff],ax=ax,x=-0.15,y=0.5,fontsize=fsz_title,rotation='horizontal')
        
        # Plot BSF
        plotbsf = bsf_savg.isel(season=sid).BSF * mask
        cl      = ax.contour(plotbsf.lon,plotbsf.lat,plotbsf,
                             levels=cints_bsf,colors="k",linewidths=0.75,transform=proj)
        
        # Plot Mask
        cl2      = ax.contour(mask.lon,mask.lat,mask_plot,
                             levels=[0,1,2],colors="w",linewidths=2,transform=proj)
    
    if vlms is not None:
        cb = fig.colorbar(pcm,ax=axs[ff,:].flatten(),orientation='vertical',fraction=0.02,pad=0.01)
        cb.set_label("%s" % vunits[ff],fontsize=fsz_axis)
    
#plt.suptitle("%s. Detrainment Damping Comparison" % (mons3[im]),y=0.60,fontsize=16)
savename = "%sVerticalGradients_SeasonalAvg_EnsAvg_%s_norm%s.png" % (figpath,compare_name,normfactor)
plt.savefig(savename,dpi=150,bbox_inches='tight')
#ds_savg = [ds.lbd_d.assign_coords(mon=proc.get_xryear()).rename({'mon':'time'}) for ds in ds_corr]
#ds_savg = [ds.groupby('time.season').mean('time') for ds in ds_savg]

# -------------------------------------------------------
#%% Try computing mean for only for the detraining period

# Load kprev
hpath    = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/proc/model_input/mld/"
ds_kprev = xr.open_dataset(hpath + "CESM1_HTR_FULL_kprev_NAtl.nc").load()

# Resize to match domain in 3d ocn
ds_kprev,_ = proc.resize_ds([ds_kprev,dstemp_dtgrad])

# Debug, delete this later
# kprev = ds_kprev.isel(lat=22,lon=22,ens=1).h.values
# dsin  = dstemp_dtgrad.isel(lat=22,lon=22,ens=1).grad.values

def extract_detrain(dsin,kprev,debug=False,use_abs=True):
    
    # Identify months where detrainment is not occuring
    dtmon = kprev == 0.
    
    val_out = np.zeros(12) * np.nan
    for eim in range(12):
        entrain_mon = eim + 1
        detrain_mon = int(np.round(kprev[eim]))
        #dtmons = np.arange(detrain_mon)
        
        if detrain_mon == 0:
            continue
        if entrain_mon < detrain_mon:
            dtmons = np.hstack([np.arange(detrain_mon,12+1),np.arange(0,entrain_mon+1)])
        elif entrain_mon == detrain_mon:
            dtmons = np.arange(1,13,1)
        else:
            dtmons = np.arange(detrain_mon,entrain_mon+1)
        
        if debug:
            print("\n")
            print("Entrain Month %i"%entrain_mon)
            print("Detrain Month %i"%detrain_mon)
            print(dtmons)
        
        dsdt = dsin[eim,dtmons-1]
        if use_abs:
            val_out[eim] = np.nanmean(np.abs(dsdt))
        else:
            val_out[eim] = np.nanmean(dsdt)

    return val_out


detrainmean_byvar = []
for vv in range(2):
    
    detrainmean= xr.apply_ufunc(
        extract_detrain,
        ds_corr[vv].grad,
        ds_kprev.h,
        input_core_dims=[['entrain_mon','mon'],['mon']],
        output_core_dims=[['mon']],
        vectorize=True,
        )
    detrainmean_byvar.append(detrainmean.copy())

# Save the Output
edict = {'grad':{'zlib':True}}
for vv in range(2):
    outname = "%sCESM1_HTR_%s_Detrain_Gradients_detrainmean.nc" % (ocnpath,vnames[vv])
    detrainmean_byvar[vv].rename('grad').to_netcdf(outname,encoding=edict)
    
#%% Do the same, but for the standard deviation values

ds_Tstd = xr.open_dataset(ocnpath + "CESM1_HTR_TEMP_Detrain_Stdev.nc").load()
ds_Sstd = xr.open_dataset(ocnpath + "CESM1_HTR_SALT_Detrain_Stdev.nc").load()
vnames = ["TEMP","SALT"]
instds = [ds_Tstd,ds_Sstd]
detrainmean_std = []
for vv in range(2):
    meanstds= xr.apply_ufunc(
        extract_detrain,
        instds[vv]['std'],
        ds_kprev.h,
        input_core_dims=[['entrain_mon','mon'],['mon']],
        output_core_dims=[['mon']],
        vectorize=True,
        )
    detrainmean_std.append(meanstds.copy())
    
    
    # Save the Output
    edict = {'stdev':{'zlib':True}}
    outname = "%sCESM1_HTR_%s_Detrain_stdev_detrainmean.nc" % (ocnpath,vnames[vv])
    meanstds.rename('stdev').to_netcdf(outname,encoding=edict)



#% Debugg
# kprev = ds_kprev.isel(lat=22,lon=22,ens=1).h.values
# dsin  = ds_Tstd.isel(lat=22,lon=22,ens=1)['std'].values

#%% Do the same but for 2nd derivative ---------------------------------------
ds_corr2 = [dstemp_dtgrad2,dssalt_dtgrad2]
detrainmean_byvar2 = []

for vv in range(2):
    
    detrainmean= xr.apply_ufunc(
        extract_detrain,
        ds_corr2[vv].grad,
        ds_kprev.h,
        input_core_dims=[['entrain_mon','mon'],['mon']],
        output_core_dims=[['mon']],
        vectorize=True,
        )
    detrainmean_byvar2.append(detrainmean.copy())

# Save the Output
edict = {'grad':{'zlib':True}}
for vv in range(2):
    outname = "%sCESM1_HTR_%s_Detrain_Gradients_detrainmean_dz2.nc" % (ocnpath,vnames[vv])
    detrainmean_byvar2[vv].rename('grad').to_netcdf(outname,encoding=edict)
    
    

#%% Load output from above

detrainmean_byvar = []
meanstds          = []
for vv in range(2):
    
    outname = "%sCESM1_HTR_%s_Detrain_Gradients_detrainmean.nc" % (ocnpath,vnames[vv])
    dsload  = xr.open_dataset(outname).load()
    detrainmean_byvar.append(dsload.copy())
    
    outname = "%sCESM1_HTR_%s_Detrain_stdev_detrainmean.nc" % (ocnpath,vnames[vv])
    dsload  = xr.open_dataset(outname).load()
    meanstds.append(dsload.copy())
    



#%% Visualize Normalization

kmonths = [8,9,10]
sid     = 3
vlms    = None
fig,axs,_ = viz.init_orthomap(2,3,bboxplot,figsize=(12,6.5))

vlims_in = np.array([[[0,0.25], [0,1.5] ,[0,.3]],
                     [[0,0.025],[0,.3], [0,.3]]])


# vunits_in = np.array([["$\frac{\degree C}{m}$","$\degree C$","$m^{-1}$",]
#              ["$\frac{psu}{m}$","$psu$","$m^{-1}$",]])


cmaps    = ['cmo.thermal','cmo.haline']
vunits   = ["\degree C","psu"]
for vv in range(2):
    for ii in range(3):
        ax    = axs[vv,ii]
        vlms  = vlims_in[vv,ii,:]
        if ii == 0:
            title    = "Absolute Gradient"
            plotvar  = detrainmean_byvar[vv].isel(mon=kmonths).mean('mon').mean('ens').grad
            vunit    = r"$\frac{%s}{m}$" % (vunits[vv])
        elif ii == 1:
            title = "Std. Dev."
            plotvar = meanstds[vv].isel(mon=kmonths).mean('mon').mean('ens').stdev
            vunit    = "$%s$" % vunits[vv]
        elif ii == 2:
            title = "Normalized Gradient"
            grads = detrainmean_byvar[vv].isel(mon=kmonths).mean('mon').mean('ens').grad
            stdev = meanstds[vv].isel(mon=kmonths).mean('mon').mean('ens').stdev
            plotvar = grads/stdev
            vunit   = r"$m^{-1}$"
            
        ax     = viz.add_coast_grid(ax,bbox=bboxplot,fill_color="lightgray")
        if vv == 0:
            ax.set_title(title)
            
        
        if ii == 0:
            viz.add_ylabel(vnames[vv],ax=ax,x=-0.20,y=0.5,fontsize=fsz_title,rotation='horizontal')
            
        # Plot Things
        if vlms is None:
            pcm     = ax.pcolormesh(lon,lat,plotvar,transform=proj,cmap=cmaps[vv])
        else:
            pcm     = ax.pcolormesh(lon,lat,plotvar,transform=proj,vmin=vlms[0],vmax=vlms[1],cmap=cmaps[vv])

        cb      = fig.colorbar(pcm,ax=ax,orientation='horizontal',fraction=0.02)
        cb.set_label(vunit)
        
        
        # Plot BSF
        plotbsf = bsf_savg.isel(season=sid).BSF * mask
        cl      = ax.contour(plotbsf.lon,plotbsf.lat,plotbsf,
                             levels=cints_bsf,colors="gray",linewidths=0.75,transform=proj)
        
        
        # Plot Mask
        cl2      = ax.contour(mask.lon,mask.lat,mask_plot,
                             levels=[0,1,2],colors="w",linewidths=2,transform=proj)
        
savename = "%sNormalized_Vertical_Diff_SeasonalDiff_EnsAvg_%s_SON.png" % (figpath,compare_name)
plt.savefig(savename,dpi=150,bbox_inches='tight')

#%% Plot vertical gradient differences, normalized in different ways


vlms       = [-1,1]
cmap       = 'cmo.balance'
# Normfactor [None,spatial,seasonal]
normfactor = 'spatial'

# Make the plots
fig,axs,mdict                = viz.init_orthomap(1,3,bboxplot,figsize=(22,5),constrained_layout=True,)

for ss in range(3):
    
    sid    = plot_sids[ss]
    
    if normfactor is not None:
        
        # Compute Normalization factors
        if normfactor == "spatial":
            ds_in = [ds_norm_all[0][0].isel(season=sid),ds_norm_all[0][1].isel(season=sid)]
        elif normfactor == "seasonal":
            ds_in = [ds_norm_all[1][0].isel(season=sid),ds_norm_all[1][1].isel(season=sid)]
        
    # Take the Difference
    dsplot = np.abs(ds_in[1]) - np.abs(ds_in[0])
    
    ax     = axs[ss]
    ax     = viz.add_coast_grid(ax,bbox=bboxplot,fill_color="lightgray")
    ax.set_title(dsplot.season.values,fontsize=fsz_title)
    
    if ff == 0:
        ax.set_title(dsplot.season.values,fontsize=fsz_title)
    
    
    if vlms is None:
        pcm     = ax.pcolormesh(lon,lat,dsplot,transform=proj,cmap=cmap)
    else:
        pcm     = ax.pcolormesh(lon,lat,dsplot,transform=proj,vmin=vlms[0],vmax=vlms[1],cmap=cmap)
    
    # Colorbar
    if vlms is None:
        cb      = fig.colorbar(pcm,ax=ax,orientation='horizontal',fraction=0.02)
        cb.set_label("Detrainment Damping (Correlation)")
        

    
    
    # Plot BSF
    plotbsf = bsf_savg.isel(season=sid).BSF * mask
    cl      = ax.contour(plotbsf.lon,plotbsf.lat,plotbsf,
                         levels=cints_bsf,colors="k",linewidths=0.75,transform=proj)
    
    
    # Plot Mask
    cl2      = ax.contour(mask.lon,mask.lat,mask_plot,
                         levels=[0,1,2],colors="w",linewidths=2,transform=proj)

if vlms is not None:
    cb = fig.colorbar(pcm,ax=axs.flatten(),orientation='vertical',fraction=0.02,pad=0.01)
    cb.set_label("Diff in Normalized Gradient\n$SALT$ - $TEMP$",fontsize=fsz_axis)
    
#plt.suptitle("Seasonal Mean Differences in Detrainment Correlation",y=1.1,fontsize=32)
savename = "%sNormalized_Vertical_Diff_SeasonalDiff_EnsAvg_%s_%s.png" % (figpath,compare_name,normfactor)
plt.savefig(savename,dpi=150,bbox_inches='tight')


#%% Compare SALT, TEMP, and Vertical gradients at a point

"""

Things To Do
- Plot Detrainment Months
- Corrextly Shade Axis

"""

lonf = -38
latf = 59

locfn,loctitle = proc.make_locstring(lonf,latf)
varlims = ([-.1,.1],[-.01,.01])
fig,ax  = viz.init_monplot(1,1,figsize=(10,4.5))
# Plot the Damping
ls = []
for vv in range(2):
    
    #ax.set_title(ds_names[vv])
    
    lbddin = ds_all[vv].lbd_d.sel(lon=lonf,lat=latf,method='nearest')
    
    p,=ax.plot(mons3,lbddin,c=vcolors[vv],label="$\lambda^d$ (%s)" % ds_names[vv],lw=2.5)

    ls.append(p)
ax.set_ylabel("Correlation(Detrain,Entrain)")
ax.set_xlabel("Entrain Month")
ax.legend()


# Plot the gradients
ax2    = ax.twinx()
ax3    = ax.twinx()
axs_in = [ax2,ax3]
meangrads = []
# Plot the gradient during entraining month
for vv in range(2):
    axin = axs_in[vv]
    
    # Get Point
    grad_in = ds_corr[vv].mean('ens').sel(lon=lonf,lat=latf,method='nearest').grad # [entrain_mon, mon]
    
    # Sort values
    p,=axin.plot(mons3,np.mean(grad_in,0),label=vunits[vv],c=vcolors[vv],ls='dashed')
    
    # Set Axis Colors and limits
    axin.set_ylim(varlims[vv])
    axin.tick_params(axis='y', colors=vcolors[vv])
    axin.spines["right"].set_color(vcolors[vv])
    axin.set_ylabel(vunits[vv],color=vcolors[vv])
    
    # Plot the mean
    meangrad = np.nanmean(np.abs(grad_in))
    axin.axhline([meangrad],lw=0.75,c=vcolors[vv],ls='dashed',label="$\mu_{%s}$=%.3f" % (vnames[vv],meangrad))
    axin.axhline([-meangrad],lw=0.75,c=vcolors[vv],ls='dashed',label="$\mu_{%s}$=%.3f" % (vnames[vv],meangrad))
    meangrads.append(meangrad)
    
    if vv == 1:
        axin.spines["right"].set_position(("axes", 1.15))
    
    ls.append(p)
    #ls.append(p1)
    

# Set up string for mean gradient
meanstr = r"$\mu_{TEMP}$ = %.4f $\frac{\degree C}{m}$, $\mu_{SALT}$ = %.4f $\frac{psu}{m}$" % (meangrads[0],meangrads[1])

# Plot reference lines and legend
axin.axhline([0],lw=0.55,c="k")
ax.legend(ls, [l.get_label() for l in ls],loc='lower center',ncol=2)
ax.set_title("Detrainment Damping and Vertical Gradients @ %s\n%s" % (loctitle,meanstr))
        

savename = "%sVerticalGradients_vs_Detrainment_Corr_EnsAvg_%s_%s.png" % (figpath,compare_name,locfn)
plt.savefig(savename,dpi=150,bbox_inches='tight')


#ax.legend()  

#%% Do a scatterplot of detrainent correlation vs. vertical gradient

# Take this from above
ds_corrd = [ds_all[0].lbd_d,ds_all[1].lbd_d] # [variable][Mon x lat x lon]
ds_corrd = [ds*mask_plot for ds in ds_corrd]
ds_gradz = [np.abs(ds.grad.mean('ens').mean('mon')) * mask_plot for ds in ds_corr]


#%% Simple scatter

sel_mons = [9]
for im in range(12):
    sel_mons=[im]
    fig,axs = plt.subplots(1,2,constrained_layout=True,figsize=(10,4))
    
    for vv in range(2):
        ax = axs[vv]
        ds_in = [ds_corrd[vv],ds_gradz[vv].rename({'entrain_mon':'mon'})]
        
        ds_in = [ds.isel(mon=sel_mons).values.flatten() for ds in ds_in]
        #ax.scatter(ds_corrd[vv].values.flatten(),ds_gradz[vv].values.flatten(),label=ds_names[vv])
        ax.scatter(ds_in[0],ds_in[1],label=ds_names[vv])
        ax.set_ylabel(vunits[vv])
        ax.set_xlabel("Detrainment Correlation")
        ax.set_title("%s, Corr=%.2f" % (ds_names[vv],proc.nancorr(ds_in[0],ds_in[1])[0,1]))
    ax.legend()

# -----------------------------------------
#%% Select variable based detraining months
# -----------------------------------------



