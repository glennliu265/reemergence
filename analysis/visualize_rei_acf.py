#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Visualize the Re-emergence Index (computed in visualize_rememergence CMIP6)
as well as the pointwise autocorrelation functions for CESM1

Plots Included
    - SSS REI and Feb ACF over a specific location/month (SST and SSS)
    - Monthly or DJFM Mean REI over 3 years (SST and SSS, DJFM Ens. Mean)
    - Bounding Box and Regional ACFs
    - REI with BSF (SST and SSS, DJFM Ens. Mean)
    - Pointwise MSE (SM - CESM) over selected lags for a month

Created on Thu May  2 17:46:23 2024

@author: gliu

"""

import xarray as xr
import numpy as np
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import cartopy.crs as ccrs

import matplotlib.pyplot as plt
import sys
import glob
import os

import tqdm
import time

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

#%% 

# Indicate files containing ACFs
cesm_name   = "CESM1_1920to2005_%sACF_lag00to60_ALL_ensALL.nc"
vnames      = ["SST","SSS","TEMP"]

sst_expname = "SM_SST_EOF_LbddCorr_Rerun_SST_autocorrelation_thresALL_lag00to60.nc"
sss_expname = "SM_SSS_EOF_LbddCorr_Rerun_lbdE_SSS_autocorrelation_thresALL_lag00to60.nc"

#%% Load ACFs and REI

acfs_byvar  = []
rei_byvar   = []
for vv in range(3):
    ds = xr.open_dataset(procpath + cesm_name % vnames[vv]).acf.squeeze()
    acfs_byvar.append(ds)
    
    dsrei = xr.open_dataset("%s%s_CESM/Metrics/REI_Pointwise.nc" % (output_path,vnames[vv])).rei.load()
    rei_byvar.append(dsrei)
    
#%% Add ACFs from stochastic model

sm_sss   = xr.open_dataset(procpath+sss_expname).SSS.load()        # (lon: 65, lat: 48, mons: 12, thres: 1, lags: 61)
sm_sst   = xr.open_dataset(procpath+sst_expname).SST.load()

sm_vars  = [sm_sst,sm_sss]

#%% Load mixed layer depth

ds_h    = xr.open_dataset(input_path + "mld/CESM1_HTR_FULL_HMXL_NAtl.nc").h.load()
#id_hmax = np.argmax(ds_h.mean(0),0)

#%% Load BSF and Ice Mask (copied from compare_detrainment_damping)

bsf      = dl.load_bsf()

# Load Land Ice Mask
icemask  = xr.open_dataset(input_path + "masks/CESM1LE_HTR_limask_pacificmask_enssum_lon-90to20_lat0to90.nc")

# Resize
#bsf,icemask,_    = proc.resize_ds([bsf,icemask,acfs_in_rsz[0]])
bsf_savg = proc.calc_savg_mon(bsf)

#
mask = icemask.MASK.squeeze()
mask_plot = xr.where(np.isnan(mask),0,mask)#mask.copy()

mask_apply = icemask.MASK.squeeze().values
#mask_plot[np.isnan(mask)] = 0


#%% Indicate Plotting Parameters (taken from visualize_rem_cmip6)


bboxplot        = [-80,0,10,65]
mpl.rcParams['font.family'] = 'Avenir'
mons3           = proc.get_monstr(nletters=3)

fsz_tick        = 18
fsz_axis        = 20
fsz_title       = 16

rhocrit = proc.ttest_rho(0.05,2,86)

#%% Identify a location

lags      = acfs_byvar[1].lags
xtks      = np.arange(0,37,3)

vcolors   = ["hotpink","navy"]

kmonth    = 2
lonf      = -54#-30
latf      = 59#50
yr        = 0
cints     = np.arange(0,0.55,0.05)

nens      = 42

locfn,loctitle = proc.make_locstring(lonf,latf)


fig       = plt.figure(figsize=(16,6.5))
gs        = gridspec.GridSpec(4,4)

# --------------------------------- # Locator
ax1       = fig.add_subplot(gs[0:3,0],projection=ccrs.PlateCarree())
ax1       = viz.add_coast_grid(ax1,bbox=bboxplot,fill_color="k")

# Plot Salinity 
ax1.set_title("%s REI (%s, Year %i)" % (vnames[1],mons3[kmonth],yr+1,),fontsize=fsz_title)
plotvar   = rei_byvar[1].isel(yr=yr,mon=kmonth).mean('ens')
pcm       = ax1.contourf(plotvar.lon,plotvar.lat,plotvar,levels=cints,cmap='cmo.deep',extend='both')
cl        = ax1.contour(plotvar.lon,plotvar.lat,plotvar,levels=cints,colors='gray',linewidths=0.55)
#ax1.clabel(cl)

ax1.axhline([latf],color="k",lw=0.75)
ax1.axvline([lonf],color="k",lw=0.75)
ax1.plot(lonf,latf,color="k",marker="o",fillstyle="none",markersize=5)
fig.colorbar(pcm,ax=ax1,orientation='horizontal',fraction=0.045,pad=0.07,location='bottom')
#fig.colorbar(pcm,ax=ax1)
viz.add_ylabel(loctitle,ax=ax1,x=-.2,fontsize=fsz_axis)

# ---------------------------------
ax2       = fig.add_subplot(gs[1:3,1:])
ax2,_     = viz.init_acplot(kmonth,xtks,lags,title="",)


ax2.axhline([0.],color="k",lw=0.55,ls='solid')
ax2.axhline([rhocrit],color="k",lw=0.55,ls='dashed')
ax2.axhline([-rhocrit],color="k",lw=0.55,ls='dashed')

for vv in range(2):
    for e in range(nens):
        plotacf = acfs_byvar[vv].isel(ens=e,mons=kmonth).sel(lon=lonf,lat=latf,method='nearest')
        ax2.plot(lags,plotacf,alpha=0.1,c=vcolors[vv],label="",zorder=-1)
    
    plotacf = acfs_byvar[vv].isel(mons=kmonth).sel(lon=lonf,lat=latf,method='nearest')
    mu      = plotacf.mean('ens')
    sigma   = plotacf.std('ens')
    #plotacf = acfs_byvar[vv].isel(mons=kmonth).sel(lon=lonf,lat=latf,method='nearest').mean('ens')
    ax2.plot(lags,mu,alpha=1,c=vcolors[vv],label=vnames[vv] + "(CESM1)",zorder=1,lw=2.5)
    ax2.fill_between(lags,mu-sigma,mu+sigma,label="",alpha=0.2,zorder=-3,color=vcolors[vv])

    # Plot stochastic model
    plotacf = sm_vars[vv].squeeze().sel(lon=lonf,lat=latf,method='nearest').isel(mons=kmonth)
    ax2.plot(lags,plotacf,alpha=1,c=vcolors[vv],label=vnames[vv] + "(SM)",zorder=1,ls='dashed',lw=2.5)

ax2.legend(ncols=4)
ax2.set_ylim([-.25,1])
ax2.set_ylabel("")
ax2 = viz.label_sp("%s ACF" % (mons3[kmonth]),ax=ax2,fig=fig,labelstyle="%s",usenumber=True,y=0.2,alpha=0.2,fontsize=fsz_axis)


savename= "%sPoint_ACF_Summary_REIDX_%s.png" % (figpath,locfn)
plt.savefig(savename,bbox_inches='tight',dpi=150,transparent=True)

#%% Template for Space x Timeseries plot:

fig       = plt.figure(figsize=(16,10))
gs        = gridspec.GridSpec(3,4)

# --------------------------------- Locator/Map
ax1       = fig.add_subplot(gs[:,0],projection=ccrs.PlateCarree())
ax1       = viz.add_coast_grid(ax1,bbox=bboxplot,fill_color="k")

# --------------------------------- Timeseries
ax2       = fig.add_subplot(gs[1,1:])

#%% Remake SST/SSS Re-emergence Plots (from viz_reemergence_CMIP6)

kmonths = [11,0,1,2]
vv      = 1


fsz_title = 26


rei_in  = rei_byvar[vv].isel(mon=kmonths,).mean('mon').mean('ens') # [Year x Lat x Lon]
lon     = rei_in.lon
lat     = rei_in.lat

bbplot2 = [-80,0,15,65]
levels  = np.arange(0,0.55,0.05)
plevels = np.arange(0,0.6,0.1)

if vv == 0:
    cmapin='cmo.dense'
else:
    cmapin='cmo.deep'

fig,axs,mdict = viz.init_orthomap(1,3,bbplot2,figsize=(16,8),constrained_layout=True,centlat=45)


for yy in range(3):
    
    ax  = axs.flatten()[yy]
    blb = viz.init_blabels()
    if yy !=0:
        blb['left']=False
    else:
        blb['left']=True
    blb['lower']=True
    ax           = viz.add_coast_grid(ax,bboxplot,fill_color="lightgray",fontsize=20,blabels=blb,
                                    fix_lon=np.arange(-80,10,10),fix_lat=np.arange(0,70,10),grid_color="k")
    plotvar = rei_in.isel(yr=yy)
    
    pcm     = ax.contourf(lon,lat,plotvar,cmap=cmapin,levels=levels,transform=mdict['noProj'],extend='both',zorder=-2)
    cl      = ax.contour(lon,lat,plotvar,colors='darkslategray',linewidths=.5,linestyles='solid',levels=levels,transform=mdict['noProj'],zorder=-2)
    
    # Plot Mask
    ax.contour(icemask.lon,icemask.lat,mask_plot,colors="w",linewidths=1.5,
               transform=mdict['noProj'],levels=[0,1],zorder=-1)
    
    ax.set_title("Year %i" % (yy+1),fontsize=fsz_title)

cb = fig.colorbar(pcm,ax=axs.flatten(),fraction=0.0105,pad=0.01)
cb.ax.tick_params(labelsize=fsz_tick)
cb.set_label("%s Re-emergence Index" % vnames[vv],fontsize=fsz_axis)

savename = "%sCESM1_%s_RemIdx_DJFM_EnsAvg.png" % (figpath,vnames[vv])
plt.savefig(savename,dpi=200,bbox_inches='tight',transparent=True)

#%% Plot a region bounding box and examine the ACFs at each point (and region mean)

kmonth = 2
#bbsel     = [-65,-55,37,40] # Sargasso Sea
#bbsel = [-37,-33,44,48] # SPG Center

#bbsel = [-45,-35,20,25] # Azores Hi
#bbsel = [-35,-30,55,60] # SE Greenland
#bbsel = [-]
#bbsel  = [-45,-40,20,25]

#bbsel = [-40,-20,40,50] # NE Atlantic (Frankignoul et al. 2021)
bbsel = [-50,-20,20,30] # 
xtks  = np.arange(0,48,3)

lonstart = np.arange(-80,5,5)
latstart = np.arange(20,65,5)


vcolors_sm = ["firebrick","violet"]
# Uncomment to run for each regions
# nlo= len(lonstart)
# nla = len(latstart)

# for llo in range(nlo):
#     for lla in range(nla):
        
        
#         bbsel     = [lonstart[llo],lonstart[llo]+5,
#                      latstart[lla],latstart[lla]+5]


# --- Indent this twice to loop for boxes
bbfn,bbstr = proc.make_locstring_bbox(bbsel)

bbstr = "%i to %i $\degree$W, %i to %i $\degree$N" % (bbsel[0],bbsel[1],bbsel[2],bbsel[3])

fig       = plt.figure(figsize=(16,6.5))
gs        = gridspec.GridSpec(4,4)

# --------------------------------- # Locator
ax1       = fig.add_subplot(gs[0:3,0],projection=ccrs.PlateCarree())
ax1       = viz.add_coast_grid(ax1,bbox=bboxplot,fill_color="lightgray")

# Plot Salinity 
ax1.set_title(bbstr,fontsize=14)
plotvar   = rei_byvar[1].isel(mon=kmonths).mean('ens').mean('mon').mean('yr')
pcm       = ax1.contourf(plotvar.lon,plotvar.lat,plotvar,levels=cints,cmap='cmo.deep',extend='both',zorder=-2)
cl        = ax1.contour(plotvar.lon,plotvar.lat,plotvar,levels=cints,colors='darkslategray',linewidths=.5,zorder=-2)
#ax1.clabel(cl)

# Plot Mask
ax1.contour(icemask.lon,icemask.lat,mask_plot,colors="w",linewidths=1.5,levels=[0,1],zorder=-1)


viz.plot_box(bbsel,ax=ax1,linewidth=2.5,color="violet")

fig.colorbar(pcm,ax=ax1,orientation='horizontal',fraction=0.045,pad=0.07,location='bottom')


# ---------------------------------
ax2       = fig.add_subplot(gs[1:3,1:])
ax2,_     = viz.init_acplot(kmonth,xtks,lags,title="",)


ax2.axhline([0.],color="k",lw=1,ls='solid')
ax2.axhline([rhocrit],color="k",lw=0.55,ls='dashed')
ax2.axhline([-rhocrit],color="k",lw=0.55,ls='dashed')

for vv in range(2):
    
    
    # Plot ACF at each point (CESM)
    plotacf     = proc.sel_region_xr(acfs_byvar[vv].isel(mons=kmonth),bbsel).mean('ens')
    nlon,nlat,_ = plotacf.shape
    for a in range(nlat):
        for o in range(nlon):
            pacf = plotacf.isel(lat=a,lon=o)
            ax2.plot(lags,pacf,alpha=0.05,c=vcolors[vv],label="",zorder=-1)
            
    # Plot regional Mean
    mu      = plotacf.mean('lat').mean('lon')
    ax2.plot(lags,mu,alpha=1,c=vcolors[vv],label=vnames[vv] + "(CESM1)",zorder=1,lw=2.5)
    
    # -----------------------------------------
    
    # Plot ACF at each point (Stochastic Model)
    plotacf     = proc.sel_region_xr(sm_vars[vv].isel(mons=kmonth),bbsel).squeeze()
    
    for a in range(nlat):
        for o in range(nlon):
            pacf = plotacf.isel(lat=a,lon=o)
            ax2.plot(lags,pacf,alpha=0.05,c=vcolors_sm[vv],label="",zorder=-1,ls='dashed')
            
    # Plot regional Mean
    mu      = plotacf.mean('lat').mean('lon')
    ax2.plot(lags,mu,alpha=1,c=vcolors_sm[vv],label=vnames[vv] + "(SM)",zorder=1,lw=2.5,ls='dashed')
    
    # for e in range(nens):
    #     plotacf = acfs_byvar[vv].isel(ens=e,mons=kmonth).sel(lon=lonf,lat=latf,method='nearest')
    #     ax2.plot(lags,plotacf,alpha=0.1,c=vcolors[vv],label="",zorder=-1)
    
    
    #sigma   = plotacf.mean('lat').mean('lon')
    #plotacf = acfs_byvar[vv].isel(mons=kmonth).sel(lon=lonf,lat=latf,method='nearest').mean('ens')
    
    #ax2.fill_between(lags,mu-sigma,mu+sigma,label="",alpha=0.2,zorder=-3,color=vcolors[vv])

    # Plot stochastic model
    #plotacf = sm_vars[vv].squeeze().sel(lon=lonf,lat=latf,method='nearest').isel(mons=kmonth)
    #ax2.plot(lags,plotacf,alpha=1,c=vcolors[vv],label=vnames[vv] + "(SM)",zorder=1,ls='dashed',lw=2.5)

ax2.legend(ncols=4,fontsize=12,loc='upper right')
ax2.set_ylim([-.25,1])
ax2.set_ylabel("")
ax2 = viz.label_sp("%s ACF" % (mons3[kmonth]),ax=ax2,fig=fig,labelstyle="%s",usenumber=True,y=0.2,alpha=0.2,fontsize=fsz_axis)

savename= "%sPoint_ACF_Summary_REIDX_%s_mon%02i.png" % (figpath,bbfn,kmonth+1)
plt.savefig(savename,bbox_inches='tight',dpi=150,transparent=True)

# -----------------------------------------------------------------------------
#%% Loop of above to shift box around, and grab values
# -----------------------------------------------------------------------------

kmonth = 6
xtks   = np.arange(0,48,3)

lonstart   = np.arange(-80,0,5)
latstart   = np.arange(20,65,5)


vcolors_sm = ["firebrick","violet"]
#Uncomment to run for each regions
nlo= len(lonstart)
nla = len(latstart)
frame = 0
for lla in range(nla):
        for llo in range(nlo):
            
            bbsel     = [lonstart[llo],lonstart[llo]+5,
                          latstart[lla],latstart[lla]+5]
    
            # --- Indent this twice to loop for boxes
            bbfn,bbstr = proc.make_locstring_bbox(bbsel)
            
            bbstr = "%i to %i $\degree$W, %i to %i $\degree$N" % (bbsel[0],bbsel[1],bbsel[2],bbsel[3])
            
            fig       = plt.figure(figsize=(16,6.5))
            gs        = gridspec.GridSpec(4,4)
            
            # --------------------------------- # Locator
            ax1       = fig.add_subplot(gs[0:3,0],projection=ccrs.PlateCarree())
            ax1       = viz.add_coast_grid(ax1,bbox=bboxplot,fill_color="lightgray")
            
            # Plot Salinity 
            ax1.set_title(bbstr,fontsize=14)
            plotvar   = rei_byvar[1].isel(mon=kmonths).mean('ens').mean('mon').mean('yr')
            pcm       = ax1.contourf(plotvar.lon,plotvar.lat,plotvar,levels=cints,cmap='cmo.deep',extend='both',zorder=-2)
            cl        = ax1.contour(plotvar.lon,plotvar.lat,plotvar,levels=cints,colors='darkslategray',linewidths=.5,zorder=-2)
            #ax1.clabel(cl)
            
            # Plot Mask
            ax1.contour(icemask.lon,icemask.lat,mask_plot,colors="w",linewidths=1.5,levels=[0,1],zorder=-1)
            
            
            viz.plot_box(bbsel,ax=ax1,linewidth=2.5,color="violet")
            
            fig.colorbar(pcm,ax=ax1,orientation='horizontal',fraction=0.045,pad=0.07,location='bottom')
            
            
            # ---------------------------------
            ax2       = fig.add_subplot(gs[1:3,1:])
            ax2,_     = viz.init_acplot(kmonth,xtks,lags,title="",)
            
            
            ax2.axhline([0.],color="k",lw=1,ls='solid')
            ax2.axhline([rhocrit],color="k",lw=0.55,ls='dashed')
            ax2.axhline([-rhocrit],color="k",lw=0.55,ls='dashed')
            
            for vv in range(2):
                
                
                # Plot ACF at each point (CESM)
                plotacf     = proc.sel_region_xr(acfs_byvar[vv].isel(mons=kmonth),bbsel).mean('ens')
                nlon,nlat,_ = plotacf.shape
                for a in range(nlat):
                    for o in range(nlon):
                        pacf = plotacf.isel(lat=a,lon=o)
                        ax2.plot(lags,pacf,alpha=0.05,c=vcolors[vv],label="",zorder=-1)
                        
                # Plot regional Mean
                mu      = plotacf.mean('lat').mean('lon')
                ax2.plot(lags,mu,alpha=1,c=vcolors[vv],label=vnames[vv] + "(CESM1)",zorder=1,lw=2.5)
                
                # -----------------------------------------
                
                # Plot ACF at each point (Stochastic Model)
                plotacf     = proc.sel_region_xr(sm_vars[vv].isel(mons=kmonth),bbsel).squeeze()
                
                for a in range(nlat):
                    for o in range(nlon):
                        pacf = plotacf.isel(lat=a,lon=o)
                        ax2.plot(lags,pacf,alpha=0.05,c=vcolors_sm[vv],label="",zorder=-1,ls='dashed')
                        
                # Plot regional Mean
                mu      = plotacf.mean('lat').mean('lon')
                ax2.plot(lags,mu,alpha=1,c=vcolors_sm[vv],label=vnames[vv] + "(SM)",zorder=1,lw=2.5,ls='dashed')
                
                # for e in range(nens):
                #     plotacf = acfs_byvar[vv].isel(ens=e,mons=kmonth).sel(lon=lonf,lat=latf,method='nearest')
                #     ax2.plot(lags,plotacf,alpha=0.1,c=vcolors[vv],label="",zorder=-1)
                
                
                #sigma   = plotacf.mean('lat').mean('lon')
                #plotacf = acfs_byvar[vv].isel(mons=kmonth).sel(lon=lonf,lat=latf,method='nearest').mean('ens')
                
                #ax2.fill_between(lags,mu-sigma,mu+sigma,label="",alpha=0.2,zorder=-3,color=vcolors[vv])
            
                # Plot stochastic model
                #plotacf = sm_vars[vv].squeeze().sel(lon=lonf,lat=latf,method='nearest').isel(mons=kmonth)
                #ax2.plot(lags,plotacf,alpha=1,c=vcolors[vv],label=vnames[vv] + "(SM)",zorder=1,ls='dashed',lw=2.5)
            
            ax2.legend(ncols=4,fontsize=12,loc='upper right')
            ax2.set_ylim([-.25,1])
            ax2.set_ylabel("")
            ax2 = viz.label_sp("%s ACF" % (mons3[kmonth]),ax=ax2,fig=fig,labelstyle="%s",usenumber=True,y=0.2,alpha=0.2,fontsize=fsz_axis)
            
            
            
            savename= "%sPoint_ACF_Summary_REIDX_frame%03i_%s_mon%02i.png" % (figpath,frame,bbfn,kmonth+1)
            plt.savefig(savename,bbox_inches='tight',dpi=150,transparent=True)
            
            frame += 1

# -----------------------------------------------------------------------------
#%% Compute Pointwise MSE (first calculate differences)
# -----------------------------------------------------------------------------

diffs_byvar = []
for vv in range(2):
    smvals        = sm_vars[vv].squeeze() # (65, 48, 12, 61)
    cmvals        = acfs_byvar[vv].mean('ens') # (89, 96, 12, 61)
    smvals,cmvals = proc.resize_ds([smvals,cmvals])
    
    diffs_sq = ((smvals-cmvals)**2)
    diffs_byvar.append(diffs_sq) # (89, 96, 12, 61)


#%% Plot Pointwise MSE

proj = ccrs.PlateCarree()
vlms = [0,.02]

sellags = [0,61]
kmonth  = 6

fig,axs,_=viz.init_orthomap(1,2,bbplot2,figsize=(12,8),constrained_layout=True,centlat=45)


for vv in range(2):
    
    
    
    ax  = axs[vv]
    mse = diffs_byvar[vv].sel(lags=slice(sellags[0],sellags[1])).mean('lags').isel(mons=kmonth) 
    
    
    ax           = viz.add_coast_grid(ax,bboxplot,fill_color="lightgray",fontsize=20,blabels=blb,
                                    fix_lon=np.arange(-80,10,10),fix_lat=np.arange(0,70,10),grid_color="k")
    
    ax.contour(icemask.lon,icemask.lat,mask_plot,colors="w",linewidths=1.5,
               transform=mdict['noProj'],levels=[0,1],zorder=-1)
    
    ax.set_title("%s" % (vnames[vv]),fontsize=fsz_title)
    
    pcm = ax.pcolormesh(mse.lon,mse.lat,mse.T,transform=proj,cmap="inferno",
                        vmin=vlms[0],vmax=vlms[1])
    
    viz.hcbar(pcm,ax=ax)
    #fig.colorbar(pcm,ax=ax)
    
plt.suptitle("Lags %i to %i, Month %i" % (sellags[0],sellags[1],kmonth+1),y=0.75,fontsize=24)
savename = "%sMSE_SM-CESM_SumLags_%02i_%02i_mon%02i.png" % (figpath,sellags[0],sellags[1],kmonth+1)
plt.savefig(savename,dpi=150)
    
# for vv in range(2):
#     smvals        = sm_vars[vv].squeeze() # (65, 48, 12, 61)
#     cmvals        = acfs_byvar[vv].mean('ens') # (89, 96, 12, 61)
#     smvals,cmvals = proc.resize_ds([smvals,cmvals])
    
#     mse = ((smvals-cmvals)**2).mean('lags')
    
    
    
    
    
    #mse = sm_vars[vv] - 
#%% Just Plot the REIDX patterns with BSF

setname = "SSSCSU"
rrsel = ["SAR","NAC"]

bsf_kmonth = bsf.BSF.isel(mon=kmonths).mean('mon')
bsflvl     = np.arange(-100,110,10)
plot_bsf   = False

proj      = ccrs.PlateCarree()
vlms      = [0,.02]

sellags   = [0,61]
kmonth    = 6



fig,axs,_ = viz.init_orthomap(1,2,bbplot2,figsize=(12,8),constrained_layout=True,centlat=45)

for vv in range(2):
    
    
    plotvar = rei_byvar[vv].mean('yr').mean('ens').isel(mon=kmonths).mean('mon')
    
    ax = axs[vv]
    #mse = diffs_byvar[vv].sel(lags=slice(sellags[0],sellags[1])).mean('lags').isel(mons=kmonth) 
    
    ax = viz.add_coast_grid(ax,bboxplot,fill_color="lightgray",fontsize=20,blabels=blb,
                                    fix_lon=np.arange(-80,10,10),fix_lat=np.arange(0,70,10),grid_color="k")
    
    ax.contour(icemask.lon,icemask.lat,mask_plot,colors="w",linewidths=1.5,
               transform=mdict['noProj'],levels=[0,1],zorder=-1)
    
    ax.set_title("%s" % (vnames[vv]),fontsize=fsz_title)
    
    if vv == 0:
        cmapin='cmo.dense'
    else:
        cmapin='cmo.deep'
        
    pcm     = ax.contourf(lon,lat,plotvar,cmap=cmapin,levels=levels,transform=mdict['noProj'],extend='both',zorder=-2)
    cl      = ax.contour(lon,lat,plotvar,colors='darkslategray',linewidths=.5,linestyles='solid',levels=levels,transform=mdict['noProj'],zorder=-2)
    
    if plot_bsf:
        clb = ax.contour(bsf_kmonth.lon,bsf_kmonth.lat,bsf_kmonth,levels=bsflvl,
                         colors='navy',linewidths=1.75,transform=proj,alpha=0.66)
    
    
    cb = viz.hcbar(pcm,ax=ax)
    cb.set_label("Re-emergence Index",fontsize=16)
    
    # Plot the Bounding Boxes
    rdict = rparams.region_sets[setname]
    nreg  = len(rdict['bboxes'])
    for rr in range(nreg):
        bbp    = rdict['bboxes'][rr]
        bbname = rdict['regions'][rr]
        if bbname not in rrsel:
            continue
        
        viz.plot_box(bbp,color=rdict['rcols'][rr],linewidth=2.5,proj=proj,ax=ax)
    
    
#plt.suptitle("Lags %i to %i, Month %i" % (sellags[0],sellags[1],kmonth+1),y=0.75,fontsize=24)
savename = "%sMSE_SM-CESM_REI_Locator_bsf%i.png" % (figpath,plot_bsf)
plt.savefig(savename,dpi=150,bbox_inches='tight')

#%% Pointwise RMSE (Do multi figure over several lags)
# Also take mean across all months

setname    = "SSSCSU"
rrsel      = ["SAR","NAC"]

bbplot2    =  [-80, 0, 20, 65] # [-80, 0, 15, 65]

bsf_kmonth = bsf.BSF.isel(mon=kmonths).mean('mon')
bsflvl     = np.arange(-50,60,10)
plot_bsf   = True

proj         = ccrs.PlateCarree()
plot_contour = False

#sellags  = [0,61]
maxlags   = [6,12,36]
fig,axs,_ = viz.init_orthomap(2,3,bbplot2,figsize=(18,9.5),constrained_layout=True,centlat=45)

for vv in range(2):
    # Set colormap
    if vv == 0:
        cmapin = 'inferno'
        levels = np.arange(0,0.2,0.02)
    else:
        cmapin ='pink'
        levels = np.arange(0,1.0,0.2)
    for ll in range(3):
        
        # Select Lags/Months and take mean
        sellags = [0,maxlags[ll]]
        mse     = diffs_byvar[vv].sel(lags=slice(sellags[0],sellags[1])).mean('lags').mean('mons') 
        plotvar = mse.T
        
        # Draw Coastlines
        ax = axs[vv,ll]
        ax = viz.add_coast_grid(ax,bboxplot,fill_color="lightgray",fontsize=20,blabels=blb,
                                        fix_lon=np.arange(-80,10,10),fix_lat=np.arange(0,70,10),grid_color="k")
        
        # Labels and Colorbars
        if vv == 0:
            ax.set_title("Max Lag %i" % (sellags[-1]),fontsize=fsz_title)
        if ll == 0:
            _=viz.add_ylabel(vnames[vv],ax=ax,fontsize=fsz_title,x=0.01,y=0.60)
        if ll == 2:
            cb = fig.colorbar(pcm,ax=axs[vv,:].flatten(),fraction=0.010,pad=0.01)
            cb.set_label("RMSE (Correlation)",fontsize=fsz_axis)
        # Add Ice Mask
        ax.contour(icemask.lon,icemask.lat,mask_plot,colors="w",linewidths=1.5,
                   transform=mdict['noProj'],levels=[0,1],zorder=-1)
        
        # Draw REI Map
        if plot_contour:
            pcm     = ax.contourf(plotvar.lon,plotvar.lat,plotvar,cmap=cmapin,levels=levels,transform=mdict['noProj'],extend='both',zorder=-2)
            cl      = ax.contour(plotvar.lon,plotvar.lat,plotvar,colors='darkslategray',linewidths=.5,linestyles='solid',levels=levels,transform=mdict['noProj'],zorder=-2)
        else:
            pcm     = ax.pcolormesh(plotvar.lon,plotvar.lat,plotvar,cmap=cmapin,vmin=levels[0],vmax=levels[1],transform=mdict['noProj'],zorder=-2)
            
        
        if plot_bsf:
            clb = ax.contour(bsf_kmonth.lon,bsf_kmonth.lat,bsf_kmonth,levels=bsflvl,
                             colors='k',linewidths=1.75,transform=proj,alpha=0.66)

savename = "%sRMSE_SM-CESM_MeanMonth_MeanLag_bsf%i_contour%i.png" % (figpath,plot_bsf,plot_contour)
plt.savefig(savename,dpi=150,bbox_inches='tight')

#%% Examine Lag Differences Over Sections of the ACF

vv            = 0
lag_ranges    = [[0,6],[6,18],[18,30],[30,61]]
lagrangenames = ["Initial Decorr.","REM Y1","REM Y2",">Y2"]

nrngs         = len(lag_ranges)

#acfs_byvar_in = []

lagdiffs_byvar = []
for vv in range(2):
    diffs_bylr = []
    for ll in range(nrngs):
        
        # Take Lag Range
        lr       = lag_ranges[ll]
        cesm_acf = acfs_byvar[vv].sel(lags=slice(lr[0],lr[1])).mean('ens') # (42, 89, 96, 12, 7)
        
        sm_acf   = sm_vars[vv].sel(lags=slice(lr[0],lr[1])).squeeze() # (lon: 65, lat: 48, mons: 12, lags: 7)>
        
        cesm_acf,sm_acf = proc.resize_ds([cesm_acf,sm_acf])
        
        diffs_lag_range = (sm_acf - cesm_acf).sum('lags')
        
        diffs_bylr.append(diffs_lag_range)
    
    diffs_bylr = xr.concat(diffs_bylr,dim='lag_range')
    
    diffs_bylr['lag_range']=lagrangenames
    
    lagdiffs_byvar.append(diffs_bylr.copy())

#%% Visualize summed correlatino differences across lag range

kmonths = [11,0,1,2]
vv      = 1


fsz_title   = 26

rei_in      = lagdiffs_byvar[vv].isel(mons=kmonths,).mean('mons') # [Year x Lat x Lon]
lon         = rei_in.lon
lat         = rei_in.lat

bbplot2 = [-80,0,20,65]
if vv == 0:
    levels  = np.arange(-5,5.5,.5)#np.arange(0,0.55,0.05)
else:
    levels  = np.arange(-20,22,2)#np.arange(0,0.55,0.05)
plevels = np.arange(0,0.6,0.1)

cmapin  = 'cmo.balance'

fig,axs,mdict = viz.init_orthomap(1,4,bbplot2,figsize=(18,12),constrained_layout=True,centlat=45)


for yy in range(4):
    
    ax  = axs.flatten()[yy]
    blb = viz.init_blabels()
    if yy !=0:
        blb['left']=False
    else:
        blb['left']=True
    blb['lower']=True
    ax           = viz.add_coast_grid(ax,bboxplot,fill_color="lightgray",fontsize=20,blabels=blb,
                                    fix_lon=np.arange(-80,10,10),fix_lat=np.arange(0,70,10),grid_color="k")
    plotvar = rei_in.isel(lag_range=yy).T
    
    pcm     = ax.contourf(lon,lat,plotvar,cmap=cmapin,levels=levels,transform=mdict['noProj'],extend='both',zorder=-2)
    cl      = ax.contour(lon,lat,plotvar,colors='darkslategray',linewidths=.5,linestyles='solid',levels=levels,transform=mdict['noProj'],zorder=-2)
    
    # Plot Mask
    ax.contour(icemask.lon,icemask.lat,mask_plot,colors="k",linewidths=1.5,
               transform=mdict['noProj'],levels=[0,1],zorder=-1)
    
    ax.set_title(lagrangenames[yy],fontsize=fsz_title-2)

cb = fig.colorbar(pcm,ax=axs.flatten(),fraction=0.0105,pad=0.01)
cb.ax.tick_params(labelsize=fsz_tick)
cb.set_label("Diff. in Corr. (%s, SM-CESM)" % vnames[vv],fontsize=fsz_axis-2)

savename = "%sCESM1_%s_LagRngDiff_DJFM_EnsAvg.png" % (figpath,vnames[vv])
plt.savefig(savename,dpi=200,bbox_inches='tight',transparent=True)
 

    