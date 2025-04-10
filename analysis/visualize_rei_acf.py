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

#compare_name = "CESM1LE"

# Indicate files containing ACFs
cesm_name   = "CESM1_1920to2005_%sACF_lag00to60_ALL_ensALL.nc"
vnames      =  ["SST","SSS"] #["SST","SSS","TEMP"]
#vnames = ["acf","acf"]

sst_expname = "SM_SST_Revision_Qek_TauReg_AutoCorr_RevisionD1_lag00to60_ALL_ensALL.nc"
sss_expname = "SM_SSS_Revision_Qek_TauReg_AutoCorr_RevisionD1_lag00to60_ALL_ensALL.nc"
compare_name = "RevisionD1"

# Jclim first submission
#compare_name = PaperDraft01
#sst_expname = "SM_SST_Draft03_Rerun_QekCorr_SST_autocorrelation_thresALL_lag00to60.nc"#"SM_SST_Draft01_Rerun_QekCorr_SST_autocorrelation_thresALL_lag00to60.nc"#"SM_SST_EOF_LbddCorr_Rerun_SST_autocorrelation_thresALL_lag00to60.nc"
#sss_expname = "SM_SSS_Draft03_Rerun_QekCorr_SSS_autocorrelation_thresALL_lag00to60.nc"#"SM_SSS_Draft01_Rerun_QekCorr_SSS_autocorrelation_thresALL_lag00to60.nc"#"SM_SSS_EOF_LbddCorr_Rerun_lbdE_neg_SSS_autocorrelation_thresALL_lag00to60.nc"

#sst_expname = "SM_SST_EOF_LbddCorr_Rerun_SST_autocorrelation_thresALL_lag00to60.nc"
#sss_expname = "SM_SSS_EOF_LbddCorr_Rerun_lbdE_neg_SSS_autocorrelation_thresALL_lag00to60.nc"

# Load Region Information
regionset       = "SSSCSU"
regiondicts     = rparams.region_sets[regionset]
bboxes          = regiondicts['bboxes']
regions_long    = regiondicts['regions_long']
rcols           = regiondicts['rcols']
rsty            = regiondicts['rsty']
regplot         = [0,1,3]
nregs           = len(regplot)

# Load Point Information
pointset        = "PaperDraft02"
ptdict          = rparams.point_sets[pointset]
ptcoords        = ptdict['bboxes']
ptnames         = ptdict['regions']
ptnames_long    = ptdict['regions_long']
ptcols          = ptdict['rcols']
ptsty           = ptdict['rsty']

#%% Load ACFs and REI

acfs_byvar  = []
rei_byvar   = []
maxmin_byvar = []
for vv in range(2):
    ds = xr.open_dataset(procpath + cesm_name % vnames[vv]).acf.squeeze()
    acfs_byvar.append(ds)
    
    dsrei = xr.open_dataset("%s%s_CESM/Metrics/REI_Pointwise.nc" % (output_path,vnames[vv])).rei.load()
    rei_byvar.append(dsrei)
    
    dsmaxmin = xr.open_dataset("%s%s_CESM/Metrics/MaxMin_Pointwise.nc" % (output_path,vnames[vv])).corr.load()
    maxmin_byvar.append(dsmaxmin)
    
#%% Add ACFs from stochastic model

try:
    sm_sss   = xr.open_dataset(procpath+sss_expname).SSS.load()        # (lon: 65, lat: 48, mons: 12, thres: 1, lags: 61)
    sm_sst   = xr.open_dataset(procpath+sst_expname).SST.load()
except:
    sm_sss   = xr.open_dataset(procpath+sss_expname).acf.load()
    sm_sst   = xr.open_dataset(procpath+sst_expname).acf.load()

sm_vars  = [sm_sst,sm_sss]

#%% Load mixed layer depth

ds_h    = xr.open_dataset(input_path + "mld/CESM1_HTR_FULL_HMXL_NAtl.nc").h.load()
#id_hmax = np.argmax(ds_h.mean(0),0)

#%% Load BSF and Ice Mask (copied from compare_detrainment_damping)

bsf        = dl.load_bsf()

# Load Land Ice Mask
icemask    = xr.open_dataset(input_path + "masks/CESM1LE_HTR_limask_pacificmask_enssum_lon-90to20_lat0to90.nc")

# Resize
#bsf,icemask,_    = proc.resize_ds([bsf,icemask,acfs_in_rsz[0]])
bsf_savg   = proc.calc_savg_mon(bsf)

# Load Ice Mask
mask       = icemask.MASK.squeeze()
mask_plot  = xr.where(np.isnan(mask),0,mask)#mask.copy()

mask_apply = icemask.MASK.squeeze().values
#mask_plot[np.isnan(mask)] = 0

# mask_reg_sub    = proc.sel_region_xr(mask,bboxplot)
# mask_reg_ori    = xr.ones_like(mask) * 0
# mask_reg        = mask_reg_ori + mask_reg_sub



# Load Gulf Stream
ds_gs      = dl.load_gs()
ds_gs      = ds_gs.sel(lon=slice(-90,-50))
ds_gs2     = dl.load_gs(load_u2=True)


#%% Indicate Plotting Parameters (taken from visualize_rem_cmip6)


bboxplot                    = [-80,0,10,65]
mpl.rcParams['font.family'] = 'Avenir'
mons3                       = proc.get_monstr(nletters=3)
fsz_tick                    = 18
fsz_axis                    = 20
fsz_title                   = 16

rhocrit = proc.ttest_rho(0.05,2,86)

proj= ccrs.PlateCarree()



#%% Load the Diff by lag which was calculated below
selmon = [1,2]

rpath   = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/proc/"

load_mean = True
if load_mean== True:
    loadname = "Mean"
else:
    loadname = "Diff"
fns     = [
         "CESM1_vs_SM_%s_SST_LagRng%s_DJFM_EnsAvg.nc" % (compare_name,loadname),
         "CESM1_vs_SM_%s_SSS_LagRng%s_DJFM_EnsAvg.nc" % (compare_name,loadname)
         ]
vnames = ["SST","SSS"]

ds_diff = []
ds_sum = []
for vv in range(2):
    ds = xr.open_dataset(rpath + fns[vv])[vnames[vv]].load()
    ds_diff.append(ds)
    ds_sum.append(ds.sum('lag_range').isel(mons=selmon).mean('mons'))
    
ds_diff = xr.merge(ds_diff)
ds_sum  = xr.merge(ds_sum)


#%% Identify a location


# plotvar_bg      = ds_sum.SSS.T
# plotvar_bg_name = "SSSdiffbylag"
# cints_bg        = np.arange(-40,44,4)
# cmap_bg         = 'cmo.balance'

plotvar_bg      = ds_sum.SST.T
plotvar_bg_name = "SSTdiffbylag"
cints_bg        = np.arange(-20,22,2)
cmap_bg         = 'cmo.balance'


incl_ens  = False # True to plot individual ensemble members

lags      = acfs_byvar[1].lags
xtks      = np.arange(0,37,3)

vcolors   = ["hotpink","navy"]

# Indicate Selected point and base month
kmonth    = 2
lonf      = -70 #-54#-30
latf      = 30  #59#50---
yr        = 0
nens      = 42

locfn,loctitle = proc.make_locstring(lonf,latf)


fig       = plt.figure(figsize=(16,6.5))
gs        = gridspec.GridSpec(4,4)

# --------------------------------- # Locator
ax1       = fig.add_subplot(gs[0:3,0],projection=ccrs.PlateCarree())
ax1       = viz.add_coast_grid(ax1,bbox=bboxplot,fill_color="k")

# Plot Salinity 
if plotvar_bg is None:
    plotvar   = rei_byvar[1].isel(yr=yr,mon=kmonth).mean('ens')
    title     = "%s REI (%s, Year %i)" % (vnames[1],mons3[kmonth],yr+1,)
    cints     = np.arange(0,0.55,0.05)
    cmap      = "cmo.deep"
else:
    plotvar   = plotvar_bg
    title     = plotvar_bg_name
    cints     = cints_bg
    cmap      = cmap_bg
    
ax1.set_title(title,fontsize=fsz_title)
pcm       = ax1.contourf(plotvar.lon,plotvar.lat,plotvar,levels=cints,cmap=cmap,extend='both')
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
    if incl_ens:
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


savename= "%sPoint_ACF_Summary_REIDX_%s_%s_inclens%i.png" % (figpath,locfn,plotvar_bg_name,incl_ens)
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

kmonths = [1,2]
vv      = 0


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
    
    # Plot Gulf Stream Position
    ax.plot(ds_gs.lon,ds_gs.lat.mean('ens'),transform=proj,lw=.75,c="k")
    

cb = fig.colorbar(pcm,ax=axs.flatten(),fraction=0.0105,pad=0.01)
cb.ax.tick_params(labelsize=fsz_tick)
cb.set_label("%s Re-emergence Index" % vnames[vv],fontsize=fsz_axis)

savename = "%sCESM1_%s_RemIdx_DJFM_EnsAvg.png" % (figpath,vnames[vv])
plt.savefig(savename,dpi=200,bbox_inches='tight',transparent=True)

# <o><o><o><o><o><o><o><o><o><o><o><o><o><o>
#%% Paper Outline Version of <kmonth> REI Index from Above!
# <o><o><o><o><o><o><o><o><o><o><o><o><o><o>

plot_bbox = True
kmonths   = [1,2]

fig,axs,_ = viz.init_orthomap(2,2,bboxplot,figsize=(20,14.5))

ii        = 0

for vv in range(2):
    
    if vv == 0:
        cmapin='cmo.dense'
    else:
        cmapin='cmo.deep'
    
    for yy in range(2):
        
        # Select Axis
        ax  = axs[vv,yy]
        
        # Set Labels
        blb = viz.init_blabels()
        if yy !=0:
            blb['left']=False
        else:
            blb['left']=True
            viz.add_ylabel(vnames[vv],ax=ax,rotation='horizontal',fontsize=fsz_title)
        if vv == 1:
            blb['lower'] =True
        else:
            ax.set_title("Year %i" % (yy+1),fontsize=fsz_title)
        ax           = viz.add_coast_grid(ax,bboxplot,fill_color="lightgray",fontsize=20,blabels=blb,
                                        fix_lon=np.arange(-80,10,10),fix_lat=np.arange(0,70,10),grid_color="k")
        
        # Prepare variable for plotting
        rei_in  = rei_byvar[vv].isel(mon=kmonths,).mean('mon').mean('ens') #* mask_reg # [Year x Lat x Lon]
        lon     = rei_in.lon
        lat     = rei_in.lat
        plotvar = rei_in.isel(yr=yy) #* mask_reg
        
        # Add contours
        pcm     = ax.contourf(lon,lat,plotvar,cmap=cmapin,levels=levels,transform=mdict['noProj'],extend='both',zorder=-2)
        cl      = ax.contour(lon,lat,plotvar,colors='darkslategray',linewidths=.5,linestyles='solid',levels=levels,transform=mdict['noProj'],zorder=-2)
        ax.clabel(cl,fontsize=fsz_tick,inline_spacing=2)
        
        # Plot Mask
        ax.contour(icemask.lon,icemask.lat,mask_plot,colors="cyan",linewidths=1.5,
                   transform=mdict['noProj'],levels=[0,1],zorder=-1)
        
        ax.set_title("Year %i" % (yy+1),fontsize=fsz_title)
        
        
        # Add Subplot labels
        viz.label_sp(ii,ax=ax,fontsize=fsz_title,alpha=0.75)
        ii+=1
        
        # Plot Bounding Boxes
        if plot_bbox:
            
            if (yy == 0) and (vnames[vv] == "SST"): #Plot Irminger Sea
                rr = 3
                rbbx = bboxes[rr]
                viz.plot_box(rbbx,ax=ax,color=rcols[rr],linestyle=rsty[rr],leglab=regions_long[rr],linewidth=2.5,return_line=True)
                
        
            
            if (yy == 0) and (vnames[vv] == "SSS"): #Plot Sargasso Sea
                rr = 0
                rbbx = bboxes[rr]
                viz.plot_box(rbbx,ax=ax,color=rcols[rr],linestyle=rsty[rr],leglab=regions_long[rr],linewidth=2.5,return_line=True)
            
                
            if (yy == 0) and (vnames[vv] == 'SSS'): #Plot North Atlantic Current
                rr = 1
                rbbx = bboxes[rr]
                viz.plot_box(rbbx,ax=ax,color=rcols[rr],linestyle=rsty[rr],leglab=regions_long[rr],linewidth=2.5,return_line=True)
                
        
        # Plot Gulf Stream Position
        ax.plot(ds_gs.lon,ds_gs.lat.mean('ens'),transform=proj,lw=.75,c="k")

    
    cb = fig.colorbar(pcm,ax=axs[vv,:].flatten(),fraction=0.0155,pad=0.03)
    cb.ax.tick_params(labelsize=fsz_tick)
    cb.set_label("%s Re-emergence Index" % vnames[vv],fontsize=fsz_axis)

savename = "%sCESM1_%s_RemIdx_DJFM_EnsAvg_PaperOutline.png" % (figpath,vnames[vv])
plt.savefig(savename,dpi=200,bbox_inches='tight',transparent=True)

#%% Examine max/min correlation values

kmonths       = [1,2]
vv            = 0
maxminid      = 0
zoom          = False

if zoom:
    bbplot_in = [-70,-20,55,65]
    centlon   = -45
    centlat   = 60
else:
    bbplot_in = bbplot2
    centlon   = -40
    centlat   = 45

if maxminid ==0:
    cblab = "Summertime Minima (Corr)"
    levels  = np.arange(0,1.05,0.05)
else:
    cblab = "Wintertime Maxima (Corr)"
    levels  = np.arange(0,1.05,0.05)

rei_levels  = np.arange(0.15,1.05,0.05)
fsz_title   = 26

maxmin_in   = maxmin_byvar[vv].isel(mon=kmonths,maxmin=maxminid).mean('mon').mean('ens') # [Year x Lat x Lon]
rei_in      = rei_byvar[vv].isel(mon=kmonths,).mean('mon').mean('ens') # [Year x Lat x Lon]

lon         = maxmin_in.lon
lat         = maxmin_in.lat
bbplot2     = [-80,0,15,65]

plevels     = np.arange(0,0.6,0.1)

if vv == 0:
    cmapin='cmo.dense'
else:
    cmapin='cmo.deep'

fig,axs,mdict = viz.init_orthomap(1,3,bbplot_in,
                                  figsize=(16,8),constrained_layout=True,
                                  centlat=centlat,centlon=centlon)


for yy in range(3):
    
    ax  = axs.flatten()[yy]
    blb = viz.init_blabels()
    if yy !=0:
        blb['left']=False
    else:
        blb['left']=True
    blb['lower']=True
    ax           = viz.add_coast_grid(ax,bbplot_in,fill_color="lightgray",fontsize=20,blabels=blb,
                                    fix_lon=np.arange(-80,10,10),fix_lat=np.arange(0,70,10),grid_color="k")
    plotvar = maxmin_in.isel(yr=yy)
    
    pcm     = ax.contourf(lon,lat,plotvar,cmap=cmapin,levels=levels,transform=mdict['noProj'],extend='both',zorder=-2)
    cl      = ax.contour(lon,lat,plotvar,colors='darkslategray',linewidths=.5,linestyles='solid',levels=levels,transform=mdict['noProj'],zorder=-2)
    
    
    
    # Plot Mask
    ax.contour(icemask.lon,icemask.lat,mask_plot,colors="w",linewidths=1,
               transform=mdict['noProj'],levels=[0,1],zorder=-1,linestyles='dotted')
    
    ax.set_title("Year %i" % (yy+1),fontsize=fsz_title)
    
    # Plot Gulf Stream Position
    ax.plot(ds_gs.lon,ds_gs.lat.mean('ens'),transform=proj,lw=.75,c="k")
    
    
    # Plot REI Contours
    if vv == 1:
        if maxminid == 0:
            rei_contcol = "hotpink"
        else:
            rei_contcol = "cyan"
    elif vv == 0:
        if maxminid == 0:
            rei_contcol = "firebrick"
        else:
            rei_contcol = "hotpink"
        
    plotvar = rei_in.isel(yr=yy)
    cl      = ax.contour(plotvar.lon,plotvar.lat,plotvar,
                         colors=rei_contcol,linewidths=0.3,alpha=1,
                         linestyles='solid',levels=rei_levels,transform=mdict['noProj'],zorder=-2)
        
    
    

cb = fig.colorbar(pcm,ax=axs.flatten(),fraction=0.0105,pad=0.01)
cb.ax.tick_params(labelsize=fsz_tick)
cb.set_label("%s %s" % (vnames[vv],cblab),fontsize=fsz_axis)

savename = "%sCESM1_%s_RemIdx_DJFM_EnsAvg_max%0i.png" % (figpath,vnames[vv],maxminid)
if zoom:
    savename = proc.addstrtoext(savename,"_zoom",adjust=0)
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
    
    if len(mse.shape) > 2:
        mse = mse.mean('ens') # Additional Ens Dimension for those calcualted with calc_crosscorr
        
    
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

setname     = "SSSCSU"
rrsel       = ["SAR","NAC"]

bsf_kmonth  = bsf.BSF.isel(mon=kmonths).mean('mon')
bsflvl      = np.arange(-100,110,10)
plot_bsf    = False

proj        = ccrs.PlateCarree()
vlms        = [0,.02]

sellags     = [0,61]
kmonth      = 6

fig,axs,_   = viz.init_orthomap(1,2,bbplot2,figsize=(12,8),constrained_layout=True,centlat=45)

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
        
        if len(mse.shape) > 2:
            mse = mse.mean('ens') # Additional Ens Dimension for those calcualted with calc_crosscorr
            
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
#lag_ranges    = [[0,6],[6,18],[18,30],[30,61]]
#lagrangenames = ["Initial Decorr.","REM Y1","REM Y2",">Y2"]

lag_ranges    = [[1,6],[6,18],[18,60]]
lagrangenames = ["Initial Decorrelation\n(Lags 1-6)","Year 1 Re-emergence\n(Lags 6-18)","Long-Term Persistence\n(Lags 18 to 60)"]


lagstrs        = ["%i to %i" % (lr[0],lr[1]) for lr in lag_ranges]
nrngs         = len(lag_ranges)

#acfs_byvar_in = []

lagdiffs_byvar = []
lagmeans_byvar = []
for vv in range(2):
    diffs_bylr = []
    means_bylr = []
    for ll in range(nrngs):
        
        # Take Lag Range
        lr       = lag_ranges[ll]
        cesm_acf = acfs_byvar[vv].sel(lags=slice(lr[0],lr[1])).mean('ens') # (42, 89, 96, 12, 7)
        
        sm_acf   = sm_vars[vv].sel(lags=slice(lr[0],lr[1])).squeeze() # (lon: 65, lat: 48, mons: 12, lags: 7)>
        
        if len(sm_acf.shape) > 4:
            sm_acf = sm_acf.mean('ens') # Additional Ens Dimension for those calcualted with calc_crosscorr
            
        
        cesm_acf,sm_acf = proc.resize_ds([cesm_acf,sm_acf])
        
        diffs_lag_range = (sm_acf - cesm_acf).sum('lags')
        means_lag_range = (sm_acf - cesm_acf).mean('lags')
        
        diffs_bylr.append(diffs_lag_range)
        means_bylr.append(means_lag_range)
        
    
    diffs_bylr = xr.concat(diffs_bylr,dim='lag_range')
    means_bylr = xr.concat(means_bylr,dim='lag_range')
    
    diffs_bylr['lag_range']=lagrangenames
    means_bylr['lag_range']=lagrangenames
    
    lagdiffs_byvar.append(diffs_bylr.copy())
    lagmeans_byvar.append(means_bylr.copy())

#%% Visualize summed correlatino differences across lag range

kmonths     = [1,2]
vv          = 1
selmonstr   = proc.mon2str(kmonths)


npanels     = len(lagmeans_byvar[vv].lag_range)

fsz_title   = 26

#rei_in      = lagdiffs_byvar[vv].isel(mons=kmonths,).mean('mons') # [Year x Lat x Lon]
rei_in      = lagmeans_byvar[vv].isel(mons=kmonths,).mean('mons') # [Year x Lat x Lon]
lon         = rei_in.lon
lat         = rei_in.lat

bbplot2 = [-80,0,20,65]
if vv == 0:
    #levels  = np.arange(-5,5.5,.5)#np.arange(0,0.55,0.05)
    levels = np.arange(-.5,.55,0.05)
else:
    #levels  = np.arange(-15,16,1)#np.arange(0,0.55,0.05)
    levels = np.arange(-1,1.1,0.1)
plevels = np.arange(0,0.6,0.1)

cmapin  = 'cmo.balance'

if npanels == 4:
    fig,axs,mdict = viz.init_orthomap(1,4,bbplot2,figsize=(18,12),constrained_layout=True,centlat=45)
else:
    fig,axs,mdict = viz.init_orthomap(1,3,bbplot2,figsize=(26,10),constrained_layout=True,centlat=45)
for yy in range(npanels):
    
    
    
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

savename = "%sCESM1_%s_LagRngDiff_%s_EnsAvg.png" % (figpath,vnames[vv],selmonstr)
print(savename)
plt.savefig(savename,dpi=200,bbox_inches='tight',transparent=True)

# =============================================================================
#%% Try Above but for the paper draft =======================================
# =============================================================================

kmonths     = [1,2]
vv          = 0
#npanels     = len(lagmeans_byvar[vv].lag_range)
fsz_title   = 30
fsz_axis    = 24
fsz_tick    = 20
plot_point  = True
drop_col3   = True

lon         = rei_in.lon
lat         = rei_in.lat

bbplot2 = [-80,0,20,65]
if vv == 0:
    #levels  = np.arange(-5,5.5,.5)#np.arange(0,0.55,0.05)
    levels = np.arange(-.5,.55,0.05)
else:
    #levels  = np.arange(-15,16,1)#np.arange(0,0.55,0.05)
    levels = np.arange(-1,1.1,0.1)
plevels = np.arange(0,0.6,0.1)

cmapin        = 'cmo.balance'
if drop_col3:
    fig,axs,mdict = viz.init_orthomap(2,2,bbplot2,figsize=(19.5,17),constrained_layout=True,centlat=45)
else:
    fig,axs,mdict = viz.init_orthomap(2,3,bbplot2,figsize=(24,14.5),constrained_layout=True,centlat=45)
ii = 0
for vv in range(2):
    #rei_in     = lagdiffs_byvar[vv].isel(mons=kmonths,).mean('mons') # [Year x Lat x Lon]
    rei_in      = lagmeans_byvar[vv].isel(mons=kmonths,).mean('mons') # [Year x Lat x Lon]
    
    for yy in range(3):
        
        if drop_col3 and yy == 2:
            continue
        
        
        ax  = axs[vv,yy]
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
        ax.clabel(cl,fontsize=fsz_tick)
        
        
        # # Plot Mask
        # ax.contour(icemask.lon,icemask.lat,mask_plot,colors="cyan",linewidths=1.5,
        #            transform=mdict['noProj'],levels=[0,1],zorder=-1)
        
        # Plot Gulf Stream Position
        ax.plot(ds_gs2.lon.mean('mon'),ds_gs2.lat.mean('mon'),transform=proj,lw=1.75,c='k',ls='dashdot')
        
        # Plot Ice Edge
        ax.contour(icemask.lon,icemask.lat,mask_plot,colors="cyan",linewidths=2.5,
                   transform=proj,levels=[0,1],zorder=-1)
        
        if vv == 0:
            ax.set_title(lagrangenames[yy],fontsize=fsz_title)
        
        # Plot Regions
        if plot_point:
            nregs = len(ptnames)
            for ir in range(nregs):
                pxy   = ptcoords[ir]
                ax.plot(pxy[0],pxy[1],transform=proj,markersize=20,markeredgewidth=.5,c=ptcols[ir],
                        marker='*',markeredgecolor='k')
                
        else:
            
            for ir in range(nregs):
                rr   = regplot[ir]
                rbbx = bboxes[rr]
                
                ls_in = rsty[rr]
                if ir == 2:
                    ls_in = 'dashed'
                
                viz.plot_box(rbbx,ax=ax,color=rcols[rr],linestyle=ls_in,leglab=regions_long[rr],linewidth=2.5,return_line=True)

        
        if yy == 0:
            viz.add_ylabel(vnames[vv],ax=ax,rotation='vertical',fontsize=fsz_axis+6,y=0.6,x=-0.01)
            
        viz.label_sp(ii,alpha=0.75,ax=ax,fontsize=fsz_title,y=1.08,x=-.02)
        ii+=1
            

cb = viz.hcbar(pcm,ax=axs.flatten(),fraction=0.035,pad=0.010)
cb.ax.tick_params(labelsize=fsz_tick)
cb.set_label("Mean Diff. in Corr. (Stochastic Model - CESM1)",fontsize=fsz_axis)
    
savename = "%sDiff_bylag.png" % (figpath)
plt.savefig(savename,dpi=200,bbox_inches='tight',)#transparent=True)

#%% Try the above, but using different color ranges

kmonths     = [11,0,1,2]
vv          = 0
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

fig,axs,mdict = viz.init_orthomap(1,4,bbplot2,figsize=(28,12),constrained_layout=True,centlat=45)

for yy in range(4):
    if vv == 1: # SSS
        if yy == 0:
            levels = np.arange(-3,3.3,0.3)
        elif yy == 1 or yy == 2:
            levels = np.arange(-10,11,1)
        elif yy == 3:
            levels = np.arange(-20,22,2)
    elif vv == 0: # SST
        if yy == 0:
            levels = np.arange(-2,2.2,0.2)
        elif yy == 1:
            levels = np.arange(-3,3.3,0.3)
        elif yy == 2:
            levels = np.arange(-4,4.4,0.4)
        elif yy == 3:
            levels = np.arange(-5,5.5,0.5)
    
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
    ax.clabel(cl)
    
    # Plot Mask
    ax.contour(icemask.lon,icemask.lat,mask_plot,colors="k",linewidths=1.5,
               transform=mdict['noProj'],levels=[0,1],zorder=-1)
    
    ax.set_title("%s (%s months)" % (lagrangenames[yy],lagstrs[yy]),fontsize=fsz_title-2)

    cb = fig.colorbar(pcm,ax=ax,fraction=0.03,pad=0.0001,orientation='horizontal')
    cb.ax.tick_params(labelsize=fsz_tick)
#cb.set_label("Diff. in Corr. (%s, SM-CESM)" % vnames[vv],fontsize=fsz_axis-2)

savename = "%sCESM1_%s_LagRngDiff_DJFM_EnsAvg_Sep.png" % (figpath,vnames[vv])
plt.savefig(savename,dpi=200,bbox_inches='tight',transparent=True)

#%% Save Lagdiff Output

#sss_lagdiffs = lagdiffs_byvar[1].name("SSS")
#sst_lagdiffs = lagdiffs_byvar[0].name("SST")

for vv in range(2):
    
    
    # Save Lag diffs (sum)
    ds_out = lagdiffs_byvar[vv].rename(vnames[vv])
    edict  = proc.make_encoding_dict(ds_out)
    
    savename = "%sCESM1_vs_SM_%s_%s_LagRngDiff_DJFM_EnsAvg.nc" % (procpath,compare_name,vnames[vv],)
    print(savename)
    ds_out.to_netcdf(savename,encoding=edict)
    
    
    # Save lag means
    ds_out = lagmeans_byvar[vv].rename(vnames[vv])
    edict  = proc.make_encoding_dict(ds_out)
    savename = "%sCESM1_vs_SM_%s_%s_LagRngMean_DJFM_EnsAvg.nc" % (procpath,compare_name,vnames[vv],)
    print(savename)
    ds_out.to_netcdf(savename,encoding=edict)

#%% Make a silly quick plot

ilag = 12
imon = 1

fig,axs,_ = viz.init_orthomap(1,5,bboxplot,figsize=(28,12))

for ii in range(5):
    
    ax       =axs[ii]
    ax       =viz.add_coast_grid(ax,bbox=bboxplot)
    plotvar  =diffs_byvar[0].isel(ens=ii,mons=imon,lags=ilag)
    pcm      = ax.pcolormesh(plotvar.lon,plotvar.lat,plotvar.T,
                             cmap='cmo.balance',vmin=-0.5,vmax=0.5
                             ,transform=proj)
    
    


