#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Compare Re-emergence Index calculated in calc_remidx_general
Copies format in visualize_rei_acf

Created on Fri Jul 12 08:35:06 2024

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

# # Indicate files containing ACFs
# cesm_name   = "CESM1_1920to2005_%sACF_lag00to60_ALL_ensALL.nc"
# vnames      =  ["SST","SSS"] #["SST","SSS","TEMP"]
# sst_expname = "SM_SST_EOF_LbddCorr_Rerun_SST_autocorrelation_thresALL_lag00to60.nc"
# sss_expname = "SM_SSS_EOF_LbddCorr_Rerun_lbdE_neg_SSS_autocorrelation_thresALL_lag00to60.nc"

# #sst_expname = "SM_SST_EOF_LbddCorr_Rerun_SST_autocorrelation_thresALL_lag00to60.nc"
# #sss_expname = "SM_SSS_EOF_LbddCorr_Rerun_lbdE_neg_SSS_autocorrelation_thresALL_lag00to60.nc"


# Indicate Experiment Names (copying format from compare_regional_metrics)
comparename     = "CESM_Coarse_Draft1"
expnames        = ["SST_CESM","SSS_CESM","SST_CESM1_5deg_lbddcoarsen_rerun","SSS_CESM1_5deg_lbddcoarsen"]
expvars         = ["SST","SSS","SST","SSS"]
expnames_long   = ["SST (CESM1)","SSS (CESM1)","SST (SM Coarse)","SSS (SM Coarse)"]
expnames_short  = ["CESM_SST","CESM_SSS","SM5_SST","SM5_SSS"]
ecols           = ["firebrick","navy","hotpink","cornflowerblue"]
els             = ["solid",'solid','dashed','dashed']
emarkers        = ["d","x","o","+"]

comparename     = "CESM_Coarse_v_Ori_Draft1"
expnames        = ["SST_EOF_LbddCorr_Rerun","SSS_EOF_LbddCorr_Rerun_lbdE_neg","SST_CESM1_5deg_lbddcoarsen_rerun","SSS_CESM1_5deg_lbddcoarsen"]
expvars         = ["SST","SSS","SST","SSS"]
expnames_long   = ["SST (SM)","SSS (SM)","SST (SM Coarse)","SSS (SM Coarse)"]
expnames_short  = ["SM_SST","SM_SSS","SM5_SST","SM5_SSS"]
ecols           = ["firebrick","navy","hotpink","cornflowerblue"]
els             = ["solid",'solid','dashed','dashed']
emarkers        = ["d","x","o","+"]

# #% Paper Draft Comparison
comparename     = "CESM_Draft1"
expnames        = ["SST_CESM","SSS_CESM","SST_EOF_LbddCorr_Rerun","SSS_EOF_LbddCorr_Rerun_lbdE_neg"]
expvars         = ["SST","SSS","SST","SSS"]
expnames_long   = ["SST (CESM1)","SSS (CESM1)","SST (Stochastic Model)","SSS (Stochastic Model)"]
expnames_short  = ["CESM_SST","CESM_SSS","SM_SST","SM_SSS"]
ecols           = ["firebrick","navy","hotpink","cornflowerblue"]
els             = ["solid",'solid','dashed','dashed']
emarkers        = ["d","x","o","+"]

# #% Paper Draft Comparison
comparename     = "CESM_Draft2"
expnames        = ["SST_CESM","SSS_CESM","SST_Draft01_Rerun_QekCorr","SSS_Draft01_Rerun_QekCorr"]
expvars         = ["SST","SSS","SST","SSS"]
expnames_long   = ["SST (CESM1)","SSS (CESM1)","SST (Stochastic Model)","SSS (Stochastic Model)"]
expnames_short  = ["CESM_SST","CESM_SSS","SM_SST","SM_SSS"]
ecols           = ["firebrick","navy","hotpink","cornflowerblue"]
els             = ["solid",'solid','dashed','dashed']
emarkers        = ["d","x","o","+"]


# #% Paper Draft Comparison
comparename     = "Draft3"
expnames        = ["SST_CESM","SSS_CESM","SST_Draft03_Rerun_QekCorr","SSS_Draft03_Rerun_QekCorr"]
expvars         = ["SST","SSS","SST","SSS"]
expnames_long   = ["SST (CESM1)","SSS (CESM1)","SST (Stochastic Model)","SSS (Stochastic Model)"]
expnames_short  = ["CESM_SST","CESM_SSS","SM_SST","SM_SSS"]
ecols           = ["firebrick","navy","hotpink","cornflowerblue"]
els             = ["solid",'solid','dashed','dashed']
emarkers        = ["d","x","o","+"]

# #% Paper Draft Comparison
comparename     = "RevisionD1"
expnames        = ["SST_CESM","SSS_CESM","SST_Revision_Qek_TauReg","SSS_Revision_Qek_TauReg"]
expvars         = ["SST","SSS","SST","SSS"]
expnames_long   = ["SST (CESM1)","SSS (CESM1)","SST (Stochastic Model)","SSS (Stochastic Model)"]
expnames_short  = ["CESM_SST","CESM_SSS","SM_SST","SM_SSS"]
ecols           = ["firebrick","navy","hotpink","cornflowerblue"]
els             = ["solid",'solid','dashed','dashed']
emarkers        = ["d","x","o","+"]


# #% Compare LbdE Effect
# comparename     = "sm_lbdE_effect"
# expnames        = ["SSS_CESM","SSS_Draft01_Rerun_QekCorr","SSS_Draft01_Rerun_QekCorr_NoLbde"]
# expvars         = ["SSS","SSS","SSS"]
# expnames_long   = ["SSS (CESM1)","SSS (Stochastic Model)","SSS (Stochastic Model, No $\lambda^e$)"]
# expnames_short  = ["CESM_SSS","SM_SSS","SM_SSS_NoLbde"]
# ecols           = ["k","magenta","forestgreen"]
# els             = ["solid",'dashed','dotted']
# emarkers        = ["d","x","o"]



#%%  Load Set information

pointset       = "PaperDraft02"
regionset      = "SSSCSU"
setname        = regionset

#rrsel          = ["SAR","NAC","STGe"]
rrsel = ["SAR","NAC","IRM"]

# Get Region Info
rdict                       = rparams.region_sets[regionset]
regions                     = rdict['regions']
bboxes                      = rdict['bboxes']
rcols                       = rdict['rcols']
rsty                        = rdict['rsty']
regions_long                = rdict['regions_long']
nregs                       = len(bboxes)


# Get Point Info
ptdict      = rparams.point_sets[pointset]
ptcoords    = ptdict['bboxes']
ptnames     = ptdict['regions']
ptnames_long = ptdict['regions_long']
ptcols      = ptdict['rcols']
ptsty       = ptdict['rsty']


#%% Plotting variables

darkmode = False

# Plotting Information
bbplot                      = [-80,0,20,65]
mpl.rcParams['font.family'] = 'Avenir'
proj                        = ccrs.PlateCarree()
mons3                       = proc.get_monstr()

# Font Sizes
fsz_title                   = 32
fsz_tick                    = 18
fsz_axis                    = 24
fsz_legend                  = 18

#%% Load REI and Min Max  and T2

nexps           = len(expnames)
rei_byvar       = []
maxmin_byvar    = []
t2_byvar        = []
for ex in range(nexps):
    st      = time.time()
    vname   = expvars[ex]
    expname = expnames[ex]
    
    
    # Load REI
    dsrei = xr.open_dataset("%s%s/Metrics/REI_Pointwise.nc" % (output_path,expname)).rei.load()
    rei_byvar.append(dsrei)
    
    # Load Max/Min Corr
    dsmaxmin = xr.open_dataset("%s%s/Metrics/MaxMin_Pointwise.nc" % (output_path,expname)).corr.load()
    maxmin_byvar.append(dsmaxmin)
    
    # # Load T2
    # dst2 = xr.open_dataset("%s%s/Metrics/T2_Timescale.nc" % (output_path,expname)).T2.load()
    # t2_byvar.append(dst2)
    
    print("Loaded output for %s in %.2fs" % (expname,time.time()-st))

#%% Load Land Ice Mask
bboxplot   = bbplot
# Load Land Ice Mask
icemask    = xr.open_dataset(input_path + "masks/CESM1LE_HTR_limask_pacificmask_enssum_lon-90to20_lat0to90.nc")
mask       = icemask.MASK.squeeze()
mask_plot  = xr.where(np.isnan(mask),0,mask)#mask.copy()

mask_apply = icemask.MASK.squeeze().values

# Get A region mask
mask_reg_sub    = proc.sel_region_xr(mask,bboxplot)
mask_reg_ori    = xr.ones_like(mask) * 0
mask_reg        = mask_reg_ori + mask_reg_sub

ds_gs2          = dl.load_gs(load_u2=True)

# <0> <0> <0> <0> <0> <0> <0> <0> <0> <0> <0> <0> <0> <0> <0> <0>
#%% Compare Re-emergence Indices for a given year
# Note this was usef for paper draft 1

import cmocean as cmo

ds_mask_mat = []
for ex in range(4):
    yy          = 0
    selmons     = [1,2]
    plotvar = rei_byvar[ex].isel(yr=yy,mon=selmons).mean('mon')
    if 'ens' in list(plotvar.dims):
        plotvar = plotvar.mean('ens')
    ds_mask_mat.append(plotvar)
mask2 = proc.make_mask(ds_mask_mat)


if darkmode:
    dfcol = "w"
    transparent = True
    plt.style.use('dark_background')
    mpl.rcParams['font.family'] = 'Avenir'
else:
    dfcol = "k"
    transparent = False
    plt.style.use('default')
    mpl.rcParams['font.family'] = 'Avenir'

yy          = 0 # Year Index
selmons     = [1,2] # Month Indices
selmonstr   = proc.mon2str(selmons)
plot_point  = True
poster_ver  = False # Omit all
include_title    = False

# plotting choice
levels      = np.arange(0,0.55,0.05)
fig,axs,_   = viz.init_orthomap(2,2,bbplot,figsize=(26,18.5),centlat=45,)

modelname   = ["CESM1","Stochastic Model"]
vnames_plot = ["SST","SSS"]
plotorder= [0,2,1,3]

for ex in range(4):
    
    # Set up Axis
    ax           = axs.flatten()[plotorder[ex]]
    ax           = viz.add_coast_grid(ax,bbplot,fill_color="lightgray",fontsize=20,
                                    fix_lon=np.arange(-80,10,10),fix_lat=np.arange(0,70,10),grid_color='k')
    
    if plotorder[ex] < 2:
        ax.set_title(modelname[plotorder[ex]],fontsize=fsz_title)
    #ax.set_title(expnames_long[ex],fontsize=fsz_title)
    
    # Set plotting options
    vname = expvars[ex]
    if vname == "SSS":
        cmap_in = cmo.cm.deep #"cmo.deep"
        cmap_in.set_under('lightyellow')
        cmap_in.set_over('royalblue')
    elif vname == "SST":
        cmap_in = cmo.cm.dense#cmo.dense"
        cmap_in.set_under('lightcyan')
        cmap_in.set_over('darkorchid')
        
    
    # Prepare Plotting Variable
    plotvar = rei_byvar[ex].isel(yr=yy,mon=selmons).mean('mon')
    if "ens" in list(plotvar.dims):
        plotvar = plotvar.mean('ens')
    
    
    #plotvar = ds_mask_mat[ex]
    #mask3 = proc.make_mask(plotvar,nanval=0.)
    
    lon     = plotvar.lon
    lat     = plotvar.lat
    plotvar = plotvar * mask_reg #* mask2 #* mask3
    
    # Note: Assign small value to regions that are 0
    # To fix the issue with locations with 0. REI...
    plotvar = xr.where(plotvar == 0.,1e-7,plotvar)
    
    
    # Add contours
    pcm     = ax.contourf(plotvar.lon,plotvar.lat,plotvar,cmap=cmap_in,levels=levels,transform=proj,extend='both',zorder=-2)
    cl      = ax.contour(plotvar.lon,plotvar.lat,plotvar,colors='darkslategray',linewidths=.5,linestyles='solid',levels=levels,transform=proj,zorder=-2)
    ax.clabel(cl,fontsize=fsz_tick,inline_spacing=2)
    
    #plt.contourf(plotvar.lon,plotvar.lat,plotvar,cmap=cmap_in,levels=levels,transform=proj,extend='both',zorder=-2)
    
    # Plot Land Ice Mask
    ax.contour(icemask.lon,icemask.lat,mask_plot,colors="lightgray",linewidths=2,
                transform=proj,levels=[0,1],zorder=-1)
    
    if ex > 1:
        
        cb = fig.colorbar(pcm,ax=axs[ex%2,:].flatten(),fraction=0.015,pad=0.015)
        cb.ax.tick_params(labelsize=fsz_tick)
        cb.set_label("%s Re-emergence Index" % vname,fontsize=fsz_axis)
        #fig.colorbar()
    
    if plotorder[ex]%2 == 0:
        
        viz.add_ylabel("%s" % vnames_plot[ex],ax=ax,y=0.65,x=0.01,
                       fontsize=fsz_title)
    ax=viz.label_sp(plotorder[ex],ax=ax,x=0.05,y=1.01,alpha=0,fig=fig,
                 labelstyle="%s)",usenumber=False,fontsize=fsz_title,fontcolor=dfcol)
    
    
    # Add additional features
    ax.plot(ds_gs2.lon.mean('mon'),ds_gs2.lat.mean('mon'),transform=proj,lw=2.5,c='k',ls='dashdot')

    # Plot Ice Edge
    ax.contour(icemask.lon,icemask.lat,mask_plot,colors="cyan",linewidths=2,
               transform=proj,levels=[0,1],zorder=-1)
    
    #ax.set_facecolor('gray')
    
    # Plot the Bounding Boxes
    if not poster_ver:
        if plot_point: # Plot points
            
            nreg = len(ptnames)
            for rr in range(nreg):
                pxy   = ptcoords[rr]
                pname = ptnames[rr]
                if pname not in rrsel:
                    continue
                
                if ex == 0 and pname == "IRM":
                    ax.plot(pxy[0],pxy[1],transform=proj,markersize=46,markeredgewidth=.5,c=ptcols[rr],
                            marker='*',markeredgecolor='k')
                elif ex == 1 and pname != "IRM":
                    ax.plot(pxy[0],pxy[1],transform=proj,markersize=46,markeredgewidth=.5,c=ptcols[rr],
                            marker='*',markeredgecolor='k')
                  
        else:
            rdict = rparams.region_sets[setname]
            nreg  = len(rdict['bboxes'])
            for rr in range(nreg):
                bbp    = rdict['bboxes'][rr]
                bbname = rdict['regions'][rr]
                if bbname not in rrsel:
                    continue
                
                if bbname == "STGe" and plotorder[ex] == 0:
                    viz.plot_box(bbp,color=rdict['rcols'][rr],linewidth=4,proj=proj,ax=ax)
                elif plotorder[ex] == 2 and bbname != "STGe":
                    viz.plot_box(bbp,color=rdict['rcols'][rr],linewidth=4,proj=proj,ax=ax)
if include_title:
    plt.suptitle("Re-emergence Index, Year %i" % (yy+1),fontsize=fsz_title+6)
savename = "%sACF_REI_Comparison_%s_Year%02i_Mon%s.png" % (figpath,comparename,yy+1,selmonstr)
if darkmode:
    
    savename = proc.addstrtoext(savename,"_darkmode")
plt.savefig(savename,dpi=150,bbox_inches='tight',transparent=transparent)

#%%

lons = [ds.lon.data for ds in ds_mask_mat]
lats = [ds.lat.data for ds in ds_mask_mat]

    
#%% Compare Max/Min Correlation

yy          = 0 # Year Index
selmons     = [1,2] # Month Indices
selmonstr   = proc.mon2str(selmons)
maxminid    = 0 # 0 for min,1 for max

# plotting choice
if maxminid == 0:
    levels      = np.arange(0,1.05,0.05)
    title       = "Min Summertime Correlation"
else:
    levels      = np.arange(0.25,1.05,0.05)
    title       = "Max Wintertime Correlation"
fig,axs,_   = viz.init_orthomap(2,2,bbplot,figsize=(22,18),centlat=45,)

for ex in range(4):
    
    # Set up Axis
    ax           = axs.flatten()[ex]
    ax           = viz.add_coast_grid(ax,bbplot,fill_color="lightgray",fontsize=20,
                                    fix_lon=np.arange(-80,10,10),fix_lat=np.arange(0,70,10),grid_color="k")
    
    ax.set_title(expnames_long[ex],fontsize=fsz_title)
    
    # Set plotting options
    vname = expvars[ex]
    if vname == "SSS":
        cmap_in = "cmo.deep"
    elif vname == "SST":
        cmap_in = "cmo.dense"
    
    # Prepare Plotting Variable
    plotvar = maxmin_byvar[ex].isel(yr=yy,mon=selmons).mean('mon').isel(maxmin=maxminid)
    if "ens" in list(plotvar.dims):
        plotvar = plotvar.mean('ens')
    lon     = plotvar.lon
    lat     = plotvar.lat
    
    # Add contours
    pcm     = ax.contourf(lon,lat,plotvar,cmap=cmap_in,levels=levels,transform=proj,extend='both',zorder=-2)
    cl      = ax.contour(lon,lat,plotvar,colors='darkslategray',linewidths=.5,linestyles='solid',levels=levels,transform=proj,zorder=-2)
    ax.clabel(cl,fontsize=fsz_tick,inline_spacing=2)
    
    
    
    # Plot Land Ice Mask
    ax.contour(icemask.lon,icemask.lat,mask_plot,colors="lightgray",linewidths=2,
               transform=proj,levels=[0,1],zorder=-1)
    
        
    if ex > 1:
        
        cb = viz.hcbar(pcm,ax=axs[:,ex%2].flatten(),fraction=0.025)
        cb.ax.tick_params(labelsize=fsz_tick)
        cb.set_label("%s %s" % (vname,title),fontsize=fsz_axis)
        #fig.colorbar()
    

plt.suptitle(title,fontsize=fsz_title+6)

savename = "%sACF_REI_Comparison_%s_Year%02i_MaxMin%i_Mon%s.png" % (figpath,comparename,yy+1,maxminid,selmonstr,)
plt.savefig(savename,dpi=150,bbox_inches='tight')
  
#%% Put all the above in the same plot

yy          = 0 # Year Index
selmons     = [1,2] # Month Indices

for ex in range(nexps):
    
    # plotting choice
    if maxminid == 0:
        levels      = np.arange(0,1.05,0.05)
        title       = "Min Summertime Correlation"
    else:
        levels      = np.arange(0.25,1.05,0.05)
        title       = "Max Wintertime Correlation"
    fig,axs,_   = viz.init_orthomap(1,3,bbplot,figsize=(26,12.5),centlat=45,)
    
    
    for ii in range(3):
        ax = axs.flatten()[ii]
        ax           = viz.add_coast_grid(ax,bbplot,fill_color="lightgray",fontsize=20,
                                        fix_lon=np.arange(-80,10,10),fix_lat=np.arange(0,70,10),grid_color="k")
        
        
        
        
        if ii == 0:
            title   = "REI"
            plotvar = rei_byvar[ex].isel(yr=yy,mon=selmons).mean('mon')
            levels  = np.arange(0,0.55,0.05)
        elif ii == 1:
            title   = "Max Wintertime Correlation"
            plotvar = maxmin_byvar[ex].isel(yr=yy,mon=selmons).mean('mon').isel(maxmin=1)
            levels  = np.arange(0.25,1.05,0.05)
        elif ii == 2:
            title   = "Min Summertime Correlation"
            plotvar = maxmin_byvar[ex].isel(yr=yy,mon=selmons).mean('mon').isel(maxmin=0)
            levels  = np.arange(0,1.05,0.05)
        
        
        if "ens" in list(plotvar.dims):
            plotvar = plotvar.mean('ens')
        lon     = plotvar.lon
        lat     = plotvar.lat
        
        
        # Add contours
        pcm     = ax.contourf(lon,lat,plotvar,cmap=cmap_in,levels=levels,transform=proj,extend='both',zorder=-2)
        cl      = ax.contour(lon,lat,plotvar,colors='darkslategray',linewidths=.5,linestyles='solid',levels=levels,transform=proj,zorder=-2)
        ax.clabel(cl,fontsize=fsz_tick,inline_spacing=2)
        
        
        
        # Plot Land Ice Mask
        ax.contour(icemask.lon,icemask.lat,mask_plot,colors="lightgray",linewidths=2,
                   transform=proj,levels=[0,1],zorder=-1)
        
        
        ax.set_title(title,fontsize=fsz_title)
        cb = viz.hcbar(pcm,ax=ax,fraction=0.035)
        cb.ax.tick_params(labelsize=fsz_tick)
        if ii == 0:
            
            ax = viz.add_ylabel(expnames_long[ex],ax=ax,x=-0.2,fontsize=fsz_axis,rotation='horizontal')
        
        
    plt.savefig("%sREI_decomposition_%s_mons_wint.png"% (figpath,expnames[ex]),dpi=150,bbox_inches='tight')

#%% Examine Differences between two experiments

id_exp1     = 3
id_exp2     = 1

diffname_fn = "%s_v_%s" % (expnames[id_exp1],expnames[id_exp2])

fig,axs,_   = viz.init_orthomap(1,3,bbplot,figsize=(26,8),centlat=45,)

for ii in range(3):
    ax           = axs.flatten()[ii]
    ax           = viz.add_coast_grid(ax,bbplot,fill_color="lightgray",fontsize=20,
                                    fix_lon=np.arange(-80,10,10),fix_lat=np.arange(0,70,10),grid_color="k")
    

    
    
    if ii == 0:
        title       = "REI Diff. "
        plotvar1    = rei_byvar[id_exp1].isel(yr=yy,mon=selmons).mean('mon')
        plotvar2    = rei_byvar[id_exp2].isel(yr=yy,mon=selmons).mean('mon')
        plotvar     = plotvar1 - plotvar2
        levels      = np.arange(-0.50,0.55,0.05)
    elif ii == 1:
        title       = "Wintertime Max Diff."
        plotvar1    = maxmin_byvar[id_exp1].isel(yr=yy,mon=selmons).mean('mon').isel(maxmin=1)
        plotvar2    = maxmin_byvar[id_exp2].isel(yr=yy,mon=selmons).mean('mon').isel(maxmin=1)
        plotvar     = plotvar1 - plotvar2
        levels      = np.arange(-0.28,0.32,0.04)
    elif ii == 2:
        
        title       = "Summertime Min Diff."
        plotvar1    = maxmin_byvar[id_exp1].isel(yr=yy,mon=selmons).mean('mon').isel(maxmin=0)
        plotvar2    = maxmin_byvar[id_exp2].isel(yr=yy,mon=selmons).mean('mon').isel(maxmin=0)
        plotvar     = plotvar1 - plotvar2
        levels      = np.arange(-0.28,0.32,0.04)
    
    cmap_in = 'cmo.balance'
    
    if "ens" in list(plotvar.dims):
        plotvar = plotvar.mean('ens')
    lon     = plotvar.lon
    lat     = plotvar.lat
    
    
    # Add contours
    pcm     = ax.contourf(lon,lat,plotvar,cmap=cmap_in,levels=levels,transform=proj,extend='both',zorder=-2)
    cl      = ax.contour(lon,lat,plotvar,colors='darkslategray',linewidths=.5,linestyles='solid',levels=levels,transform=proj,zorder=-2)
    ax.clabel(cl,fontsize=fsz_tick,inline_spacing=2)
    
    
    
    # Plot Land Ice Mask
    ax.contour(icemask.lon,icemask.lat,mask_plot,colors="lightgray",linewidths=2,
               transform=proj,levels=[0,1],zorder=-1)
    
    
    ax.set_title(title,fontsize=fsz_title)
    cb = viz.hcbar(pcm,ax=ax,fraction=0.045)
    cb.ax.tick_params(labelsize=fsz_tick)
    #if ii == 0:
        #ax = viz.add_ylabel(expnames_long[ex],ax=ax,x=-0.2,fontsize=fsz_axis,rotation='horizontal')
plt.suptitle("%s - %s" % (expnames_long[id_exp1], expnames_long[id_exp2]),fontsize=fsz_title+10)
plt.savefig("%sREI_decomposition_%s_mons_wint_diffs_%s.png"% (figpath,expnames[ex],diffname_fn),dpi=150,bbox_inches='tight')

#%% Plot T2


if darkmode:
    dfcol = "w"
    transparent = True
    plt.style.use('dark_background')
    mpl.rcParams['font.family'] = 'Avenir'
else:
    dfcol = "k"
    transparent = False
    plt.style.use('default')
    mpl.rcParams['font.family'] = 'Avenir'

yy          = 0 # Year Index
selmons     = [1,2] # Month Indices
selmonstr   = proc.mon2str(selmons)

# plotting choice
levels      = np.arange(0,36,1)#np.arange(0,66,3)
fig,axs,_   = viz.init_orthomap(2,2,bbplot,figsize=(26,18.5),centlat=45,)

modelname   = ["CESM1","Stochastic Model"]
vnames_plot = ["SST","SSS"]
plotorder= [0,2,1,3]

for ex in range(4):
    
    # Set up Axis
    ax           = axs.flatten()[plotorder[ex]]
    ax           = viz.add_coast_grid(ax,bbplot,fill_color="lightgray",fontsize=20,
                                    fix_lon=np.arange(-80,10,10),fix_lat=np.arange(0,70,10),grid_color="k")
    
    if plotorder[ex] < 2:
        ax.set_title(modelname[plotorder[ex]],fontsize=fsz_title)
    #ax.set_title(expnames_long[ex],fontsize=fsz_title)
    
    # Set plotting options
    vname = expvars[ex]
    if vname == "SSS":
        cmap_in = "cmo.deep"
    elif vname == "SST":
        cmap_in = "cmo.dense"
    
    # Prepare Plotting Variable
    plotvar = t2_byvar[ex].isel(mon=selmons).mean('mon').T#rei_byvar[ex].isel(yr=yy,mon=selmons).mean('mon')
    if "ens" in list(plotvar.dims):
        plotvar = plotvar.mean('ens')
    lon     = plotvar.lon
    lat     = plotvar.lat
    plotvar = plotvar * mask_reg
    
    # Add contours
    pcm     = ax.contourf(plotvar.lon,plotvar.lat,plotvar,cmap=cmap_in,levels=levels,transform=proj,extend='both',zorder=-2)
    cl      = ax.contour(plotvar.lon,plotvar.lat,plotvar,colors='darkslategray',linewidths=.5,linestyles='solid',levels=levels,transform=proj,zorder=-2)
    ax.clabel(cl,fontsize=fsz_tick,inline_spacing=2)
    
    
    # # Plot Land Ice Mask
    # ax.contour(icemask.lon,icemask.lat,mask_plot,colors="lightgray",linewidths=2,
    #            transform=proj,levels=[0,1],zorder=-1)
    
    if ex > 1:
        
        cb = fig.colorbar(pcm,ax=axs[ex%2,:].flatten(),fraction=0.015,pad=0.015)
        cb.ax.tick_params(labelsize=fsz_tick)
        cb.set_label("%s Decorrelation Timescale (Months)" % vname,fontsize=fsz_axis)
        #fig.colorbar()
    
    if plotorder[ex]%2 == 0:
        
        viz.add_ylabel("%s" % vnames_plot[ex],ax=ax,y=0.65,x=0.01,
                       fontsize=fsz_title)
    ax=viz.label_sp(plotorder[ex],ax=ax,x=0.05,y=1.01,alpha=0,fig=fig,
                 labelstyle="%s)",usenumber=False,fontsize=fsz_title,fontcolor=dfcol)
    
    
    # Add additional features
    ax.plot(ds_gs2.lon.mean('mon'),ds_gs2.lat.mean('mon'),transform=proj,lw=2.5,c='firebrick',ls='dashdot')

    # Plot Ice Edge
    ax.contour(icemask.lon,icemask.lat,mask_plot,colors="cyan",linewidths=2,
               transform=proj,levels=[0,1],zorder=-1)
    
    # Plot the Bounding Boxes
    
    rdict = rparams.region_sets[setname]
    nreg  = len(rdict['bboxes'])
    for rr in range(nreg):
        bbp    = rdict['bboxes'][rr]
        bbname = rdict['regions'][rr]
        if bbname not in rrsel:
            continue
        
        if bbname == "STGe" and plotorder[ex] == 0:
            viz.plot_box(bbp,color=rdict['rcols'][rr],linewidth=4,proj=proj,ax=ax)
        elif plotorder[ex] == 2 and bbname != "STGe":
            viz.plot_box(bbp,color=rdict['rcols'][rr],linewidth=4,proj=proj,ax=ax)
    

plt.savefig("%sREI_decomposition_%s_mons_wint.png"% (figpath,expnames[ex]),dpi=150,bbox_inches='tight')

  
# plt.suptitle("Re-emergence Index, Year %i" % (yy+1),fontsize=fsz_title+6)
# savename = "%sACF_REI_Comparison_%s_Year%02i_Mon%s.png" % (figpath,comparename,yy+1,selmonstr)
# if darkmode:
#     savename = proc.addstrtoext(savename,"_darkmode")
# plt.savefig(savename,dpi=150,bbox_inches='tight',transparent=transparent)

def make_mask(ds_all,nanval=np.nan):
    if ~np.isnan(nanval):
        ds_all = [xr.where((ds == nanval),np.nan,ds) for ds in ds_all]
    mask    = [xr.where(~np.isnan(ds),1,np.nan) for ds in ds_all]
    mask    = xr.concat(mask,dim='exp')
    mask  = mask.prod('exp',skipna=False)
    return mask

#%% Look at differences in persistence between experiments


selmons     = [1,2]
selmonstr   = proc.mon2str(selmons)
vlms        = [-10,10]
pmesh       = False

cints_t2_diff = np.arange(-10,11,1)

if comparename == "sm_lbdE_effect":
    
    exp1        = t2_byvar[1].isel(mon=selmons).mean('mon')
    exp2        = t2_byvar[2].isel(mon=selmons).mean('mon')
    
    mask2 = make_mask([exp1,exp2],nanval=1.)
    
    
    plotvar     = (exp1 - exp2).T  * mask * mask2
    diffname    = "%s - %s" % (expnames_long[1],expnames_long[2])
    
    fig,ax,_    = viz.init_orthomap(1,1,bboxplot,figsize=(24,12))
    ax          = viz.add_coast_grid(ax,bbox=bboxplot,fill_color="lightgray",fontsize=fsz_tick)
    
    if pmesh:
        pcm     = ax.pcolormesh(plotvar.lon,plotvar.lat,plotvar,transform=proj,zorder=-1,
                    linewidths=1.5,cmap="cmo.balance",vmin=vlms[0],vmax=vlms[1])
    else:
        pcm     = ax.contourf(plotvar.lon,plotvar.lat,plotvar,transform=proj,zorder=-1,
                    cmap='cmo.balance',levels=cints_t2_diff,extend='both')
        cl      = ax.contour(plotvar.lon,plotvar.lat,plotvar,transform=proj,zorder=-1,
                    colors="k",linewidths=0.75,levels=cints_t2_diff)
        ax.clabel(cl,fontsize=fsz_tick)
      
    cb=viz.hcbar(pcm)
    cb.ax.tick_params(labelsize=fsz_tick)
    cb.set_label("$T^2$ Difference (Months)\n%s" % diffname,fontsize=fsz_axis)
    
    plt.savefig("%sT2_Difference_lbdE_effect.png"% (figpath,),dpi=150,bbox_inches='tight')

#%%

    
    
    
    