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
# comparename     = "CESM_Draft1"
# expnames        = ["SST_CESM","SSS_CESM","SST_EOF_LbddCorr_Rerun","SSS_EOF_LbddCorr_Rerun_lbdE_neg"]
# expvars         = ["SST","SSS","SST","SSS"]
# expnames_long   = ["SST (CESM1)","SSS (CESM1)","SST (SM Coarse)","SSS (SM Coarse)"]
# expnames_short  = ["CESM_SST","CESM_SSS","SM_SST","SM_SSS"]
# ecols           = ["firebrick","navy","hotpink","cornflowerblue"]
# els             = ["solid",'solid','dashed','dashed']
# emarkers        = ["d","x","o","+"]


#%% 

#%% Plotting variables

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

#%% Load REI and Min Max Corr

nexps           = len(expnames)
rei_byvar       = []
maxmin_byvar    = []
for ex in range(4):
    st      = time.time()
    vname   = expvars[ex]
    expname = expnames[ex]
    
    
    # Load REI
    dsrei = xr.open_dataset("%s%s/Metrics/REI_Pointwise.nc" % (output_path,expname)).rei.load()
    rei_byvar.append(dsrei)
    
    # Load Max/Min Corr
    dsmaxmin = xr.open_dataset("%s%s/Metrics/MaxMin_Pointwise.nc" % (output_path,expname)).corr.load()
    maxmin_byvar.append(dsmaxmin)
    
    print("Loaded output for %s in %.2fs" % (expname,time.time()-st))

#%% Load Land Ice Mask

# Load Land Ice Mask
icemask    = xr.open_dataset(input_path + "masks/CESM1LE_HTR_limask_pacificmask_enssum_lon-90to20_lat0to90.nc")
mask       = icemask.MASK.squeeze()
mask_plot  = xr.where(np.isnan(mask),0,mask)#mask.copy()

mask_apply = icemask.MASK.squeeze().values


#%% Compare Re-emergence Indices for a given year

yy          = 0 # Year Index
selmons     = [1,2] # Month Indices
selmonstr   = proc.mon2str(selmons)

# plotting choice
levels      = np.arange(0,0.55,0.05)
fig,axs,_   = viz.init_orthomap(2,2,bbplot,figsize=(22,15),centlat=45,)

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
    plotvar = rei_byvar[ex].isel(yr=yy,mon=selmons).mean('mon')
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
    
plt.suptitle("Re-emergence Index, Year %i" % (yy+1),fontsize=fsz_title+6)
savename = "%sACF_REI_Comparison_%s_Year%02i_Mon%s.png" % (figpath,comparename,yy+1,selmonstr)
plt.savefig(savename,dpi=150,bbox_inches='tight')

    
#%% Compare Max/Min Correlation

yy          = 0 # Year Index
selmons     = [1,2] # Month Indices
selmonstr   = proc.mon2str(selmons)
maxminid    = 1 # 0 for min,1 for max

# plotting choice
if maxminid == 0:
    levels      = np.arange(0,1.05,0.05)
    title       = "Min Summertime Correlation"
else:
    levels      = np.arange(0.25,1.05,0.05)
    title       = "Max Wintertime Correlation"
fig,axs,_   = viz.init_orthomap(2,2,bbplot,figsize=(22,15),centlat=45,)


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
    

plt.suptitle(title,fontsize=fsz_title+6)

savename = "%sACF_REI_Comparison_%s_Year%02i_MaxMin%i_Mon%s.png" % (figpath,comparename,yy+1,maxminid,selmonstr)
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
        
    