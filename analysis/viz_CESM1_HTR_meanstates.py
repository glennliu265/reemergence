#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Visualize Mean states in CESM1

Uses data processed by:
    
    [calc_monmean_CESM1.py] : SST,SSS
    

Created on Tue Jun 11 12:25:53 2024

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
fsz_axis                    = 14
fsz_title                   = 16

proj                        = ccrs.PlateCarree()

#%% Load necessary data

ds_uvel,ds_vvel = dl.load_current()
ds_bsf          = dl.load_bsf(ensavg=False)
ds_ssh          = dl.load_bsf(ensavg=False,ssh=True)

# Load data processed by [calc_monmean_CESM1.py]
ds_sss          = dl.load_monmean('SSS')
ds_sst          = dl.load_monmean('SST')


tlon  = ds_uvel.TLONG.mean('ens').values
tlat  = ds_uvel.TLAT.mean('ens').values

#%% Load Masks

# Load Land Ice Mask
icemask  = xr.open_dataset(input_path + "masks/CESM1LE_HTR_limask_pacificmask_enssum_lon-90to20_lat0to90.nc")


mask = icemask.MASK.squeeze()
mask_plot = xr.where(np.isnan(mask),0,mask)#mask.copy()

mask_apply = icemask.MASK.squeeze().values
#mask_plot[np.isnan(mask)] = 0


ds_gs = dl.load_gs()
ds_gs = ds_gs.sel(lon=slice(-90,-50))

#%% Start Visualization (Ens Mean)

im          = 0
qint        = 1
contourvar  = "SSS"
cints_bsf   = np.arange(-50,55,5)
cints_sst   = np.arange(250,310,2)
cints_sss   = np.arange(33,39,.3)
cints_ssh   = np.arange(-200,200,10)

for im in range(12):
    
    # Initialize Plot and Map
    fig,ax,_    = viz.init_orthomap(1,1,bboxplot,figsize=(12,4))
    ax          = viz.add_coast_grid(ax,bbox=bboxplot,fill_color="lightgray")
    ax.set_title(mons3[im],fontsize=fsz_title)
    
    
    # Plot Currents
    plotu = ds_uvel.isel(month=im).UVEL.mean('ens').values
    plotv = ds_vvel.isel(month=im).VVEL.mean('ens').values
    ax.quiver(tlon[::qint,::qint],tlat[::qint,::qint],plotu[::qint,::qint],plotv[::qint,::qint],
              color='navy',transform=proj,alpha=0.75)
    
    if contourvar == "BSF":
        # Plot BSF
        plotbsf = ds_bsf.BSF.mean('ens').isel(mon=im).transpose('lat','lon')
        ax.contour(plotbsf.lon,plotbsf.lat,plotbsf,transform=proj,levels=cints_bsf,
                   linewidths=0.75,colors="k",)
    elif contourvar == "SSH":
        plotbsf = ds_ssh.SSH.mean('ens').isel(mon=im).transpose('lat','lon')
        cl = ax.contour(plotbsf.lon,plotbsf.lat,plotbsf,transform=proj,levels=cints_ssh,
                   linewidths=0.75,colors="k",)
        ax.clabel(cl)
    
    elif contourvar == "SST":
        # Plot mean SST
        plotvar = ds_sst.SST.mean('ens').isel(mon=im).transpose('lat','lon') * mask_apply
        cl = ax.contour(plotvar.lon,plotvar.lat,plotvar,transform=proj,
                    linewidths=1.5,colors="hotpink",levels=cints_sst)
        ax.clabel(cl)
    
    elif contourvar == 'SSS':
        # Plot mean SSS
        plotvar = ds_sss.SSS.mean('ens').isel(mon=im).transpose('lat','lon') * mask_apply
        cl = ax.contour(plotvar.lon,plotvar.lat,plotvar,transform=proj,
                    linewidths=1.5,colors="cornflowerblue",levels=cints_sss,linestyles='dashed')
        ax.clabel(cl)
    
    # Plot Gulf Stream Position
    ax.plot(ds_gs.lon,ds_gs.lat.mean('ens'),transform=proj,lw=1.75,c="k")
    
    
    figname = "%s%s_Current_Comparison_mon%02i.png" % (figpath,contourvar,im+1)
    plt.savefig(figname,dpi=150,bbox_inches='tight')
    
    
#%% Do Mean Version of above

# Get Bounding Boxes
regionset       = "SSSCSU"
regiondicts     = rparams.region_sets[regionset]
bboxes          = regiondicts['bboxes']
regions_long    = regiondicts['regions_long']
rcols           = regiondicts['rcols']
rsty            = regiondicts['rsty']

regplot = [0,1,3]
nregs   = len(regplot)



# Initialize Plot and Map
fig,ax,_    = viz.init_orthomap(1,1,bboxplot,figsize=(18,6.5))
ax          = viz.add_coast_grid(ax,bbox=bboxplot,fill_color="lightgray")
ax.set_title("CESM1 Historical Ens. Avg., Ann. Mean",fontsize=fsz_title)


# Plot Currents
plotu = ds_uvel.UVEL.mean('ens').mean('month').values
plotv = ds_vvel.VVEL.mean('ens').mean('month').values
ax.quiver(tlon[::qint,::qint],tlat[::qint,::qint],plotu[::qint,::qint],plotv[::qint,::qint],
          color='navy',transform=proj,alpha=0.75)

# Plot Mean SST (Colors)
plotvar = ds_sst.SST.mean('ens').mean('mon').transpose('lat','lon') * mask_apply
pcm     = ax.pcolormesh(plotvar.lon,plotvar.lat,plotvar,transform=proj,zorder=-1,
            linewidths=1.5,cmap="RdYlBu_r",vmin=275,vmax=310)
cb = viz.hcbar(pcm,ax=ax,fraction=0.045)
cb.set_label("SST ($\degree C$)",fontsize=fsz_axis)


# Plot Mean SSS (Contours)
plotvar = ds_sss.SSS.mean('ens').isel(mon=im).transpose('lat','lon') * mask_apply
cl = ax.contour(plotvar.lon,plotvar.lat,plotvar,transform=proj,
            linewidths=.75,colors="firebrick",levels=cints_sss,linestyles='dashed')
ax.clabel(cl)

for ir in range(nregs):
    rr = regplot[ir]
    rbbx = bboxes[rr]
    viz.plot_box(rbbx,color=rcols[rr],linestyle=rsty[rr],leglab=regions_long[rr],linewidth=2.5,return_line=True)


# Plot Gulf Stream Position
ax.plot(ds_gs.lon,ds_gs.lat.mean('ens'),transform=proj,lw=1.75,c="k")


figname = "%sCESM1_Locator_MeanState.png" % (figpath,)
plt.savefig(figname,dpi=150,bbox_inches='tight')
# -----------------------------------------------------------------------------
#%% Load and plot currents relative to Stochastic Model Fit (Diff by Lag, as computed
# in visualize_rei_acf.py)
# -----------------------------------------------------------------------------

rpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/proc/"
fns   = [
         "CESM1_vs_SM_PaperDraft01_SST_LagRngDiff_DJFM_EnsAvg.nc",
         "CESM1_vs_SM_PaperDraft01_SSS_LagRngDiff_DJFM_EnsAvg.nc"
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

#%% Make Plots of the mean stochastic model fit 

vv          = 0

for vv in range(2):
    #vname       = "SST"#"SSS"
    vname       = vnames[vv]
    plotvar     = ds_sum[vname]
    lat         = plotvar.lat
    cmapin      = 'cmo.balance'
    if vname == "SST":
        
        levels      = np.arange(-15,16,1)
    else:
        levels      = np.arange(-40,44,4)
    
    # Initialize Plot and Map
    fig,ax,_    = viz.init_orthomap(1,1,bboxplot,figsize=(18,6.5))
    ax          = viz.add_coast_grid(ax,bbox=bboxplot,fill_color="lightgray")
    
    # cf = ax.contourf(pv.lon,pv.lat,pv.T,
    #                  cmap='cmo.balance',
    #                  transform=proj)
    
    pcm     = ax.contourf(lon,lat,plotvar.T,cmap=cmapin,levels=levels,transform=proj,extend='both',zorder=-2)
    cl      = ax.contour(lon,lat,plotvar.T,colors='darkslategray',linewidths=.5,linestyles='solid',levels=levels,transform=proj,zorder=-2)
    ax.clabel(cl)
     
    # Plot Currents
    qint=2
    plotu = ds_uvel.UVEL.mean('ens').mean('month').values
    plotv = ds_vvel.VVEL.mean('ens').mean('month').values
    ax.quiver(tlon[::qint,::qint],tlat[::qint,::qint],plotu[::qint,::qint],plotv[::qint,::qint],
              color='navy',transform=proj,alpha=0.25)
    
    
    #ax.set_title("CESM1 Historical Ens. Avg., Ann. Mean",fontsize=fsz_title)
    cb = viz.hcbar(cf,ax=ax)
    
    figname = "%sCESM1_LagDiff_withcurrent_%s.png" % (figpath,vnames[vv])
    plt.savefig(figname,dpi=150,bbox_inches='tight')
    
# ----------------------------------------------
#%% Visualize currents at far northern latitudes
# ----------------------------------------------
fsz_ticks = 16
bboxice   = [-70,-10,55,70]
fig,ax,_  = viz.init_orthomap(1,1,bboxice,figsize=(16,12),centlon=-40)
ax        = viz.add_coast_grid(ax,bbox=bboxice,fill_color='lightgray')

# Plot Mean SSS (Contours)
cints_sss_ice = np.arange(30,36.4,0.4)
plotvar = ds_sss.SSS.mean('ens').isel(mon=im).transpose('lat','lon') #* mask_apply
cl = ax.contour(plotvar.lon,plotvar.lat,plotvar,transform=proj,
            linewidths=1.5,colors="firebrick",levels=cints_sss_ice,linestyles='solid')
ax.clabel(cl,fontsize=fsz_ticks)

# Plot Currents
qint  = 1
plotu = ds_uvel.UVEL.mean('ens').mean('month').values
plotv = ds_vvel.VVEL.mean('ens').mean('month').values
ax.quiver(tlon[::qint,::qint],tlat[::qint,::qint],plotu[::qint,::qint],plotv[::qint,::qint],
          color='navy',transform=proj,alpha=0.75)



#%%




# #%%



# if contourvar == "BSF":
#     # Plot BSF
#     plotbsf = ds_bsf.BSF.mean('ens').isel(mon=im).transpose('lat','lon')
#     ax.contour(plotbsf.lon,plotbsf.lat,plotbsf,transform=proj,levels=cints_bsf,
#                linewidths=0.75,colors="k",)
# elif contourvar == "SSH":
#     plotbsf = ds_ssh.SSH.mean('ens').isel(mon=im).transpose('lat','lon')
#     cl = ax.contour(plotbsf.lon,plotbsf.lat,plotbsf,transform=proj,levels=cints_ssh,
#                linewidths=0.75,colors="k",)
#     ax.clabel(cl)

# elif contourvar == "SST":
#     # Plot mean SST
#     plotvar = ds_sst.SST.mean('ens').isel(mon=im).transpose('lat','lon') * mask_apply
#     cl = ax.contour(plotvar.lon,plotvar.lat,plotvar,transform=proj,
#                 linewidths=1.5,colors="hotpink",levels=cints_sst)
#     ax.clabel(cl)

# elif contourvar == 'SSS':
#     # Plot mean SSS
#     plotvar = ds_sss.SSS.mean('ens').isel(mon=im).transpose('lat','lon') * mask_apply
#     cl = ax.contour(plotvar.lon,plotvar.lat,plotvar,transform=proj,
#                 linewidths=1.5,colors="cornflowerblue",levels=cints_sss,linestyles='dashed')
#     ax.clabel(cl)

# # Plot Gulf Stream Position
# ax.plot(ds_gs.lon,ds_gs.lat.mean('ens'),transform=proj,lw=1.75,c="k")


# figname = "%sCESM1_Locator_MeanState.png" % (figpath,contourvar,im+1)
# plt.savefig(figname,dpi=150,bbox_inches='tight')




#%% To Do
"""

- Make some regional visualizations
- Add contours of TEMP and SALT

"""
#%% Visualize Subpolar North Atlantic

# bboxice  = 
# fig,ax,_ = viz.init_orthomap(1,1,bboxice,figsize=(12,4))
# im = 1xs







