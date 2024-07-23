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
fsz_axis                    = 22
fsz_title                   = 28

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


# Load Mixed-Layer Depth
mldpath = input_path + "mld/"
mldnc   = "CESM1_HTR_FULL_HMXL_NAtl_EnsAvg.nc"
ds_mld  = xr.open_dataset(mldpath+mldnc).h.load()

#ds_h          = dl.load_monmean('HMXL')

#%% Load Masks

# Load Land Ice Mask
icemask     = xr.open_dataset(input_path + "masks/CESM1LE_HTR_limask_pacificmask_enssum_lon-90to20_lat0to90.nc")


mask        = icemask.MASK.squeeze()
mask_plot   = xr.where(np.isnan(mask),0,mask)#mask.copy()

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

#%% Start Visualization (Ens Mean)

im          = 0
qint        = 1
contourvar  = "SSS"
cints_bsf   = np.arange(-50,55,5)
cints_sst   = np.arange(250,310,2)
cints_sss   = np.arange(33,39,.3)
cints_ssh   = np.arange(-200,200,10)
cints_rei   = np.arange(0,0.65,0.05)

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
    ls_in = rsty[rr]
    if ir == 2:
        ls_in = 'dashed'
        
    rbbx = bboxes[rr]
    viz.plot_box(rbbx,color=rcols[rr],linestyle=ls_in,leglab=regions_long[rr],linewidth=2.5,return_line=True)


# Plot Gulf Stream Position
#ax.plot(ds_gs.lon,ds_gs.lat.mean('ens'),transform=proj,lw=1.75,c="k",ls='dashed')
ax.plot(ds_gs2.lon.mean('mon'),ds_gs2.lat.mean('mon'),transform=proj,lw=1.75,c='k',ls='dashdot')

# Plot Ice Edge
ax.contour(icemask.lon,icemask.lat,mask_plot,colors="cyan",linewidths=2.5,
           transform=proj,levels=[0,1],zorder=-1)



figname = "%sCESM1_Locator_MeanState.png" % (figpath,)
plt.savefig(figname,dpi=150,bbox_inches='tight')

#%% Same Plot as above, but focus on SSS re-emergence (locator for TCM presentation)

fsz_tick = 14

# Initialize Plot and Map
fig,ax,_    = viz.init_orthomap(1,1,bboxplot,figsize=(18,6))
ax          = viz.add_coast_grid(ax,bbox=bboxplot,fill_color="lightgray",fontsize=fsz_tick)
#ax.set_title("CESM1 Historical Ens. Avg., Ann. Mean",fontsize=fsz_title)


# Plot Currents
qint  = 1
plotu = ds_uvel.UVEL.mean('ens').mean('month').values
plotv = ds_vvel.VVEL.mean('ens').mean('month').values
ax.quiver(tlon[::qint,::qint],tlat[::qint,::qint],plotu[::qint,::qint],plotv[::qint,::qint],
          color='mediumblue',transform=proj,alpha=0.35,)#scale=1e3)

# Plot REI Index
plotvar = reiplot
pcm     = ax.contourf(plotvar.lon,plotvar.lat,plotvar,transform=proj,
                      cmap='cmo.deep',levels=cints_rei,zorder=-3)
cl     = ax.contour(plotvar.lon,plotvar.lat,plotvar,transform=proj,linewidths=0.75,
                      colors='w',levels=cints_rei,zorder=-2)
ax.clabel(cl)


for ir in range(nregs):
    rr = regplot[ir]
    rbbx = bboxes[rr]
    
    ls_in = rsty[rr]
    if ir == 2:
        ls_in = 'dashed'
    
    viz.plot_box(rbbx,color=rcols[rr],linestyle=ls_in,leglab=regions_long[rr],linewidth=2.5,return_line=True)


# Plot Gulf Stream Position
ax.plot(ds_gs2.lon.mean('mon'),ds_gs2.lat.mean('mon'),transform=proj,lw=2.5,c='k',ls='dashdot')

# Plot Ice Edge
ax.contour(icemask.lon,icemask.lat,mask_plot,colors="cyan",linewidths=2.5,
           transform=proj,levels=[0,1],zorder=-1)


cb = viz.hcbar(pcm,ax=ax,fraction=0.036)
cb.ax.tick_params(labelsize=fsz_tick)
cb.set_label("Re-emergence Index",fontsize=fsz_axis)

figname = "%sCESM1_Locator_MeanState_REM.png" % (figpath,)
plt.savefig(figname,dpi=150,bbox_inches='tight')

#%% Mini Locator

fsz_tick  = 14
bblocator = [-80,-20,30,65] 
# Initialize Plot and Map
fig,ax,_    = viz.init_orthomap(1,1,bblocator,figsize=(4,2),centlon=-50)
ax          = viz.add_coast_grid(ax,bbox=bblocator,fill_color="lightgray",fontsize=0)
#ax.set_title("CESM1 Historical Ens. Avg., Ann. Mean",fontsize=fsz_title)

# # Plot Currents
qint  = 2
plotu = ds_uvel.UVEL.mean('ens').mean('month').values
plotv = ds_vvel.VVEL.mean('ens').mean('month').values
ax.quiver(tlon[::qint,::qint],tlat[::qint,::qint],plotu[::qint,::qint],plotv[::qint,::qint],
          color='mediumblue',transform=proj,alpha=0.35,)#scale=1e3)

# Plot REI Index
plotvar = reiplot
pcm     = ax.contourf(plotvar.lon,plotvar.lat,plotvar,transform=proj,
                      cmap='cmo.deep',levels=cints_rei,zorder=-3)

for ir in range(nregs):
    rr = regplot[ir]
    rbbx = bboxes[rr]
    
    ls_in = rsty[rr]
    if ir == 2:
        ls_in = 'dashed'
    
    viz.plot_box(rbbx,color=rcols[rr],linestyle=ls_in,leglab=regions_long[rr],linewidth=1.5,return_line=True)

# # Plot Gulf Stream Position
ax.plot(ds_gs2.lon.mean('mon'),ds_gs2.lat.mean('mon'),transform=proj,lw=1.5,c='k',ls='dashdot')

# Plot Ice Edge
ax.contour(icemask.lon,icemask.lat,mask_plot,colors="cyan",linewidths=1.5,
           transform=proj,levels=[0,1],zorder=-1)

# cb = viz.hcbar(pcm,ax=ax,fraction=0.036)
# cb.ax.tick_params(labelsize=fsz_tick)
# cb.set_label("Re-emergence Index",fontsize=fsz_axis)

figname = "%sCESM1_Locator_MeanState_REM_Locator.png" % (figpath,)
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

#%% Visualize Month of the deepest MLD

#cmap = mpl.colors.ListedColormap([''])

plotvar   = ds_mld.argmax('mon') + 1
cints     = np.arange(0,13,1)

cmap = plt.colormaps['twilight']
norm = mpl.colors.BoundaryNorm(cints, ncolors=12, clip=True)

fig,ax,_  = viz.init_orthomap(1,1,bboxplot,figsize=(16,12),centlon=-40)
ax        = viz.add_coast_grid(ax,bbox=bboxplot,fill_color='lightgray')


# pcm = ax.pcolormesh(plotvar.lon,plotvar.lat,
#                     plotvar*mask,transform=proj,
#                     cmap=cmap,vmin=1,vmax=12)


pcm = ax.contourf(plotvar.lon,plotvar.lat,plotvar,transform=proj,
            cmap=cmap,levels=cints,zorder=-4)
cb  = viz.hcbar(pcm)

# cl = ax.contour(plotvar.lon,plotvar.lat,plotvar,transform=proj,
#             levels=cints,zorder=-3,colors='lightgray',linewidth=0.75)
# ax.clabel(cl,fontsize=fsz_tick)

# Plot Feb
# pcm = ax.contourf(plotvar.lon,plotvar.lat,plotvar,transform=proj,
#              colors='darkviolet',levels=[2,3],zorder=-4)
# cl = ax.contour(plotvar.lon,plotvar.lat,plotvar,transform=proj,
#             levels=[1,2,3,4],zorder=2,colors='lightgray',linewidth=0.75)
# ax.clabel(cl,fontsize=fsz_tick)

# Plot Jan
im = 1
# pcm = ax.contourf(plotvar.lon,plotvar.lat,plotvar,transform=proj,
#              cmap='jet',levels=[im,im+1,im+2],zorder=-4)
# cl = ax.contour(plotvar.lon,plotvar.lat,plotvar,transform=proj,
#             levels=[im,im+1,i+2],zorder=2,colors='lightgray',linewidth=0.75)
# ax.clabel(cl,fontsize=fsz_tick)

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







