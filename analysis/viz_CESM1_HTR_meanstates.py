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
    ax.plot(ds_gs2.lon.isel(mon=im),ds_gs2.lat.isel(mon=im),transform=proj,lw=1.75,c='k',ls='dashdot')
    #ax.plot(ds_gs.lon,ds_gs.lat.mean('ens'),transform=proj,lw=1.75,c="k")
    
    # # Plot Regional Bounding Boxes
    # for ir in range(nregs):
    #     rr = regplot[ir]
    #     ls_in = rsty[rr]
    #     if ir == 2:
    #         ls_in = 'dashed'
            
    #     rbbx = bboxes[rr]
    #     viz.plot_box(rbbx,color=rcols[rr],linestyle=ls_in,leglab=regions_long[rr],linewidth=2.5,return_line=True)


    
    
    figname = "%s%s_Current_Comparison_mon%02i.png" % (figpath,contourvar,im+1)
    plt.savefig(figname,dpi=150,bbox_inches='tight')
    
# ===========================
#%% Do Mean Version of above
# Submission 01
# ===========================

fsz_tick    = 14
qint        = 2
plot_point  = True
pmesh       = False


cints_sst_degC  = np.arange(250,310,2) - 273.15

# Get Bounding Boxes
regionset       = "SSSCSU"
regiondicts     = rparams.region_sets[regionset]
bboxes          = regiondicts['bboxes']
regions_long    = regiondicts['regions_long']
rcols           = regiondicts['rcols']
rsty            = regiondicts['rsty']

# Get Point Info
pointset        = "PaperDraft02"
ptdict          = rparams.point_sets[pointset]
ptcoords        = ptdict['bboxes']
ptnames         = ptdict['regions']
ptnames_long    = ptdict['regions_long']
ptcols          = ptdict['rcols']
ptsty           = ptdict['rsty']

cints_sssmean = np.arange(34,37.6,0.2)

regplot = [0,1,3]
nregs   = len(regplot)

# Initialize Plot and Map
fig,ax,_    = viz.init_orthomap(1,1,bboxplot,figsize=(24,6.5))
ax          = viz.add_coast_grid(ax,bbox=bboxplot,fill_color="lightgray",fontsize=fsz_tick)
#ax.set_title("CESM1 Historical Ens. Avg., Ann. Mean",fontsize=fsz_title)

# Plot Currents
plotu = ds_uvel.UVEL.mean('ens').mean('month').values
plotv = ds_vvel.VVEL.mean('ens').mean('month').values
ax.quiver(tlon[::qint,::qint],tlat[::qint,::qint],plotu[::qint,::qint],plotv[::qint,::qint],
          color='darkslateblue',transform=proj,alpha=0.75)

# Plot Mean SST (Colors)
plotvar = ds_sst.SST.mean('ens').mean('mon').transpose('lat','lon') - 273.15 * mask_apply
if pmesh:
    pcm     = ax.pcolormesh(plotvar.lon,plotvar.lat,plotvar,transform=proj,zorder=-1,
                linewidths=1.5,cmap="RdYlBu_r",vmin=280,vmax=300)
else:
    cints_sstmean = np.arange(280,301,1)
    cints_sstmean_degC = np.arange(5,31)#np.arange(280,301,1) - 273.15
    
    pcm     = ax.contourf(plotvar.lon,plotvar.lat,plotvar,transform=proj,zorder=-1,
                cmap="RdYlBu_r",levels=cints_sstmean_degC,extend='both')
cb = viz.hcbar(pcm,ax=ax,fraction=0.045)
cb.set_label("SST ($\degree C$)",fontsize=fsz_axis)
cb.ax.tick_params(labelsize=fsz_tick)

# Plot Mean SSS (Contours)
plotvar = ds_sss.SSS.mean('ens').mean('mon').transpose('lat','lon') * mask_reg
cl = ax.contour(plotvar.lon,plotvar.lat,plotvar,transform=proj,
            linewidths=1.5,colors="darkviolet",levels=cints_sssmean,linestyles='dashed',zorder=1)
ax.clabel(cl,fontsize=fsz_tick)

if plot_point:
    nregs = len(ptnames)
    for ir in range(nregs):
        pxy   = ptcoords[ir]
        ax.plot(pxy[0],pxy[1],transform=proj,markersize=20,markeredgewidth=.5,c=ptcols[ir],
                marker='*',markeredgecolor='k',zorder=4)
else:
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
plt.savefig(figname,dpi=200,bbox_inches='tight')

#%% Same Plot as above, but focus on SSS re-emergence (locator for TCM presentation)

fsz_tick    = 14

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
cb.ax.tick_params(labelsize=fsz_tick)

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

vv = 0

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

bbsar               = [-80,-40,30,50]
#cints_sst_sar   = np.arange(285,305,0.5)
cints_sst_sar       = np.arange(290,300.5,0.5)
cints_sst_sar_full  = np.arange(285,305.5,0.5)
vlims_umod_sar      = [0,0.5]
qint                = 1
scale               = 4

cints_sss_sar        = np.arange(35,36.8,0.1)
cints_sss_sar_full   = np.arange(34,37.1,0.1)

# Toggles
plot_sst_fill   = True # True to Plot ContourFill SST instead of |Uvel|
plot_sss        = True # Set to True to plot the SSS Contorfill

for im in range(12):
    fig,ax,_    = viz.init_orthomap(1,1,bbsar,figsize=(18,6),centlon=-60,centlat=40)
    ax          = viz.add_coast_grid(ax,bbox=bbsar,fill_color="lightgray",fontsize=fsz_tick)
    
    
    # Plot Currents (divide by 100 to get cm)
    plotu = ds_uvel.isel(month=im).UVEL.mean('ens').values#/100
    plotv = ds_vvel.isel(month=im).VVEL.mean('ens').values#/100
    qv    = ax.quiver(tlon[::qint,::qint],tlat[::qint,::qint],plotu[::qint,::qint],plotv[::qint,::qint],
              color='navy',transform=proj,alpha=0.75,units='width',scale=scale)
    
    # # Add Quiverkey
    qk  = ax.quiverkey(qv,.0,0.85,0.5,r"0.5 $\frac{m}{s}$",fontproperties=dict(size=fsz_tick))
    
    # Plot Sargasso Bounding Box
    for ir in range(nregs):
        rr = regplot[ir]
        ls_in = rsty[rr]
        if ir == 2:
            ls_in = 'dashed'
            
        rbbx = bboxes[rr]
        viz.plot_box(rbbx,color=rcols[rr],linestyle=ls_in,leglab=regions_long[rr],linewidth=2.5,return_line=True)
    
    
    # Plot Gulf Stream Position
    ax.plot(ds_gs2.lon.mean('mon'),ds_gs2.lat.mean('mon'),transform=proj,lw=2.5,c='firebrick',ls='dashdot')
    
    # Set Title based on month
    ax.set_title("%s Mean Currents (Vectors) and SST Gradients (Contours)" % mons3[im],fontsize=fsz_axis)
    
    # Plot SST Contours
    if plot_sss:
        plotvar = ds_sss.SSS.mean('ens').isel(mon=im).transpose('lat','lon') * mask_reg
        if plot_sst_fill:
            pcm      = ax.contourf(plotvar.lon,plotvar.lat,plotvar,transform=proj,
                        linewidths=.75,cmap="cmo.haline",levels=cints_sss_sar,linestyles='dashed',zorder=-4)
            
            filllab  = "SSS ($psu$)"
        
        cl      = ax.contour(plotvar.lon,plotvar.lat,plotvar,transform=proj,
                    linewidths=.75,colors="k",levels=cints_sss_sar_full,linestyles='dashed')
        ax.clabel(cl,fontsize=fsz_tick-2)
    
    else:
        plotvar = ds_sst.SST.mean('ens').isel(mon=im).transpose('lat','lon') * mask_reg
        if plot_sst_fill:
            pcm      = ax.contourf(plotvar.lon,plotvar.lat,plotvar,transform=proj,
                        linewidths=.75,cmap="cmo.thermal",levels=cints_sst_sar,linestyles='dashed',zorder=-4)
            
            filllab  = "SST ($\degree$C)"
        
        cl      = ax.contour(plotvar.lon,plotvar.lat,plotvar,transform=proj,
                    linewidths=.75,colors="k",levels=cints_sst_sar_full,linestyles='dashed')
        ax.clabel(cl,fontsize=fsz_tick-2)
    
    
    # # Plot Mean Velocity (Colors)
    if plot_sst_fill is False:
        plotvar = ds_umod.isel(month=im).mean('ens') #* mask_apply
        pcm     = ax.pcolormesh(tlon,tlat,plotvar,transform=proj,zorder=-1,
                    linewidths=1.5,cmap="cmo.ice_r",vmin=vlims_umod_sar[0],vmax=vlims_umod_sar[1])
        filllab = "Surface Current [m/s]"
    cb = viz.hcbar(pcm,ax=ax,fraction=0.045)
    cb.set_label(filllab,fontsize=fsz_tick)
    cb.ax.tick_params(labelsize=12)
    
    savename = "%sSargasso_Sea_Mean_Currents_plotsst%i_plotsss%i_mon%02i.png" % (figpath,plot_sst_fill,plot_sss,im+1)
    plt.savefig(savename,dpi=150,bbox_inches='tight')



#%%
# Plot Contours of the Amplitude/Velocity



# What should we plot... I guess we should plot the mean currents for that month



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


#%% Investigate what is happening in the sargasso sea (why is there a minima in in Aug-Sep variance?)


fig,ax = plt.subplots

#%% To Do
"""

- Make some regional visualizations
- Add contours of TEMP and SALT

"""


# ----------------------------------------------
#%% Irminger Sea (copied from above visualizations of regional stuff)
# ----------------------------------------------
lonirm = -36
latirm = 58

fsz_ticks = 16
bboxirm   = [-60,-20,45,70]
fig,ax,_  = viz.init_orthomap(1,1,bboxirm,figsize=(25,14),centlon=-40)
ax        = viz.add_coast_grid(ax,bbox=bboxirm,fill_color='lightgray')
cints_rei_2 = np.arange(0,1.05,0.02)

regmaskr   = proc.sel_region_xr(mask,bboxirm)

# # Plot Mean SSS (Contours)
# cints_sss_ice = np.arange(30,36.4,0.4)
# plotvar = ds_sss.SSS.mean('ens').isel(mon=im).transpose('lat','lon') #* mask_apply
# cl = ax.contour(plotvar.lon,plotvar.lat,plotvar,transform=proj,
#             linewidths=1.5,colors="firebrick",levels=cints_sss_ice,linestyles='solid')
# ax.clabel(cl,fontsize=fsz_ticks)

# Plot Currents
qint  = 1
plotu = ds_uvel.UVEL.mean('ens').mean('month').values
plotv = ds_vvel.VVEL.mean('ens').mean('month').values
ax.quiver(tlon[::qint,::qint],tlat[::qint,::qint],plotu[::qint,::qint],plotv[::qint,::qint],
          color='navy',transform=proj,alpha=0.75)


# Plot REI Index (SST)
plotvar = reiplot_sst #* regmaskr
cl      = ax.contour(plotvar.lon,plotvar.lat,plotvar,transform=proj,linewidths=0.75,
                      colors='lightgray',levels=cints_rei_2,zorder=-2)
ax.clabel(cl,fontsize=fsz_tick)

pcm     = ax.pcolormesh(plotvar.lon,plotvar.lat,plotvar,transform=proj,
                      cmap='cmo.dense',vmin=0,vmax=0.55,zorder=-3)

cb = viz.hcbar(pcm)
cb.ax.tick_params(labelsize=fsz_tick)
cb.set_label("REI (SST)",fontsize=fsz_axis)

# Select A Point
plotvar = proc.sel_region_xr(plotvar,bboxirm)
for (i, j), z in np.ndenumerate(plotvar):
    ax.text(plotvar.lon.data[j], plotvar.lat.data[i], '{:0.3f}'.format(z),
            ha='center', va='center',transform=proj,fontsize=9,color='w',zorder=-1)

# Plot the Point
ax.plot(lonirm,latirm,marker="x",color="y",transform=proj,markersize=25)

# Plot Bounding Box
ir = 2
rr = regplot[ir]
rbbx = bboxes[rr]
ls_in = rsty[rr]
if ir == 2:
    ls_in = 'dashed'
viz.plot_box(rbbx,ax=ax,color=rcols[rr],linestyle=ls_in,leglab=regions_long[rr],linewidth=4,return_line=True,proj=proj)

# Plot Re-emergence Index for SST
# # Plot Gulf Stream Position
ax.plot(ds_gs2.lon.mean('mon'),ds_gs2.lat.mean('mon'),transform=proj,lw=4,c='k',ls='dashdot')

# Plot Ice Edge
ax.contour(icemask.lon,icemask.lat,mask_plot,colors="cyan",linewidths=4,
           transform=proj,levels=[0,1],zorder=-1)

figname = "%sIRM_Locator_Point.png" % (figpath)
plt.savefig(figname,dpi=150,bbox_inches='tight')


# ----------------------------------------------
#%% North Atlantic Current (NAC)
# ----------------------------------------------
lonnac = -34
latnac = 46

fsz_ticks = 16
bbox_nac  = [-60,-10,35,55]
bboxin    = bbox_nac#[-60,-20,45,70]
fig,ax,_  = viz.init_orthomap(1,1,bboxin,figsize=(25,14),centlon=-35)
ax        = viz.add_coast_grid(ax,bbox=bboxin,fill_color='lightgray')
cints_rei_2 = np.arange(0,1.05,0.02)

regmaskr   = proc.sel_region_xr(mask,bboxin)

# # Plot Mean SSS (Contours)
# cints_sss_ice = np.arange(30,36.4,0.4)
# plotvar = ds_sss.SSS.mean('ens').isel(mon=im).transpose('lat','lon') #* mask_apply
# cl = ax.contour(plotvar.lon,plotvar.lat,plotvar,transform=proj,
#             linewidths=1.5,colors="firebrick",levels=cints_sss_ice,linestyles='solid')
# ax.clabel(cl,fontsize=fsz_ticks)

# Plot Currents
qint  = 1
plotu = ds_uvel.UVEL.mean('ens').mean('month').values
plotv = ds_vvel.VVEL.mean('ens').mean('month').values
ax.quiver(tlon[::qint,::qint],tlat[::qint,::qint],plotu[::qint,::qint],plotv[::qint,::qint],
          color='navy',transform=proj,alpha=0.75)

# Plot REI Index (SST)
plotvar = reiplot_sss #* regmaskr
cl      = ax.contour(plotvar.lon,plotvar.lat,plotvar,transform=proj,linewidths=0.75,
                      colors='k',levels=cints_rei_2,zorder=-2)
ax.clabel(cl,fontsize=fsz_tick)

pcm     = ax.pcolormesh(plotvar.lon,plotvar.lat,plotvar,transform=proj,
                      cmap='cmo.deep',vmin=0,vmax=0.55,zorder=-3)

cb = viz.hcbar(pcm)
cb.ax.tick_params(labelsize=fsz_tick)
cb.set_label("REI (SST)",fontsize=fsz_axis)

# Select A Point
plotvar = proc.sel_region_xr(plotvar,bboxin)
print("REI is %f" % (plotvar.sel(lon=lonnac,lat=latnac,method='nearest')))
for (i, j), z in np.ndenumerate(plotvar):
    ax.text(plotvar.lon.data[j], plotvar.lat.data[i], '{:0.3f}'.format(z),
            ha='center', va='center',transform=proj,fontsize=9,color='k',zorder=1,path_effects=[pe.withStroke(linewidth=1.5, foreground="w")])

# Plot the Point
ax.plot(lonnac,latnac,marker="x",color="y",transform=proj,markersize=25)

# Plot Bounding Box
ir = 1
rr = regplot[ir]
rbbx = bboxes[rr]
ls_in = rsty[rr]
if ir == 2:
    ls_in = 'dashed'
viz.plot_box(rbbx,ax=ax,color=rcols[rr],linestyle=ls_in,leglab=regions_long[rr],linewidth=4,return_line=True,proj=proj)

# Plot Re-emergence Index for SST
# # Plot Gulf Stream Position
ax.plot(ds_gs2.lon.mean('mon'),ds_gs2.lat.mean('mon'),transform=proj,lw=4,c='k',ls='dashdot')

# Plot Ice Edge
ax.contour(icemask.lon,icemask.lat,mask_plot,colors="cyan",linewidths=4,
           transform=proj,levels=[0,1],zorder=-1)

figname = "%sSAR_Locator_Point.png" % (figpath)
plt.savefig(figname,dpi=150,bbox_inches='tight')



# ----------------------------------------------
#%% Sargasso Sea (SAR)
# ----------------------------------------------
lonsar = -65
latsar = 36

fsz_ticks = 16
bbox_sar  = [-80,-40,30,50] 
bboxin    = bbox_sar#[-60,-20,45,70]
fig,ax,_  = viz.init_orthomap(1,1,bboxin,figsize=(25,14),centlon=-60)
ax        = viz.add_coast_grid(ax,bbox=bboxin,fill_color='lightgray')
cints_rei_2 = np.arange(0,1.05,0.02)

regmaskr   = proc.sel_region_xr(mask,bboxin)

# # Plot Mean SSS (Contours)
# cints_sss_ice = np.arange(30,36.4,0.4)
# plotvar = ds_sss.SSS.mean('ens').isel(mon=im).transpose('lat','lon') #* mask_apply
# cl = ax.contour(plotvar.lon,plotvar.lat,plotvar,transform=proj,
#             linewidths=1.5,colors="firebrick",levels=cints_sss_ice,linestyles='solid')
# ax.clabel(cl,fontsize=fsz_ticks)

# Plot Currents
qint  = 1
plotu = ds_uvel.UVEL.mean('ens').mean('month').values
plotv = ds_vvel.VVEL.mean('ens').mean('month').values
ax.quiver(tlon[::qint,::qint],tlat[::qint,::qint],plotu[::qint,::qint],plotv[::qint,::qint],
          color='navy',transform=proj,alpha=0.75)


# Plot REI Index (SST)
plotvar = reiplot_sss #* regmaskr
cl      = ax.contour(plotvar.lon,plotvar.lat,plotvar,transform=proj,linewidths=0.75,
                      colors='k',levels=cints_rei_2,zorder=-2)
ax.clabel(cl,fontsize=fsz_tick)

pcm     = ax.pcolormesh(plotvar.lon,plotvar.lat,plotvar,transform=proj,
                      cmap='cmo.deep',vmin=0,vmax=0.55,zorder=-3)

cb = viz.hcbar(pcm)
cb.ax.tick_params(labelsize=fsz_tick)
cb.set_label("REI (SST)",fontsize=fsz_axis)

# Select A Point
plotvar = proc.sel_region_xr(plotvar,bboxin)
print("REI is %f" % (plotvar.sel(lon=lonsar,lat=latsar,method='nearest')))
for (i, j), z in np.ndenumerate(plotvar):
    ax.text(plotvar.lon.data[j], plotvar.lat.data[i], '{:0.3f}'.format(z),
            ha='center', va='center',transform=proj,fontsize=9,color='k',zorder=1,)#path_effects=[pe.withStroke(linewidth=1.5, foreground="w")])

# Plot the Point
ax.plot(lonsar,latsar,marker="x",color="y",transform=proj,markersize=25)

# Plot Bounding Box
ir = 0
rr = regplot[ir]
rbbx = bboxes[rr]
ls_in = rsty[rr]
if ir == 2:
    ls_in = 'dashed'
viz.plot_box(rbbx,ax=ax,color=rcols[rr],linestyle=ls_in,leglab=regions_long[rr],linewidth=4,return_line=True,proj=proj)

# Plot Re-emergence Index for SST
# # Plot Gulf Stream Position
ax.plot(ds_gs2.lon.mean('mon'),ds_gs2.lat.mean('mon'),transform=proj,lw=4,c='k',ls='dashdot')

# Plot Ice Edge
ax.contour(icemask.lon,icemask.lat,mask_plot,colors="cyan",linewidths=4,
           transform=proj,levels=[0,1],zorder=-1)

figname = "%sSAR_Locator_Point.png" % (figpath)
plt.savefig(figname,dpi=150,bbox_inches='tight')

#%% Visualize Subpolar North Atlantic

# bboxice  = 
# fig,ax,_ = viz.init_orthomap(1,1,bboxice,figsize=(12,4))
# im = 1xs







