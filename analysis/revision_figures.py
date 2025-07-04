#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Revision Figures

Figures for revision of the SSS Paper
Created on Thu Mar  6 08:59:37 2025

@author: gliu

"""

import numpy as np
import xarray as xr
import sys
import time
import matplotlib.pyplot as plt
import tqdm
import os
import matplotlib as mpl
import cartopy.crs as ccrs

# ----------------------------------
#%% Import custom modules and paths
# ----------------------------------

# Indicate the Machine!
machine = "Astraeus"

# First Load the Parameter File
sys.path.append("../")
cwd = os.getcwd()
sys.path.append(cwd+ "/..")
import reemergence_params as rparams

# Paths and Load Modules
pathdict   = rparams.machine_paths[machine]

sys.path.append(pathdict['amvpath'])
sys.path.append(pathdict['scmpath'])
from amv import proc,viz
import scm
import amv.loaders as dl
import yo_box as ybx

# Set needed paths
figpath     = pathdict['figpath']
proc.makedir(figpath)
input_path  = pathdict['input_path']
output_path = pathdict['output_path']
procpath    = pathdict['procpath']


#%% Indicate Paths

#figpath   = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/02_Figures/20240411/"
datpath   = pathdict['raw_path']
outpath   = pathdict['input_path']+"forcing/"
rawpath   = pathdict['raw_path']

revpath   = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/revision_data/"


#%% Indicate some plotting variables

darkmode = False
if darkmode:
    dfcol = "w"
    bgcol = np.array([15,15,15])/256
    transparent = True
    plt.style.use('dark_background')
    mpl.rcParams['font.family']     = 'Avenir'
else:
    dfcol = "k"
    bgcol = "w"
    transparent = False
    plt.style.use('default')


bboxplot                        = [-80,0,20,65]
mpl.rcParams['font.family']     = 'Avenir'
mons3                           = proc.get_monstr(nletters=3)

fsz_tick                        = 18
fsz_axis                        = 20
fsz_title                       = 32

rhocrit                         = proc.ttest_rho(0.05,2,86)
proj                            = ccrs.PlateCarree()

# Get Region Info
regionset = "SSSCSU"
rdict                       = rparams.region_sets[regionset]
regions                     = rdict['regions']
bboxes                      = rdict['bboxes']
rcols                       = rdict['rcols']
rsty                        = rdict['rsty']
regions_long                = rdict['regions_long']
nregs                       = len(bboxes)

regions_long = ('Sargasso Sea', 'N. Atl. Current',  'Irminger Sea')

# Get Point Info
pointset        = "PaperDraft02"
ptdict          = rparams.point_sets[pointset]
ptcoords        = ptdict['bboxes']
ptnames         = ptdict['regions']
ptnames_long    = ptdict['regions_long']
ptcols          = ptdict['rcols']
ptsty           = ptdict['rsty']


#%% Load Land Ice Mask

# Load the currents
ds_uvel,ds_vvel = dl.load_current()
ds_bsf          = dl.load_bsf(ensavg=False)
ds_ssh          = dl.load_bsf(ensavg=False,ssh=True)

# Convert Currents to m/sec instead of cmsec
ds_uvel = ds_uvel/100
ds_vvel = ds_vvel/100
tlon  = ds_uvel.TLONG.mean('ens').values
tlat  = ds_uvel.TLAT.mean('ens').values

# Load Land Ice Mask
icemask    = xr.open_dataset(input_path + "masks/CESM1LE_HTR_limask_pacificmask_enssum_lon-90to20_lat0to90.nc")
mask       = icemask.MASK.squeeze()
mask_plot  = xr.where(np.isnan(mask),0,mask)#mask.copy()

mask_apply = icemask.MASK.squeeze().values

# Load data processed by [calc_monmean_CESM1.py]
ds_sss          = dl.load_monmean('SSS')
ds_sst          = dl.load_monmean('SST')


# Get A region mask
mask_reg_sub    = proc.sel_region_xr(mask,bboxplot)
mask_reg_ori    = xr.ones_like(mask) * 0
mask_reg        = mask_reg_ori + mask_reg_sub

ds_gs2          = dl.load_gs(load_u2=True)


# ===================================
#%% Figure R1: Number of Modes Needed
# ===================================

# Data Source: /preprocessing/correct_EOF_forcing_SSS.py

fname           = "EOF_Number_Modes_Needed_90pct_Fprime.npy"
nmodes_needed   = np.load(revpath + fname)
eof_thres       = 0.90
mons3           = proc.get_monstr()

fig,ax          = viz.init_monplot(1,1,figsize=(6,4.5))

ax.bar(mons3,nmodes_needed,alpha=0.5,color='darkred',edgecolor="k")
ax.set_xlim([-1,12])
ax.set_title("Number of Modes \n Needed to Explain %.2f" % (eof_thres*100) + "% of Variance",fontsize=16)
ax.set_ylabel("Number of Modes",fontsize=14)
ax.set_xlabel("Month",fontsize=14)
ax.bar_label(ax.containers[0],label_type='center',c='k')

savename        = "%sEOF_Nmodes_Needed.png" % figpath
plt.savefig(savename,dpi=150,bbox_inches='tight')




# =============================================================================
#%% Plot Mean States of the Currents, and the mixed-layer depth
# =============================================================================

# Process Mean SST and SSS
sst_mean = ds_sst.SST.mean('ens').mean('mon').transpose('lat','lon')  - 273.15 #  Covert to Celsius
sss_mean = ds_sss.SSS.mean('ens').mean('mon').transpose('lat','lon') 

# Set some colorbar limits
cints_sstmean_degC  = np.arange(6,27)
cints_sssmean       = np.arange(34,37.6,0.2)

# Load the mixed-layer depth
mldpath     = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/proc/model_input/mld/"
ncmld       = mldpath + "CESM1_HTR_FULL_HMXL_NAtl_EnsAvg.nc"
ds_mld      = xr.open_dataset(ncmld).load()

# Set plotting options for mld
vlms        = [0,200]# None
cints_sp    = np.arange(200,1500,100)# None
cmap_mld    = 'cmo.dense'
vlabel      = "Max Seasonal Mixed Layer Depth (meters)"

#%% Do some plotting
2
fsz_axis        = 24
fsz_title       = 28
fsz_tick        = 22
fsz_leg         = 18
lw_plot         = 4.5
use_star        = True
qint            = 2
plot_point      = True

fig,axs,_       = viz.init_orthomap(1,2,bboxplot,figsize=(20,10))
ii              = 0

for vv in range(2):
    
    ax      = axs[vv]
    ax      = viz.add_coast_grid(ax,bbox=bboxplot,fill_color="lightgray",fontsize=fsz_tick)
    
    
    # Plot Gulf Stream Position
    #ax.plot(ds_gs.lon,ds_gs.lat.mean('ens'),transform=proj,lw=1.75,c="k",ls='dashed')
    ax.plot(ds_gs2.lon.mean('mon'),ds_gs2.lat.mean('mon'),transform=proj,lw=2.5,c='k',ls='dashdot',zorder=1)

    # Plot Ice Edge
    ax.contour(icemask.lon,icemask.lat,mask_plot,colors="cyan",linewidths=2.5,
               transform=proj,levels=[0,1],zorder=1)
    
    # Label Subplot
    viz.label_sp(ii,alpha=0.15,ax=ax,fontsize=fsz_title,y=1.08,x=-.02,
                 fontcolor=dfcol)
    ii +=1


# (1) Mean State Plot ---------------------------------------------------------
ax                 = axs[0]

# Plot SST
plotvar            = sst_mean * mask_reg
pcm                = ax.contourf(plotvar.lon,plotvar.lat,plotvar,transform=proj,zorder=-1,
            cmap="RdYlBu_r",levels=cints_sstmean_degC,extend='both')
cb = viz.hcbar(pcm,ax=ax,fraction=0.045)
cb.set_label("SST ($\degree C$)",fontsize=fsz_axis)
cb.ax.tick_params(labelsize=fsz_tick)

# Plot SSS
plotvar            = sss_mean * mask_reg
cl = ax.contour(plotvar.lon,plotvar.lat,plotvar,transform=proj,
            linewidths=1.5,colors="darkviolet",levels=cints_sssmean,linestyles='dashed',zorder=1)
ax.clabel(cl,fontsize=fsz_tick)

# Plot Currents
plotu = ds_uvel.UVEL.mean('ens').mean('month').values
plotv = ds_vvel.VVEL.mean('ens').mean('month').values
ax.quiver(tlon[::qint,::qint],tlat[::qint,::qint],plotu[::qint,::qint],plotv[::qint,::qint],
          color='gray',transform=proj,alpha=0.55,zorder=-1)

if plot_point:
    nregs = len(ptnames)
    for ir in range(nregs):
        pxy   = ptcoords[ir]
        ax.plot(pxy[0],pxy[1],transform=proj,markersize=20,markeredgewidth=.5,c=ptcols[ir],
                marker='*',markeredgecolor='k',zorder=4)
        





# (2) Wintertime MLD Plot -----------------------------------------------------
ax = axs[1]




plot_dots = True
# fsz_tick = 26
# fsz_axis = 32


# Plot maximum MLD
plotvar     = ds_mld.h.max('mon') * mask_reg


# Just Plot the contour with a colorbar for each one
if vlms is None:
    pcm = ax.pcolormesh(plotvar.lon,plotvar.lat,plotvar,transform=proj,cmap=cmap_mld,zorder=-1)
    cb = fig.colorbar(pcm,ax=ax)
else:
    pcm = ax.pcolormesh(plotvar.lon,plotvar.lat,plotvar,transform=proj,
                        cmap=cmap_mld,vmin=vlms[0],vmax=vlms[1],zorder=-4)
    
# Do special contours
if cints_sp is not None:
    cl = ax.contour(plotvar.lon,plotvar.lat,plotvar,transform=proj,
                    levels=cints_sp,colors="w",linewidths=1.1,zorder=2)
    ax.clabel(cl,fontsize=fsz_tick,zorder=6)

if plot_dots:
    bboxr       = proc.get_bbox(mask_reg)
    hreg        = proc.sel_region_xr(ds_mld.h,bboxr)
    hmax        = hreg.argmax('mon').values
    # Special plot for MLD (mark month of maximum)
    hmask_feb  = (hmax == 1) * mask_reg#ds_mask # Note quite a fix, as 0. points will be rerouted to april
    hmask_mar  = (hmax == 2) * mask_reg #ds_mask
    
    smap = viz.plot_mask(hmask_feb.lon,hmask_feb.lat,hmask_feb.T,reverse=True,
                          color="mediumseagreen",markersize=4,marker="+",
                          ax=ax,proj=proj,geoaxes=True)
    
    smap = viz.plot_mask(hmask_mar.lon,hmask_mar.lat,hmask_mar.T,reverse=True,
                          color="palegoldenrod",markersize=3,marker="o",
                          ax=ax,proj=proj,geoaxes=True)
    
    ax.scatter(-100,-100,c="mediumseagreen",s=125,marker="+",label="February Max",transform=proj)
    ax.scatter(-100,-100,c="palegoldenrod",s=125,marker="o",label="March Max",transform=proj)
    
    leg = ax.legend(fontsize=fsz_leg,framealpha=0.85,
                    bbox_to_anchor=(.35,1.08), loc="upper left",  bbox_transform=ax.transAxes)
    leg.get_frame().set_linewidth(0.0)
    
if vlms is not None:
    cb = viz.hcbar(pcm,ax=ax,fraction=0.045)
    cb.set_label(vlabel,fontsize=fsz_axis)
    cb.ax.tick_params(labelsize=fsz_tick)
    
    
savename        = "%sCESM_MeanState_Winter_MLD.png" % figpath
plt.savefig(savename,dpi=150,bbox_inches='tight')



# =============================================================================
#%% Fig. ##: Plot Difference By Lag
# =============================================================================
# Copied calculations + plotting from visualize_rei_acf.py
selmon = [1,2]
compare_name = "RevisionD1" # PaperDraft01

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
    
#ds_diff = xr.merge(ds_diff)


    

#%% Plot the Figure


lagmeans_byvar = ds_diff#[ds_diff[vn] for vn in vnames]
lagrangenames  = ds_diff[0].lag_range.data
kmonths     = [1,2]
vv          = 0
fsz_title   = 30
fsz_axis    = 24
fsz_tick    = 20
plot_point  = True
drop_col3   = True

lon         = lagmeans_byvar[0].lon
lat         = lagmeans_byvar[0].lat

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
    
savename = "%sDiff_bylag_RFScript.png" % (figpath)
plt.savefig(savename,dpi=200,bbox_inches='tight',)#transparent=True)
