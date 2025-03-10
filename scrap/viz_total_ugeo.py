#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Visualize Total Geostrophic Transport

Includes geostrophic transport plot used for Draft 04/05...

Copied from advection_sanity_Check on 2024.10.31

Created on Thu Oct 31 13:53:41 2024

@author: gliu
"""

import numpy as np
import xarray as xr
import sys
import time
import matplotlib.pyplot as plt
import matplotlib as mpl
import cartopy.crs as ccrs
import os

from tqdm import tqdm

from cmcrameri import cm
import matplotlib.patheffects as pe

# ----------------------------------
#%% Import custom modules and paths
# ----------------------------------

# Indicate the Machine!
machine = "Astraeus"

# First Load the Parameter File
sys.path.append("../")
import reemergence_params as rparams

# Paths and Load Modules
machine    = "Astraeus"
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
rawpath     = pathdict['raw_path']

# ------------------------------
#%% Set some plotting parameters
# ------------------------------

bboxplot                        = [-80,0,20,65]
mpl.rcParams['font.family']     = 'Avenir'
mons3                           = proc.get_monstr(nletters=3)

fsz_tick                        = 18
fsz_axis                        = 20
fsz_title                       = 16

rhocrit                         = proc.ttest_rho(0.05,2,86)
proj                            = ccrs.PlateCarree()


dtmon   = 3600*24*30


# Get Point Info
pointset    = "PaperDraft02"
ptdict      = rparams.point_sets[pointset]
ptcoords    = ptdict['bboxes']
ptnames     = ptdict['regions']
ptnames_long = ptdict['regions_long']
ptcols      = ptdict['rcols']
ptsty       = ptdict['rsty']

#%% Load Mask and other plotting variables


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


# ------------------------------------
#%%  Load Geostrophic Advection Terms
# ------------------------------------

vnames   = ["SST","SSS"]
ds_ugeos = []

for vv in range(2):
    vname           = vnames[vv]
    savename        = rawpath + "CESM1_HTR_FULL_Ugeo_%s_Transport_Full.nc" % (vname)
    ds              = xr.open_dataset(savename).load()
    ds_ugeos.append(ds)
    

# Just select 1 ensemble member
ds_ugeos_ens = [ds.isel(ens=1) for ds in ds_ugeos]

    
#% Compute Total Transport

ugeo_transport  = [ds.UET + ds.VNT for ds in ds_ugeos]


#%% Load additional variables to help

# Load the geostrophic currents
st               = time.time()
nc_ugeo          = "CESM1LE_ugeo_NAtl_19200101_20050101_bilinear.nc"
path_ugeo        = rawpath
ds_ugeo          = xr.open_dataset(path_ugeo + nc_ugeo).load()
print("Loaded ugeo in %.2fs" % (time.time()-st))

ds_ugeo_mean     = ds_ugeo.isel(ens=1).mean('time')

# Load time-mean SST and SSS
# Load data processed by [calc_monmean_CESM1.py]
ds_sss           = dl.load_monmean('SSS')
ds_sst           = dl.load_monmean('SST')

# Load stdev Ratio of experiments (from pointwise metrics)
metpath         = rawpath + "paper_metrics/"
ds_ratios       = xr.open_dataset(metpath + "Stdev_Ratios_AnnMean_SM_v_CESM.nc")


#%% Draft 04 --> Visualize the Standard Deviation of Geostrophic Advection


ugeo_transports   = [ds.UET + ds.VNT for ds in ds_ugeos]
ugeo_stdev_monvar = [ds.groupby('time.month').std('time') for ds in ugeo_transports]
ugeo_stdev        = [ds.std('time') for ds in ugeo_transports]

#%% Draft 04/05 Plot

#vmaxes       = [0.5,0.075]
lw_gs       = 3.5
plotcontour = False
plotadv     = False
plotratio   = False

vmaxes      = [.5,0.1]
vcints      = [np.arange(0,1.2,0.2),np.arange(0,0.28,0.04)]
vcmaps      = [cm.lajolla_r,cm.acton_r]
vunits      = ["\degree C","psu"]
pmesh       = True


fsz_axis    = 24
fsz_title   = 28
fsz_tick    = 16

titles      = [r"$\sigma(u_{geo} \cdot \nabla SST$)",r"$\sigma(u_{geo} \cdot \nabla SSS)$"]

mean_contours = [ds_sst.mean('ens').mean('mon').SST,ds_sss.mean('ens').mean('mon').SSS]
cints_sst   = np.arange(250,310,2)
cints_sss   = np.arange(34,37.6,0.2)
cints_mean   = [cints_sst,cints_sss]

fig,axs,_   = viz.init_orthomap(1,2,bboxplot,figsize=(20,10))
ii = 0
for vv in range(2):
    
    vmax    = vmaxes[vv]
    cmap    = vcmaps[vv]
    vunit   = vunits[vv]
    cints   = vcints[vv]
    
    ax      = axs[vv]
    ax      = viz.add_coast_grid(ax,bbox=bboxplot,fill_color="lightgray",fontsize=fsz_tick)
    
    ax.set_title(titles[vv],fontsize=fsz_title)
    
    plotvar = ugeo_stdev_monvar[vv].mean('ens').mean('month') * dtmon * mask
    
    if pmesh:
        pcm     = ax.pcolormesh(plotvar.lon,plotvar.lat,plotvar,
                                transform=proj,vmin=0,vmax=vmax,cmap=cmap)
    else:
        pcm     = ax.contourf(plotvar.lon,plotvar.lat,plotvar,levels=cints,
                                transform=proj,cmap=cmap,extend='both')
    
    cl      = ax.contour(plotvar.lon,plotvar.lat,plotvar,levels=cints,
                            transform=proj,colors="lightgray",lw=0.75)
    ax.clabel(cl,fontsize=fsz_tick)
    
    cb_lab = "[$%s$/mon]" % vunit
    
    cb = viz.hcbar(pcm,ax=ax,fraction=0.05,pad=0.01)
    cb.ax.tick_params(labelsize=fsz_tick)
    cb.set_label(cb_lab,fontsize=fsz_axis)
     
    # Add Other Features
    # Plot Gulf Stream Position
    ax.plot(ds_gs2.lon.mean('mon'),ds_gs2.lat.mean('mon'),transform=proj,lw=lw_gs,
            c="k",ls='dashdot',path_effects=[pe.Stroke(linewidth=6.5, foreground='deepskyblue'), pe.Normal()])#c=[0.15,]*3

    # Plot Ice Edge
    ax.contour(icemask.lon,icemask.lat,mask_plot,colors="cyan",linewidths=lw_gs,
               transform=proj,levels=[0,1],zorder=-1)
    
    
    nregs = len(ptnames)
    for ir in range(nregs):
        pxy   = ptcoords[ir]
        ax.plot(pxy[0],pxy[1],transform=proj,markersize=20,markeredgewidth=.5,c=ptcols[ir],
                marker='*',markeredgecolor='k')
    
    # Plot the Geostrophic Currents
    if plotadv:
        qint  = 2
        plotu = (ds_ugeo_mean.ug * mask).data #/ 100 * mask
        plotv = (ds_ugeo_mean.vg * mask).data #/ 100
        lonmesh,latmesh = np.meshgrid(ds_ugeo_mean.lon.data,ds_ugeo_mean.lat.data)
        ax.quiver(lonmesh[::qint,::qint],latmesh[::qint,::qint],plotu[::qint,::qint],plotv[::qint,::qint],
                  color='cornflowerblue',transform=proj,alpha=1,scale=5)
    if plotcontour:
        # Plot mean Contours
        plotvar = mean_contours[vv] * mask
        cl = ax.contour(plotvar.lon,plotvar.lat,plotvar,transform=proj,
                    linewidths=1.5,colors="dimgray",levels=cints_mean[vv],linestyles='dashed')
        ax.clabel(cl,fontsize=fsz_tick)
        
    if plotratio:
        plotvar     = ds_ratios[vnames[vv]]
        cints_ratio = np.arange(0,220,20)#np.array([0.01,0.25,0.50,1.0,1.5,2]) * 100#np.sort(np.append(cints,0))

        cl          = ax.contour(plotvar.lon,plotvar.lat,plotvar,transform=proj,
                    linewidths=3,levels=cints_ratio,cmap='cmo.balance',linestyles='dashed')
        ax.clabel(cl,fontsize=fsz_tick)
        
        
    
    
    viz.label_sp(ii,alpha=0.75,ax=ax,fontsize=fsz_title,x=0.05)
    ii += 1
    
    
savename = "%sUgeo_Contribution_Total.png" % figpath
plt.savefig(savename,dpi=150,bbox_inches='tight')

