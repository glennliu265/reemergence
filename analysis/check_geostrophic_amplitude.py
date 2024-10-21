#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Check roughly the order of magnitude for the geostrophic components

Created on Tue Oct 15 22:29:57 2024

@author: gliu
"""
import xarray as xr
import numpy as np
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import cartopy.crs as ccrs
import scipy as sp

import matplotlib.pyplot as plt
import sys
import glob
import os

import tqdm
import time

from cmcrameri import cm

#%% Import Custom Modules

# Import AMV Calculation
from amv import proc,viz
import amv.loaders as dl

# Import stochastic model scripts
import scm

# ----------------------------------
# %% Import custom modules and paths
# ----------------------------------

# Import re-eergemce parameters

# Indicate the Machine!
machine     = "Astraeus"

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

proc.makedir(figpath)


#%%

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

# Get A region mask
mask_reg_sub    = proc.sel_region_xr(mask,bboxplot)
mask_reg_ori    = xr.ones_like(mask) * 0
mask_reg        = mask_reg_ori + mask_reg_sub

ds_gs2          = dl.load_gs(load_u2=True)


# #%%  Indicate Experients (copying upper setion of viz_regional_spectra )

# regionset       = "SSSCSU"
# comparename     = "SST_SSS_Paper_Draft01_Original"
# expnames        = ["SST_CESM","SSS_CESM","SST_Draft03_Rerun_QekCorr","SSS_Draft03_Rerun_QekCorr"]
# expnames_long   = ["CESM1 (SST)","CESM1 (SSS)","Stochastic Model (SST)","Stochastic Model (SSS)"]
# expnames_short  = ["CESM1_SST","CESM1_SSS","SM_SST","SM_SSS"]
# ecols           = ["firebrick","navy","hotpink",'cornflowerblue']
# els             = ["solid","solid",'dashed','dashed']
# emarkers        = ["o","d","o","d"]


# cesm_exps       = ["SST_CESM","SSS_CESM","SST_cesm2_pic","SST_cesm1_pic",
#                   "SST_cesm1le_5degbilinear","SSS_cesm1le_5degbilinear",]
# #%% Load the stochastic model output (using sm output loader)
# # Hopefully this doesn't clog up the memory too much

# nexps = len(expnames)
# ds_all = []
# for e in tqdm.tqdm(range(nexps)):
    
#     # Get Experiment information
#     expname        = expnames[e]
    
#     if "SSS" in expname:
#         varname = "SSS"
#     elif "SST" in expname:
#         varname = "SST"
    
#     # For stochastic model output
#     ds = dl.load_smoutput(expname,output_path)
    
#     if expname in cesm_exps:
#         print("Detrending and deseasoning")
#         ds = proc.xrdeseason(ds[varname])
#         if 'ens' in list(ds.dims):
#             ds = ds - ds.mean('ens')
#         else:
#             ds = proc.xrdetrend(ds)
#         ds = xr.where(np.isnan(ds),0,ds) # Sub with zeros for now
#     else:
#         ds = ds[varname]
    
#     ds_all.append(ds)
    
# #%% Compute Monthly Variance First, then take the regional average

# ds_all_monvar = [ds.groupby('time.month').var('time') for ds in ds_all]

#%% Load the ugeo terms 

st            = time.time()
ugeoprime     = xr.open_dataset(rawpath + 'ugeoprime_gradT_gradS_NATL.nc').load()
ugeobar       = xr.open_dataset(rawpath + 'ugeobar_gradTprime_gradSprime_NATL.nc').load()

ugeores_T     = xr.open_dataset(rawpath + "CESM1_HTR_FULL_Ugeoprime_SSTprime_Transport_Full.nc").load()
ugeores_S     = xr.open_dataset(rawpath + "CESM1_HTR_FULL_Ugeoprime_SSSprime_Transport_Full.nc").load()
print("Loaded files in %.2fs" % (time.time()-st))

terms_T       = [ugeobar.SST,ugeoprime.SST,ugeores_T.ug_dTdx + ugeores_T.vg_dTdy]
terms_S       = [ugeobar.SSS,ugeoprime.SSS,ugeores_S.ug_dTdx + ugeores_S.vg_dTdy]

#%% First, let's examine the total standard deviation

std_T = [ds.std('time').mean('ens') for ds in terms_T]
std_S = [ds.std('time').mean('ens') for ds in terms_S]

#%% Now do a quick plot

dtmon           = 3600*24*30
plotnames       = ["Ubar","Uprime","Residual"]

fig,axsall      = plt.subplots(2,3,subplot_kw={'projection':proj},constrained_layout=True,figsize=(16,8.5))

for vv in range(2):
    
    if vv == 0:
        stdplot = std_T
        vmax      = 0.5
        vunit     = "degC/mon"
    else:
        stdplot = std_S
        vmax      = 0.1
        vunit     = "psu/mon"
    
    axs = axsall[vv,:]
    for ax in axs:
        ax.set_extent(bboxplot)
        ax.coastlines()
    
    for ii in range(3):
        ax = axs[ii]
        plotvar = stdplot[ii]  * mask * dtmon
        pcm     = ax.pcolormesh(plotvar.lon,plotvar.lat,plotvar,vmin=-vmax,vmax=vmax)
        cb      = viz.hcbar(pcm,ax=ax)
        cb.set_label(vunit)
        
        ax.set_title(plotnames[ii])
        
    

#%% Check the amplitude, first for SST (by printing the values)

reg_T = [proc.sel_region_xr(ds,bboxplot) for ds in std_T]
reg_S = [proc.sel_region_xr(ds,bboxplot) for ds in std_S]

[print(np.nanmean(ds * mask)) for ds in std_T]
[print(np.nanmean(ds * mask)) for ds in std_S]

#%%

#%% Make Draft 04 Plot (Geostrophic Advection Amplitude)


#%% Load some data

ugeoprime_monvar    = xr.open_dataset(rawpath  + 'ugeoprime_gradT_gradS_NATL_monvar.nc').load()
ugeobar_monvar      = xr.open_dataset(rawpath  + 'ugeobar_gradTprime_gradSprime_NATL_monvar.nc').load()

ugeores_monvar = []
for vv in range(2):
    if vv == 0:
        invar = terms_T[-1].groupby('time.month').var('time').rename('SST')
    else:
        invar = terms_S[-1].groupby('time.month').var('time').rename("SSS")
    ugeores_monvar.append(invar)
        
    
    

vnames    = ["SST","SSS"]
ugeo_names = ["ubar","uprime"]

ugeo_names_2 = [
    r"$\overline{u_{geo}} \cdot \nabla T'$",
    r"$u_{geo}' \cdot \nabla \overline{T}$",
    r"$\overline{u_{geo}} \cdot \nabla S'$",
    r"$u_{geo}' \cdot \nabla \overline{S}$",
    ]




#%% Make a plot of the maximum variance (and month which it occurs)

plot_point = True
Draft4     = True # Set to True to just plot 1 row

if Draft4:
    fig,axs,_ = viz.init_orthomap(1,4,bboxplot,figsize=(28,7.5))
    
else:
    fig,axs,_ = viz.init_orthomap(2,4,bboxplot,figsize=(28,12.5))

for ax in axs.flatten():
    ax = viz.add_coast_grid(ax,bbox=bboxplot,fill_color="lightgray",fontsize=24)


irow = 0
ii=0
for vv in range(2):

    vname = vnames[vv]
    
    if vv == 0:
        vmax = 0.5
        cmap = 'cmo.thermal'
        vunit = '\degree C'
        ccol  = "k"
        cints = np.arange(0,1.5,0.4)
        if Draft4:
            axvar = axs[:2]
        else:
            axvar  = axs[0,:2]
    else:
        vmax = 0.075
        cmap = 'cmo.rain'
        vunit = "psu"
        ccol  = "lightgray"
        cints = np.arange(0,1.5,0.1)
        if Draft4:
            axvar = axs[2:]
        else:
            axvar  = axs[0,2:]
    
    for ui in range(2):
        
        if ui == 0:
            ugeoin = ugeobar_monvar
            ulab   = "\overline{u}"
            
        elif ui == 1:
            ugeoin = ugeoprime_monvar
            ulab   = "u'"
            
            
        # First, plot the maximum variance -----------------------------------
        if not Draft4:
            ax        = axs[0,irow]
        else:
            ax        = axs[irow]
        plotvar   = (np.sqrt(ugeoin[vname].mean('ens').max('month'))) * dtmon * mask
        
        
        pcm     = ax.pcolormesh(plotvar.lon,plotvar.lat,plotvar,
                                transform=proj,vmin=0,vmax=vmax,cmap=cmap)
        
        cl    = ax.contour(plotvar.lon,plotvar.lat,plotvar,levels=cints,
                                transform=proj,colors=ccol,linewidths=0.75)
        ax.clabel(cl,fontsize=fsz_tick)
        
        lab     = ugeo_names_2[irow]#"%s,$%s$" % (vnames[vv],ulab)
        if ui == 1:
            
            cb = viz.hcbar(pcm,ax=axvar.flatten())
            cb.ax.tick_params(labelsize=fsz_tick)
            
            cb.set_label("$%s$ month$^{-1}$" % vunit,fontsize=fsz_axis)
        
        ax.set_title(lab,fontsize=fsz_axis)
        
        
        # Plot Additional Features (Ice Edge, ETC)
        ax.plot(ds_gs2.lon.mean('mon'),ds_gs2.lat.mean('mon'),transform=proj,lw=2.5,c='cornflowerblue',ls='dashdot')

        # Plot Ice Edge
        ax.contour(icemask.lon,icemask.lat,mask_plot,colors="cyan",linewidths=2.5,
                   transform=proj,levels=[0,1],zorder=-1)
        viz.label_sp(ii,alpha=0.75,ax=ax,fontsize=fsz_axis,y=1.08,x=-.02)
        
        if plot_point:
            nregs = len(ptnames)
            for ir in range(nregs):
                pxy   = ptcoords[ir]
                ax.plot(pxy[0],pxy[1],transform=proj,markersize=20,markeredgewidth=.5,c=ptcols[ir],
                        marker='*',markeredgecolor='k')
        viz.label_sp(ii,alpha=0.75,ax=ax,fontsize=fsz_axis,y=1.08,x=-.02)
        #ii+=1
        
        # Second, plot the month of maximum variance -------------------------
        
        if not Draft4:
            ax        = axs[1,irow]
            plotvarm   = (proc.nanargmaxds(ugeoin[vname].mean('ens'),'month')+1) * mask 
            
            pcm2       = ax.pcolormesh(plotvarm.lon,plotvarm.lat,plotvarm,
                                    transform=proj,vmin=1,vmax=12,cmap='twilight')
            
            
            # Plot Additional Features (Ice Edge, ETC)
            ax.plot(ds_gs2.lon.mean('mon'),ds_gs2.lat.mean('mon'),transform=proj,lw=2.5,c='cornflowerblue',ls='dashdot')
    
            # Plot Ice Edge
            ax.contour(icemask.lon,icemask.lat,mask_plot,colors="cyan",linewidths=2.5,
                       transform=proj,levels=[0,1],zorder=-1)
            viz.label_sp(ii+4,alpha=0.75,ax=ax,fontsize=fsz_axis,y=1.08,x=-.02)
            
            
            if plot_point:
                nregs = len(ptnames)
                for ir in range(nregs):
                    pxy   = ptcoords[ir]
                    ax.plot(pxy[0],pxy[1],transform=proj,markersize=20,markeredgewidth=.5,c=ptcols[ir],
                            marker='*',markeredgecolor='k')
            # scints  = [3,7]
        # cl      = ax.contour(plotvarm.lon,plotvarm.lat,plotvarm,
        #                      transform=proj,levels=scints,colors='cyan')
        # ax.clabel(cl,fontsize=fsz_tick)
        
        
        # # Plot Labels for points
        # for (i, j), z in np.ndenumerate(plotvarm):
        #     try:
        #         ax.text(plotvarm.lon.data[j], plotvarm.lat.data[i], '%i' % (z),
        #                 ha='center', va='center',transform=proj,fontsize=14,color='k',zorder=4,)#path_effects=[pe.withStroke(linewidth=1.5, foreground="w")])
        #     except:
        #         pass
                
        #viz.label_sp(ii+4,alpha=0.75,ax=ax,fontsize=fsz_axis,y=1.08,x=-.02)
        irow += 1
        ii+=1

# Add Monthly Variance Colorbar
if not Draft4:
    cbax = axs[1,:].flatten()
    cb = viz.hcbar(pcm2,ax=cbax,fraction=0.0455,pad=0.01)
    cb.ax.tick_params(labelsize=fsz_tick)
    cb.ax.set_xticks(np.arange(1,13,1))
    cb.set_label("Month of Maximum Interannual Variability",fontsize=fsz_axis)


savename       = "%sUgeo_Contribution_Pointwise_MaxMonthly.png" % (figpath)
plt.savefig(savename,dpi=150,bbox_inches='tight')

#%% Plot all 3

fsz_axis        = 25
fsz_title       = 38
fsz_tick        = 24
cmap            = 'cmo.balance'
vnames          = ["SST","SSS"]

vmaxes = [1,0.1]
vunits = ["\degree C","psu"]
vcmaps = [cm.lajolla_r,cm.acton_r]

ugeo_names_col  = (
    r"$\overline{u_{geo}} \cdot \nabla %s'$",
    r"$u_{geo}' \cdot \nabla \overline{%s}$",
    r"$u_{geo}' \cdot \nabla %s'$"
    )

fig,axs,_ = viz.init_orthomap(2,3,bboxplot,figsize=(28,12.5))

ii        = 0
for vv in range(2):
    
    vname = vnames[vv]
    vunit = vunits[vv]
    
    cmap  = vcmaps[vv]
    
    for ui in range(3):
        
        
        if ui == 0:
            plotvar = ugeobar_monvar[vname].max('month').mean('ens')
        elif ui == 1:
            plotvar = ugeoprime_monvar[vname].max('month').mean('ens')
        else:
            plotvar = ugeores_monvar[vv].max('month').mean('ens')
            
        
        ax = axs[vv,ui]
        ax = viz.add_coast_grid(ax,bbox=bboxplot,fill_color="lightgray",fontsize=24)
        
        if ui == 0:
            viz.add_ylabel(vnames[vv],ax=ax,fontsize=fsz_title)
        
        
        if vv == 0:
            title= ugeo_names_col[ui] % (vnames[vv][-1])
            ax.set_title(title,fontsize=fsz_title)
            
            
        plotvar = np.sqrt(plotvar) * mask * dtmon
        # if ui > 1:
        #     plotvar * dtmon
        pcm     = ax.pcolormesh(plotvar.lon,plotvar.lat,plotvar,
                                transform=proj,cmap=cmap,
                                vmin=0,vmax=vmaxes[vv])
        
        # cl    = ax.contour(plotvar.lon,plotvar.lat,plotvar,levels=cints,
        #                         transform=proj,colors=ccol,linewidths=0.75)
        # ax.clabel(cl,fontsize=fsz_tick)
        
        # cb = viz.hcbar(pcm,ax=ax)
        # cb.ax.tick_params(labelsize=fsz_tick)
        
        if ui == 2:
            cb = fig.colorbar(pcm,ax=axs[vv,:].flatten(),fraction=0.025,pad=0.01)
            cb.ax.tick_params(labelsize=fsz_tick)
            cb.set_label("$%s$ per Month" % (vunit),fontsize=fsz_axis)
            
        
        
        viz.label_sp(ii,alpha=0.75,ax=ax,fontsize=fsz_axis,y=1.08,x=-.02)
        ii+=1
        
plt.suptitle("Max Interannual Variability for Geostrophic Advection",fontsize=fsz_title)

savename       = "%sUgeo_Terms_Stdev.png" % (figpath)
plt.savefig(savename,dpi=150,bbox_inches='tight')    
  
