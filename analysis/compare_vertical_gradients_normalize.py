#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Compare vertical gradients at/around the detraiment depth
Test 2 different types of normalization (using stdev SST/SSS, and stdev
                                         at the detrainment depth)


Copied upper section from compare_detrainment_damping on 2024.05.28

Uses output from:
    - [calc_stdev_SALT_TEMP] >> computs stdev from surface values
    - [compare_detrainment_damping] >> postprocessses output from [compute_dz_ens]

Created on Tue May 28 10:57:47 2024

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


#%% Other User Edits


proc.makedir(figpath)
ocnpath        = rawpath + "ocn_var_3d/"

# Loading Options
#ds_corr   = [dstemp_dtgrad,dssalt_dtgrad]
ds_names  = ["SST","SSS"]
vunits_grad = [r"$\frac{dT}{dz}$ ($\degree C$/meter)",r"$\frac{dS}{dz}$ (psu/meter)"]
vnames    = ["TEMP","SALT"]
vcolors   = ["hotpink","navy"]
plot_sids = [1,3,0]


vunits = ["$\degree C$","psu"]

#%% Load plotting variables, bsf, ice mask

# Plotting Params
mpl.rcParams['font.family'] = 'JetBrains Mono'
bboxplot                    = [-80,0,20,65]
proj                        = ccrs.PlateCarree()
#lon                         = ds.lon.values
#lat                         = ds.lat.values
mons3                       = proc.get_monstr()

# Font Sizes
fsz_title= 25
fsz_axis = 20
fsz_ticks= 18


#%%



#mask_plot[np.isnan(mask)] = 0

#%% Load output from above

detrainmean_byvar = []
meanstds          = []
meanstds_surf     = []

dz2_grad          = []

for vv in range(2):
    
    # Load Detrain Gradients
    outname = "%sCESM1_HTR_%s_Detrain_Gradients_detrainmean.nc" % (ocnpath,vnames[vv])
    dsload  = proc.sel_region_xr(xr.open_dataset(outname).load(),bboxplot)
    detrainmean_byvar.append(dsload['grad'].copy())
    
    # Load Detrainment Depth Standard Deviation
    outname = "%sCESM1_HTR_%s_Detrain_stdev_detrainmean.nc" % (ocnpath,vnames[vv])
    dsload  = proc.sel_region_xr(xr.open_dataset(outname).load(),bboxplot)
    meanstds.append(dsload['stdev'].copy())
    
    # Load Surface Standard Deviation
    if vv == 0:
        vn = "TEMP"
        outname = "%sCESM1LE_%s_NAtl_19200101_20050101_NN_stdev.nc" % (rawpath,vn)
    else:
        vn = "SSS"
        outname = "%sCESM1LE_%s_NAtl_19200101_20050101_bilinear_stdev.nc" % (rawpath,vn)
    dsload  = proc.sel_region_xr(xr.open_dataset(outname).load(),bboxplot)
    meanstds_surf.append(dsload[vn].copy())
    
    # Load 2nd derivatiev
    outname = "%sCESM1_HTR_%s_Detrain_Gradients_detrainmean_dz2.nc" % (ocnpath,vnames[vv])
    dsload  = proc.sel_region_xr(xr.open_dataset(outname).load(),bboxplot)
    dz2_grad.append(dsload['grad'].copy())
    
    
# %%  Load BSF and Ice Mask for plotting

dsref    = detrainmean_byvar[0]

# Settings and load
bsf      = dl.load_bsf()

# Load Land Ice Mask
icemask  = xr.open_dataset(input_path + "/masks/CESM1LE_HTR_limask_pacificmask_enssum_lon-90to20_lat0to90.nc")

# Resize
bsf,icemask,_    = proc.resize_ds([bsf,icemask,dsref])
bsf_savg = proc.calc_savg_mon(bsf)

#
mask = icemask.MASK.squeeze()
mask_plot = xr.where(np.isnan(mask),0,mask)#mask.copy()

# Plotting
cints_bsf = np.arange(-50,60,10)


#%% Compare Surface and Detrainment Standard Deviation

vlms       = [[0,1],[0,0.35]]
cmaps      = ["cmo.thermal","cmo.haline"]
selmon     = [8,9,10]
mmnames    = ["Surface","Detrain Depth"]

fig,axs,_ = viz.init_orthomap(2,2,bboxplot,figsize=(18,13.5))

for mm in range(2):
    
    if mm == 0:
        vnames_in = ds_names
        invars    = meanstds_surf
    else:
        vnames_in = [vn+" @ Detrain Depth" for vn in vnames]
        invars    = meanstds
    for vv in range(2):
        
        
        # get axis and plot coast
        ax = axs[mm,vv]
        ax = viz.add_coast_grid(ax,bbox=bboxplot,fill_color="lightgray")
        
        
        # Titles and labeling
        ax.set_title("$\sigma$(%s)" % (vnames_in[vv]),fontsize=fsz_title)
        if vv == 0:
            viz.add_ylabel(mmnames[mm],ax=ax,x=-0.20,y=0.5,fontsize=fsz_axis,rotation='horizontal')
        
        # Get Variable
        plotvar = invars[vv].mean('ens').isel(mon=selmon).mean('mon')
        lon     = plotvar.lon
        lat     = plotvar.lat
        
        
        
        # Plot Variable
        if vlms is None:
            pcm     = ax.pcolormesh(lon,lat,plotvar,transform=proj,cmap=cmaps[vv])
        else:
            vlms_in = vlms[vv]
            pcm     = ax.pcolormesh(lon,lat,plotvar,transform=proj,vmin=vlms_in[0],vmax=vlms_in[1],cmap=cmaps[vv])
        
        cb      = fig.colorbar(pcm,ax=ax,orientation='horizontal',fraction=0.045)
        cb.set_label(vunits[vv],fontsize=fsz_axis)
        
        
        # Plot BSF
        
        plotbsf = bsf.isel(mon=selmon).mean('mon').BSF * mask
        cl      = ax.contour(plotbsf.lon,plotbsf.lat,plotbsf,
                             levels=cints_bsf,colors="k",linewidths=0.75,transform=proj)
        
        
        # Plot Mask
        cl2      = ax.contour(mask.lon,mask.lat,mask_plot,
                             levels=[0,1,2],colors="w",linewidths=2,transform=proj)
        
        
savename = "%sNormalizedFactor_SurfvDetrain_EnsAvg_OND.png" % (figpath)
plt.savefig(savename,dpi=150,bbox_inches='tight')


#%% Plot Difference plot for above

vlms  = [[-.2,.2],[-.05,.05]]
cmap  = 'cmo.balance'
fig,axs,_ = viz.init_orthomap(1,2,bboxplot,figsize=(14,13.5))
vnames_in = ds_names
for vv in range(2):
    
    # Take the Difference
    invars = meanstds_surf[vv] - meanstds[vv]
    
    # get axis and plot coast
    ax = axs[vv]
    ax = viz.add_coast_grid(ax,bbox=bboxplot,fill_color="lightgray")
    
    # Titles and labeling
    ax.set_title("$\sigma$(%s)" % (vnames_in[vv]),fontsize=fsz_title)
    if vv == 0:
        viz.add_ylabel("Surface - Detrain",ax=ax,x=-0.40,y=0.5,fontsize=fsz_axis,rotation='horizontal')
    
    # Get Variable
    plotvar = invars.mean('ens').isel(mon=selmon).mean('mon')
    lon     = plotvar.lon
    lat     = plotvar.lat
    
    # Plot Variable
    if vlms is None:
        pcm     = ax.pcolormesh(lon,lat,plotvar,transform=proj,cmap=cmap)
    else:
        vlms_in = vlms[vv]
        pcm     = ax.pcolormesh(lon,lat,plotvar,transform=proj,vmin=vlms_in[0],vmax=vlms_in[1],cmap=cmap)
    
    cb      = fig.colorbar(pcm,ax=ax,orientation='horizontal',fraction=0.045,pad=0.01)
    cb.set_label(vunits[vv],fontsize=fsz_axis)
    
    
    # Plot BSF
    
    plotbsf = bsf.isel(mon=selmon).mean('mon').BSF * mask
    cl      = ax.contour(plotbsf.lon,plotbsf.lat,plotbsf,
                         levels=cints_bsf,colors="k",linewidths=0.75,transform=proj)
    
    
    # Plot Mask
    cl2      = ax.contour(mask.lon,mask.lat,mask_plot,
                         levels=[0,1,2],colors="w",linewidths=2,transform=proj)
        
        
savename = "%sNormalizedFactor_SurfvDetrain_EnsAvg_OND_diff.png" % (figpath)
plt.savefig(savename,dpi=150,bbox_inches='tight')


#%% Visualize the normalization

kmonths     = [8,9,10]
sid         = 3
vlms        = None
surfnorm    = True # Set to True to normalize by surface values

dz2 = True # Set to True to plot the second Derivative


fig,axs,_   = viz.init_orthomap(2,3,bboxplot,figsize=(12,6.5))

if dz2:
    vlims_in    = np.array([[[0,0.0045], [0,1.5] ,[0,.005]],
                         [[0,0.00025],[0,.3], [0,.005]]])
else:
    vlims_in    = np.array([[[0,0.25], [0,1.5] ,[0,.3]],
                         [[0,0.025],[0,.3], [0,.3]]])

# vunits_in = np.array([["$\frac{\degree C}{m}$","$\degree C$","$m^{-1}$",]
#              ["$\frac{psu}{m}$","$psu$","$m^{-1}$",]])

if surfnorm:
    meanstds_in = meanstds_surf
else:
    meanstds_in = meanstds

cmaps    = ['cmo.thermal','cmo.haline']
vunits   = ["\degree C","psu"]
for vv in range(2):
    for ii in range(3):
        ax    = axs[vv,ii]
        vlms  = vlims_in[vv,ii,:]
        if ii == 0:
            title    = "Absolute Gradient"
            
            if dz2:
                plotvar  = dz2_grad[vv].isel(mon=kmonths).mean('mon').mean('ens')
                vunit = r"$\frac{%s}{m^2}$" % (vunits[vv])
            else:
                plotvar  = detrainmean_byvar[vv].isel(mon=kmonths).mean('mon').mean('ens')
                vunit    = r"$\frac{%s}{m}$" % (vunits[vv])
        elif ii == 1:
            title = "Std. Dev."
            plotvar = meanstds_in[vv].isel(mon=kmonths).mean('mon').mean('ens')
            vunit    = "$%s$" % vunits[vv]
        elif ii == 2:
            title = "Normalized Gradient"
            
            
            if dz2:
                grads = dz2_grad[vv].isel(mon=kmonths).mean('mon').mean('ens')
                vunit   = r"$m^{-2}$"
            else:
                grads = detrainmean_byvar[vv].isel(mon=kmonths).mean('mon').mean('ens')
                vunit   = r"$m^{-1}$"
            stdev = meanstds[vv].isel(mon=kmonths).mean('mon').mean('ens')
            plotvar = grads/stdev
            
        ax     = viz.add_coast_grid(ax,bbox=bboxplot,fill_color="lightgray")
        if vv == 0:
            ax.set_title(title)
            
        
        if ii == 0:
            viz.add_ylabel(vnames[vv],ax=ax,x=-0.20,y=0.5,fontsize=fsz_title,rotation='horizontal')
            
        # Plot Things
        if vlms is None:
            pcm     = ax.pcolormesh(lon,lat,plotvar,transform=proj,cmap=cmaps[vv])
        else:
            pcm     = ax.pcolormesh(lon,lat,plotvar,transform=proj,vmin=vlms[0],vmax=vlms[1],cmap=cmaps[vv])

        cb      = fig.colorbar(pcm,ax=ax,orientation='horizontal',fraction=0.02)
        cb.set_label(vunit)
        
        
        # Plot BSF
        plotbsf = bsf_savg.isel(season=sid).BSF * mask
        cl      = ax.contour(plotbsf.lon,plotbsf.lat,plotbsf,
                             levels=cints_bsf,colors="gray",linewidths=0.75,transform=proj)
        
        
        # Plot Mask
        cl2      = ax.contour(mask.lon,mask.lat,mask_plot,
                             levels=[0,1,2],colors="w",linewidths=2,transform=proj)
        
savename = "%sNormalized_Vertical_Diff_SeasonalDiff_EnsAvg_SON_surfnorm%i.png" % (figpath,surfnorm)
if dz2:
    savename=proc.addstrtoext(savename,"_dz2")
plt.savefig(savename,dpi=150,bbox_inches='tight')

#%% Compute the normalized gradients


normgrads_bymethod = []
for mm in range(2):
    
    if mm == 0:
        meanstds_in = meanstds_surf
    else:
        meanstds_in = meanstds
    
    normgrads = []
    for vv in range(2):
        grads = detrainmean_byvar[vv].isel(mon=kmonths).mean('mon').mean('ens')
        stdev = meanstds_in[vv].isel(mon=kmonths).mean('mon').mean('ens')
        plotvar = grads/stdev
        normgrads.append(plotvar.copy())
    normgrads_bymethod.append(normgrads)
        
    #plotvar = normgrads[1] - normgrads[0]
    
    
    
#%% Visualize difference between normalized gradients

kmonths     = [8,9,10]
sid         = 3
vlms        = None
surfnorm    = True # Set to True to normalize by surface values
cmap       = 'cmo.balance'

fig,axs,_   = viz.init_orthomap(1,2,bboxplot,figsize=(10,10))

vlms        = [-.30,.30]#[[-.1,.1],[-.1,.1]]

# vunits_in = np.array([["$\frac{\degree C}{m}$","$\degree C$","$m^{-1}$",]
#              ["$\frac{psu}{m}$","$psu$","$m^{-1}$",]])

if surfnorm:
    meanstds_in = meanstds_surf
else:
    meanstds_in = meanstds

cmaps    = ['cmo.thermal','cmo.haline']
vunits   = ["\degree C","psu"]

for mm in range(2):
    
    ax      = axs[mm]
    title   = "Normalized Gradient (%s)" % mmnames[mm]
    
    normgrads = normgrads_bymethod[mm]
    plotvar   = normgrads[0] - normgrads[1]
    #vunit   = r"$m^{-1}$"
    
    ax     = viz.add_coast_grid(ax,bbox=bboxplot,fill_color="lightgray")
    ax.set_title(title)
    
    if ii == 0:
        viz.add_ylabel(vnames[vv],ax=ax,x=-0.20,y=0.5,fontsize=fsz_title,rotation='horizontal')
    
    # Plot Things
    if vlms is None:
        pcm     = ax.pcolormesh(lon,lat,plotvar,transform=proj,cmap=cmaps[vv])
    else:
        vlms_in = vlms
        pcm     = ax.pcolormesh(lon,lat,plotvar,transform=proj,vmin=vlms_in[0],vmax=vlms_in[1],cmap=cmap)

    
    # Plot BSF
    plotbsf = bsf_savg.isel(season=sid).BSF * mask
    cl      = ax.contour(plotbsf.lon,plotbsf.lat,plotbsf,
                         levels=cints_bsf,colors="gray",linewidths=0.75,transform=proj)

    # Plot Mask
    cl2      = ax.contour(mask.lon,mask.lat,mask_plot,
                         levels=[0,1,2],colors="w",linewidths=2,transform=proj)
   

cb      = fig.colorbar(pcm,ax=axs.flatten(),orientation='horizontal',fraction=0.035,pad=0.01)
cb.set_label("dTEMP/dz - dSALT/dz (%s)"  % vunit)     
savename = "%sNormalized_Vertical_Diff_SALTvTEMP_EnsAvg_SON.png" % (figpath)
plt.savefig(savename,dpi=150,bbox_inches='tight')


# %%


# #%% Plot the Difference in SALT and TEMP normalized differences


# vlms       = [-1,1]
# cmap       = 'cmo.balance'
# # Normfactor [None,spatial,seasonal]
# surfnorm   = True

# # Make the plots
# fig,axs,mdict                = viz.init_orthomap(1,3,bboxplot,figsize=(22,5),constrained_layout=True,)


#     if surfnorm:
        
        
        

# for ss in range(3):
    
#     sid    = plot_sids[ss]
    

        
        
#     if normfactor is not None:
        
#         # Compute Normalization factors
#         if normfactor == "spatial":
#             ds_in = [ds_norm_all[0][0].isel(season=sid),ds_norm_all[0][1].isel(season=sid)]
#         elif normfactor == "seasonal":
#             ds_in = [ds_norm_all[1][0].isel(season=sid),ds_norm_all[1][1].isel(season=sid)]
        
#     # Take the Difference
#     dsplot = np.abs(ds_in[1]) - np.abs(ds_in[0])
    
#     ax     = axs[ss]
#     ax     = viz.add_coast_grid(ax,bbox=bboxplot,fill_color="lightgray")
#     ax.set_title(dsplot.season.values,fontsize=fsz_title)
    
#     if ff == 0:
#         ax.set_title(dsplot.season.values,fontsize=fsz_title)
    
    
#     if vlms is None:
#         pcm     = ax.pcolormesh(lon,lat,dsplot,transform=proj,cmap=cmap)
#     else:
#         pcm     = ax.pcolormesh(lon,lat,dsplot,transform=proj,vmin=vlms[0],vmax=vlms[1],cmap=cmap)
    
#     # Colorbar
#     if vlms is None:
#         cb      = fig.colorbar(pcm,ax=ax,orientation='horizontal',fraction=0.02)
#         cb.set_label("Detrainment Damping (Correlation)")
        

    
    
#     # Plot BSF
#     plotbsf = bsf_savg.isel(season=sid).BSF * mask
#     cl      = ax.contour(plotbsf.lon,plotbsf.lat,plotbsf,
#                          levels=cints_bsf,colors="k",linewidths=0.75,transform=proj)
    
    
#     # Plot Mask
#     cl2      = ax.contour(mask.lon,mask.lat,mask_plot,
#                          levels=[0,1,2],colors="w",linewidths=2,transform=proj)

# if vlms is not None:
#     cb = fig.colorbar(pcm,ax=axs.flatten(),orientation='vertical',fraction=0.02,pad=0.01)
#     cb.set_label("Diff in Normalized Gradient\n$SALT$ - $TEMP$",fontsize=fsz_axis)
    
# #plt.suptitle("Seasonal Mean Differences in Detrainment Correlation",y=1.1,fontsize=32)
# savename = "%sNormalized_Vertical_Diff_SeasonalDiff_EnsAvg_%s_%s.png" % (figpath,compare_name,normfactor)
# plt.savefig(savename,dpi=150,bbox_inches='tight')

