#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Compare Monthly Variance from 2 Experiments
Copied upper section of region_analysis_manual


Currently written to compare the effect of Qek Ekman Forcing

Created on Mon Jul 22 13:57:45 2024

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

bboxplot                        = [-80,0,20,65]
mpl.rcParams['font.family']     = 'Avenir'
mons3                           = proc.get_monstr(nletters=3)

fsz_tick                        = 18
fsz_axis                        = 20
fsz_title                       = 28

rhocrit                         = proc.ttest_rho(0.05,2,86)

proj                            = ccrs.PlateCarree()

#%%  Indicate Experients (copying upper setion of viz_regional_spectra )


# Compare the Effect of Ekman Forcing (SSS)
regionset       = "SSSCSU"
comparename     = "SSS_Qek_Effect"
expnames        = ["SSS_EOF_LbddCorr_Rerun_NoQek","SSS_EOF_LbddCorr_Rerun"] 
expnames_long   = ["SSS (No Qek)","SSS (Qek)",]
expnames_short  = ["SM_NoQek","SM"]
ecols           = ["navy","forestgreen"]
els             = ["dotted","dashed"]
emarkers        = ["s","d"]

# Compare the Effect of Ekman Forcing (SST)
regionset       = "SSSCSU"
comparename     = "SST_Qek_Effect"
expnames        = ["SST_EOF_LbddCorr_Rerun_NoQek","SST_EOF_LbddCorr_Rerun"] 
expnames_long   = ["SST (No Qek)","SST (Qek)",]
expnames_short  = ["SM_NoQek","SM"]
ecols           = ["navy","forestgreen"]
els             = ["dotted","dashed"]
emarkers        = ["s","d"]


# Compare the Effect of SST-Evaporation Feedback (SSS)
regionset       = "SSSCSU"
comparename     = "SSS_lbdE_Effect"
expnames        = ["SSS_EOF_LbddCorr_Rerun","SSS_EOF_LbddCorr_Rerun_lbdE_neg",] 
expnames_long   = ["SSS","SSS ($\lambda^e$)",]
expnames_short  = ["SM","SM_lbdE",]
ecols           = ["forestgreen","magenta",]
els             = ["dashed","solid",]
emarkers        = ["d","o",]


# Compare the Effect of SST-Evaporation Feedback (SSS)
regionset       = "SSSCSU"
comparename     = "SSS_SM_v_CESM"
expnames        = ["SSS_CESM","SSS_EOF_LbddCorr_Rerun_lbdE_neg",] 
expnames_long   = ["SSS (CESM)","SSS (SM)",]
expnames_short  = ["CESM","SM",]
ecols           = ["k","magenta",]
els             = ["solid","dashed",]
emarkers        = ["d","o",]



cesm_exps       = ["SST_CESM","SSS_CESM",
                  "SST_cesm1le_5degbilinear","SSS_cesm1le_5degbilinear",]



#%% Load Bounding boxes to plot (copied form viz_CESM1_HTR_meanstates)

# Get Bounding Boxes
regiondicts     = rparams.region_sets[regionset]
bboxes          = regiondicts['bboxes']
regions_long    = regiondicts['regions_long']
rcols           = regiondicts['rcols']
rsty            = regiondicts['rsty']

regplot = [0,1,3]
nregs   = len(regplot)


#%% Load other things

# Load data processed by [calc_monmean_CESM1.py]
ds_sss          = dl.load_monmean('SSS')
ds_sst          = dl.load_monmean('SST')

#%% Load the Dataset (us sm output loader)
# Hopefully this doesn't clog up the memory too much

nexps = len(expnames)
ds_all = []
for e in tqdm.tqdm(range(nexps)):
    
    # Get Experiment information
    expname        = expnames[e]
    
    if "SSS" in expname:
        varname = "SSS"
    elif "SST" in expname:
        varname = "SST"
    
    # For stochastic model output
    ds = dl.load_smoutput(expname,output_path)
    
    if expname in cesm_exps:
        print("Detrending and deseasoning")
        ds = proc.xrdeseason(ds[varname])
        ds = ds - ds.mean('ens')
        ds = ds.rename(dict(ens='run'))
        ds = xr.where(np.isnan(ds),0,ds) # Sub with zeros for now
    else:
        ds = ds[varname]
        
    ds_all.append(ds)

#%% Load some variables to plot 

# Load Current
ds_uvel,ds_vvel = dl.load_current()

# load SSS Re-emergence index (for background plot)
ds_rei          = dl.load_rei("SSS_CESM",output_path).load().rei


# Load Gulf Stream
ds_gs = dl.load_gs()
ds_gs = ds_gs.sel(lon=slice(-90,-50))

# Load 5deg mask
maskpath = input_path + "masks/"
masknc5  = "cesm1_htr_5degbilinear_icemask_05p_year1920to2005_enssum.nc"
dsmask5 = xr.open_dataset(maskpath + masknc5)
dsmask5 = proc.lon360to180_xr(dsmask5).mask.drop_duplicates('lon')

masknc = "CESM1LE_HTR_limask_pacificmask_enssum_lon-90to20_lat0to90.nc"
dsmask = xr.open_dataset(maskpath + masknc).MASK.load()

maskin = dsmask

#%% Compute Overall Variance

ds_var      = [ds.var('time').mean('run') for ds in ds_all]
ds_monvar   = [ds.groupby('time.month').var('time').mean('run') for ds in ds_all]


#%% Make Variance Comparison Plots

if varname == "SSS":
    vlims_reg  = [0,0.010]
    #vlims_diff = [-0.010,0.010] # For Qek and LbdE Comparisons
    vlims_diff = [-.025,0.025]
    vunits     = "psu"
    
    cmap_reg   = 'cmo.haline'
    
    
    # Gradient Information
    ds_mean    = ds_sss[varname]
    cints      = np.arange(33,39,.3)
    
elif varname == "SST":
    vlims_reg  = [0,0.5]
    vlims_diff = [-0.25,0.25]
    vunits     = "\degree C"
    
    cmap_reg   = 'cmo.thermal'
    
    # Gradient Information
    ds_mean    = ds_sst[varname]
    cints      = np.arange(250,310,2)
    
    
pmesh       = True

fig,axs,_   = viz.init_orthomap(1,3,bboxplot,figsize=(26,8),centlat=45,)

for ii in range(3):
    
    ax = axs.flatten()[ii]
    ax = viz.add_coast_grid(ax,bboxplot,fill_color="lightgray",fontsize=20,
                               fix_lon=np.arange(-80,10,10),fix_lat=np.arange(0,70,10),grid_color="k")
    
    if ii == 0:
        title   = expnames_long[0]
        plotvar = ds_var[0]
        cmap_in = cmap_reg
        vlims   = vlims_reg
        cblab   = "SSS Variance [$%s^2$]" % vunits
    
    elif ii == 1:
        title = expnames_long[1]
        plotvar = ds_var[1]
        cmap_in = cmap_reg
        vlims   = vlims_reg
        cblab   = "SSS Variance [$%s^2$]" % vunits
        
    else:
        title = "%s - %s"  % (expnames_long[1], expnames_long[0])
        plotvar = ds_var[1] - ds_var[0]
        cmap_in = 'cmo.balance'
        vlims   = vlims_diff
        cblab   = "Variance Difference [$%s^2$]" % vunits
    
    ax.set_title(title,fontsize=fsz_title)
    
    pcm = ax.pcolormesh(plotvar.lon,plotvar.lat,plotvar * dsmask.squeeze(),
                        transform=proj,cmap=cmap_in,
                        vmin=vlims[0],vmax=vlims[1])
    cb = viz.hcbar(pcm,ax=ax)
    cb.ax.tick_params(labelsize=fsz_tick)
    cb.set_label(cblab,fontsize=fsz_axis)
    
    # Plot Bounding Boxes (from viz_HTR_meanstates)
    for ir in range(nregs):
        rr = regplot[ir]
        rbbx = bboxes[rr]
        viz.plot_box(rbbx,ax=ax,
                     color=rcols[rr],linestyle=rsty[rr],leglab=regions_long[rr],linewidth=2.5,return_line=True)
        
        
    # Plot Mean Gradients
    plotvar = ds_mean.mean('ens').mean('mon').transpose('lat','lon') * dsmask.squeeze()#* mask_apply
    cl = ax.contour(plotvar.lon,plotvar.lat,plotvar,transform=proj,
                linewidths=1.5,colors="gray",levels=cints,linestyles='dashed')
    ax.clabel(cl)

    
plt.suptitle("Overall %s Variance Difference" % varname,fontsize=fsz_title)
savename = "%s%s_Overall_Variance_Pointwise.png" % (figpath,comparename)
plt.savefig(savename,dpi=150,bbox_inches='tight')

#%% Plot Monthly Difference in Variance

plotmons = np.roll(np.arange(12),1)

fig,axs,_   = viz.init_orthomap(4,3,bboxplot,figsize=(26,28),centlat=45,)

for aa in range(12):
    ax = axs.flatten()[aa]
    im = plotmons[aa]
    ax.set_title(mons3[im],fontsize=fsz_title)
    
    ax = viz.add_coast_grid(ax,bboxplot,fill_color="lightgray",fontsize=20,
                               fix_lon=np.arange(-80,10,10),fix_lat=np.arange(0,70,10),grid_color="k")
    
    plotvar = ds_monvar[1].isel(month=im) - ds_monvar[0].isel(month=im)
    cmap_in = 'cmo.balance'
    vlims   = vlims_diff
    cblab   = "Variance Difference [$%s^2$]" % vunits
    
    pcm = ax.pcolormesh(plotvar.lon,plotvar.lat,plotvar * dsmask.squeeze(),
                        transform=proj,cmap=cmap_in,
                        vmin=vlims[0],vmax=vlims[1])
    
    
    # Plot Bounding Boxes
    for ir in range(nregs):
        rr = regplot[ir]
        rbbx = bboxes[rr]
        viz.plot_box(rbbx,ax=ax,
                     color=rcols[rr],linestyle=rsty[rr],leglab=regions_long[rr],linewidth=2.5,return_line=True)
    
    
    # Plot Mean Gradients
    plotvar = ds_mean.mean('ens').isel(mon=im).transpose('lat','lon') * dsmask.squeeze()#* mask_apply
    cl = ax.contour(plotvar.lon,plotvar.lat,plotvar,transform=proj,
                linewidths=1.5,colors="gray",levels=cints,linestyles='dashed')
    ax.clabel(cl)
    
    
    
cb = viz.hcbar(pcm,ax=axs.flatten(),fraction=0.025)
cb.ax.tick_params(labelsize=fsz_tick)
cb.set_label(cblab,fontsize=fsz_axis)
plt.suptitle("%s Variance Difference By Month\n%s - %s" % (varname,expnames_long[1], expnames_long[0]),fontsize=fsz_title)   
savename = "%s%s_Monthly_Variance_Pointwise.png" % (figpath,comparename)
plt.savefig(savename,dpi=150,bbox_inches='tight')



