#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

--------------------------------
    Visualize Missing Terms    
--------------------------------

Look at CESM1 vs. Stochastic Model Differences
See how Geostrophic Advection and other components might contribute to them...


Works with output computed from compare_geostrophic_advection_terms
Copied upper section of regional_analysis_manual


(1) Load SST/SSS from CESM
(2) Load SST/SSS from Stochastic Model

(3) Load Geostrophic Advection Terms

Created on Wed Aug 14 15:11:37 2024
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

#%% Load Land Ice Mask


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


#%%  Indicate Experients (copying upper setion of viz_regional_spectra )

regionset       = "SSSCSU"
comparename     = "SST_SSS_Paper_Draft01_Original"
expnames        = ["SST_CESM","SSS_CESM","SST_EOF_LbddCorr_Rerun","SSS_EOF_LbddCorr_Rerun_lbdE_neg"]
expnames_long   = ["CESM1 (SST)","CESM1 (SSS)","Stochastic Model (SST)","Stochastic Model (SSS)"]
expnames_short  = ["CESM1_SST","CESM1_SSS","SM_SST","SM_SSS"]
ecols           = ["firebrick","navy","hotpink",'cornflowerblue']
els             = ["solid","solid",'dashed','dashed']
emarkers        = ["o","d","o","d"]


cesm_exps       = ["SST_CESM","SSS_CESM","SST_cesm2_pic","SST_cesm1_pic",
                  "SST_cesm1le_5degbilinear","SSS_cesm1le_5degbilinear",]
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
        if 'ens' in list(ds.dims):
            ds = ds - ds.mean('ens')
        else:
            ds = proc.xrdetrend(ds)
        ds = xr.where(np.isnan(ds),0,ds) # Sub with zeros for now
    else:
        ds = ds[varname]
    
    ds_all.append(ds)
    
#%% Compute Monthly Variance First, then take the regional average

ds_all_monvar = [ds.groupby('time.month').var('time') for ds in ds_all]
    
#%% Load the ugeo terms 


st          = time.time()
ugeoprime   = xr.open_dataset(rawpath + 'ugeoprime_gradT_gradS_NATL.nc').load()
ugeobar     = xr.open_dataset(rawpath + 'ugeobar_gradTprime_gradSprime_NATL.nc').load()
print("Loaded files in %.2fs" % (time.time()-st))

ds_ugeos    = [ugeoprime.SST,ugeoprime.SSS,ugeobar.SST,ugeobar.SSS]



ugeos_names = [
    r"$u' \cdot \nabla \overline{T}$",
    r"$u' \cdot \nabla \overline{S}$",
    r"$\overline{u} \cdot \nabla T'$",
    r"$\overline{u} \cdot \nabla S'$",
    ]


ugeoprime_monvar    = xr.open_dataset(rawpath  + 'ugeoprime_gradT_gradS_NATL_monvar.nc').load()
ugeobar_monvar      = xr.open_dataset(rawpath  + 'ugeobar_gradTprime_gradSprime_NATL_monvar.nc').load()

ugeoprime_savg      = proc.calc_savg(ugeoprime_monvar.rename(dict(month='mon')),ds=True)
ugeobar_savg        = proc.calc_savg(ugeobar_monvar.rename(dict(month='mon')),ds=True)

ds_ugeos_monvar = [
    ugeoprime_monvar.SST,
    ugeoprime_monvar.SSS,
    ugeobar_monvar.SST,
    ugeobar_monvar.SSS,
    ]

#%% Select a region

sel_box    =  [-40,-30,40,50] # NAC
bbname     = "NAC"
#sel_box    = [-70,-55,35,40] # Sargasso Sea SSS CSU
bbfn,bbti  = proc.make_locstring_bbox(sel_box)

# Compute Regional Average
dsreg     = [proc.sel_region_xr(ds,sel_box) for ds in ds_all]
regavg_ts = [ds.mean('lat').mean('lon').data for ds in dsreg]
#monvar_ts = [ds.groupby('time.month').var('time') for ds in regavg_ts]

# Take Regional Average
ugeoreg     = [proc.sel_region_xr(ds,sel_box) for ds in ds_ugeos]
regavg_ugeo = [ds.mean('lat').mean('lon').data for ds in ugeoreg]

# Do same for monvar (Ugeo)
#tsm_ugeo = [scm.compute_sm_metrics()]
regavg_monvar = [proc.sel_region_xr(ds,sel_box) for ds in ds_ugeos_monvar]
regavg_monvar = [ds.mean('lat').mean('lon').data for ds in regavg_monvar]



# Do same for monvar (CESM1)
ds_regavg_monvar = [proc.sel_region_xr(ds,sel_box) for ds in ds_all_monvar]
ds_regavg_monvar = [ds.mean('lat').mean('lon').data for ds in ds_regavg_monvar]


#%% Compute tsm metrics

tsms = []
for ex in range(nexps):
    ss    = regavg_ts[ex]
    in_ts = [ss[e,:] for e in range(ss.shape[0])]
    tsm   = scm.compute_sm_metrics(in_ts)
    tsms.append(tsm)
    
    
tsms_ugeo = []
for ex in range(4):
    ss    = regavg_ugeo[ex]
    if ex > 1:
        ss = ss.T
    #if np.any(np.isnan(ss)):
        
    ss[np.isnan(ss)] = 0.
        
    in_ts = [ss[e,:] for e in range(ss.shape[0])]
    tsm   = scm.compute_sm_metrics(in_ts)
    tsms_ugeo.append(tsm)
    
#%% Plot Monthly Variances

lw      = 2.5
dtmon   = 3600*24*30

fig,axs = viz.init_monplot(1,2,figsize=(12,4))

# Plot for SST
ax      = axs[0]
ax.set_title("Interannual Variance by Month (SST)")

# Plot CESM1
plotvar = np.array(tsms[0]['monvars']).mean(0) 
label   = "%s" % (expnames_long[0])
ax.plot(mons3,plotvar,label=label,c="k",lw=lw)

# Plot SM
plotvar = np.array(tsms[2]['monvars']).mean(0) 
label   = "%s" % (expnames_long[2])
ax.plot(mons3,plotvar,label=label,c="forestgreen",lw=lw)

# Plot CESM - SST
plotvar = np.array(tsms[0]['monvars']).mean(0) - np.array(tsms[2]['monvars']).mean(0)
label   = "%s - %s" % (expnames_long[0], expnames_long[2])
ax.plot(mons3,plotvar,label=label,c="gray",lw=lw,ls='dashed')

# Plot UgeoPrime
plotvar = np.array(tsms_ugeo[0]['monvars']).mean(0) * dtmon**2
label   = ugeos_names[0]
ax.plot(mons3,plotvar,label=label,c="navy",lw=lw)

# Plot UgeoBar
plotvar = np.array(tsms_ugeo[2]['monvars']).mean(0) * dtmon**2
label   = ugeos_names[2]
ax.plot(mons3,plotvar,label=label,c="cornflowerblue",lw=lw,ls='dotted')

ax.legend()

# Plot for SSS
ax      = axs[1]
ax.set_title("Interannual Variance by Month (SSS)")

# Plot CESM1
plotvar = np.array(tsms[1]['monvars']).mean(0) 
label   = "%s" % (expnames_long[1])
ax.plot(mons3,plotvar,label=label,c="k",lw=lw)

# Plot SM
plotvar = np.array(tsms[3]['monvars']).mean(0) 
label   = "%s" % (expnames_long[3])
ax.plot(mons3,plotvar,label=label,c="violet",lw=lw)

# Plot CESM - SST
plotvar = np.array(tsms[1]['monvars']).mean(0) - np.array(tsms[3]['monvars']).mean(0)
label   = "%s - %s" % (expnames_long[1], expnames_long[3])
ax.plot(mons3,plotvar,label=label,c="gray",lw=lw,ls='dashed')

# Plot UgeoPrime
plotvar = np.array(tsms_ugeo[1]['monvars']).mean(0) * dtmon**2
#plotvar = np.array(tsms_ugeo[1]['monvars']) * dtmon**2
label   = ugeos_names[1]
ax.plot(mons3,plotvar,label=label,c="navy",lw=lw)

# Plot UgeoBar
plotvar = np.array(tsms_ugeo[3]['monvars']).mean(0) * dtmon**2
label   = ugeos_names[3]
ax.plot(mons3,plotvar,label=label,c="violet",lw=lw,ls='dotted')

ax.legend()

savename = "%sCESM1_v_SM_%s_%s_compare_monvar_regavgterm.png" % (figpath,comparename,bbname)
plt.savefig(savename,dpi=150,bbox_inches='tight')

#%% Plot seasonal averages ()


"""
    r"$u' \cdot \nabla \overline{T}$",
    r"$u' \cdot \nabla \overline{S}$",
    r"$\overline{u} \cdot \nabla T'$",
    r"$\overline{u} \cdot \nabla S'$",
"""
dtmon   = 3600*24*30
vname   = "SSS"
pcolor  = True
fsz_title = 24

if vname == 'SST':
    vmax = 0.80
    vunit = "\degree C"
    cmap = 'cmo.thermal'
    cints = np.arange(0,.55,0.05)
    
else:
    vmax = 0.025
    vunit = "psu"
    cmap = 'cmo.haline'
    cints = np.arange(0,0.05,0.005)

fig,axs,mdict = viz.init_orthomap(2,4,bboxplot=bboxplot,figsize=(26,10))

for ax in axs.flatten():
    ax          = viz.add_coast_grid(ax,bbox=bboxplot,fill_color="lightgray",fontsize=24)
    
for uu in range(2):
    
    if uu == 0:
        
        invar = ugeoprime_savg
        ustr  = r"u_{geo}'"
        vstr  = r"\overline{%s}" % vname
        vlabel = r"Interannual Variance $[\frac{%s}{mon}^2]$" % vunit
        
    else:
        
        invar = ugeobar_savg
        ustr  = r"\overline{u_{geo}}"
        vstr  = r"%s'" % vname
        vlabel = r"Interannual Variance $[\frac{%s}{mon}^2]$" % vunit
        
    for ss in range(4):
        
        ax      = axs[uu,ss]
        plotvar = invar[vname].isel(season=ss).mean('ens') * dtmon**2 * mask
        if pcolor:
            pcm     = ax.pcolormesh(plotvar.lon,plotvar.lat,plotvar,
                                    transform=proj,vmin=0,vmax=vmax,
                                    cmap=cmap,zorder=-2)
        else:
            
            pcm     = ax.contourf(plotvar.lon,plotvar.lat,plotvar,
                                    transform=proj,levels=cints,
                                    cmap=cmap,extend='both')
            
            cl      = ax.contour(plotvar.lon,plotvar.lat,plotvar,
                                    transform=proj,levels=cints,
                                    colors='k',linewidths=0.75)
            ax.clabel(cl)
        
        

        if ss == 0:
            viz.add_ylabel(r"$%s \cdot \nabla %s$" % (ustr,vstr),ax=ax,fontsize=fsz_title)
        if uu == 0:
            ax.set_title(plotvar.season.data.item(),fontsize=fsz_title)
            
            
        # Plot Additional Features (Ice Edge, ETC)
        ax.plot(ds_gs2.lon.mean('mon'),ds_gs2.lat.mean('mon'),transform=proj,lw=2.5,c='cornflowerblue',ls='dashdot')

        # Plot Ice Edge
        ax.contour(icemask.lon,icemask.lat,mask_plot,colors="cyan",linewidths=2.5,
                   transform=proj,levels=[0,1],zorder=-1)
        
            
    
cb = viz.hcbar(pcm,ax=axs.flatten(),fraction=0.025)
cb.set_label(vlabel,fontsize=fsz_axis)
cb.ax.tick_params(labelsize=fsz_tick)
            
savename = "%sCESM1_ugeoterms_monvar_%s.png" % (figpath,vname,)
plt.savefig(savename,dpi=150,bbox_inches='tight')
        
# -----------------------------------------------------------------------------
#%% Quick Comparison of Regionally Averaging Term then computing monthly variance or computing monthly variance then regionally averaging

fig,axs = viz.init_monplot(1,2,figsize=(12,4))

# Plot SST
ax = axs[0]
ax.set_title("Interannual Variance by Month (SST)")

# Plot UgeoPrime (Regional Average First)
plotvar = np.array(tsms_ugeo[0]['monvars']).mean(0) * dtmon**2
label   = ugeos_names[0] + " (reg avg term)"
ax.plot(mons3,plotvar,label=label,c="navy",lw=lw)

# Plot UgeoPrime (Average Monthly Variance)
plotvar = regavg_monvar[0].mean(0) * dtmon**2
label   = ugeos_names[0] + " (reg avg monvar)"
ax.plot(mons3,plotvar,label=label,c="navy",lw=lw,ls='dashed')

# Plot UgeoBar (Regional Average First)
plotvar = np.array(tsms_ugeo[2]['monvars']).mean(0) * dtmon**2
label   = ugeos_names[2]  + " (reg avg term)"
ax.plot(mons3,plotvar,label=label,c="violet",lw=lw,ls='solid')

# Plot UgeoBar (Average Monthly Variance)
plotvar = regavg_monvar[2].mean(1) * dtmon**2
label   = ugeos_names[2] + " (reg avg monvar)"
ax.plot(mons3,plotvar,label=label,c="violet",lw=lw,ls='dashed')

ax.legend(fontsize = 10,ncol=2,bbox_to_anchor=[-.07,.45,1,1])

# Plot SSS
ax = axs[1]
ax.set_title("Interannual Variance by Month (SSS)")

# Plot UgeoPrime (Regional Average First)
plotvar = np.array(tsms_ugeo[1]['monvars']).mean(0) * dtmon**2
label   = ugeos_names[1] + " (reg avg term)"
ax.plot(mons3,plotvar,label=label,c="navy",lw=lw)

# Plot UgeoPrime (Average Monthly Variance)
plotvar = regavg_monvar[1].mean(0) * dtmon**2
label   = ugeos_names[1] + " (reg avg monvar)"
ax.plot(mons3,plotvar,label=label,c="navy",lw=lw,ls='dashed')

# Plot UgeoBar (Regional Average First)
plotvar = np.array(tsms_ugeo[3]['monvars']).mean(0) * dtmon**2
label   = ugeos_names[3]  + " (reg avg term)"
ax.plot(mons3,plotvar,label=label,c="violet",lw=lw,ls='solid')

# Plot UgeoBar (Average Monthly Variance)
plotvar = regavg_monvar[3].mean(1) * dtmon**2
label   = ugeos_names[3] + " (reg avg monvar)"
ax.plot(mons3,plotvar,label=label,c="violet",lw=lw,ls='dashed')

ax.legend(fontsize = 10,ncol=2,bbox_to_anchor=[-.07,.45,1,1])

plt.suptitle("Regional Average Term vs. Regional Average Monthly Variance\n %s" % (bbti))
#savename = "%sCESM1_ugeoterms_monvar_%s.png" % (figpath,vname,)
#plt.savefig(savename,dpi=150,bbox_inches='tight')


savename = "%sCESM1_v_SM_%s_%s_ugeo_avgeffect.png" % (figpath,comparename,bbname)
plt.savefig(savename,dpi=150,bbox_inches='tight')

#%% Do same for above, but for cesm1 and stochastic model


fig,axs = viz.init_monplot(1,2,figsize=(12,4))

# Plot for SST
ax      = axs[0]
ax.set_title("Interannual Variance by Month (SST)")

# Plot CESM1
plotvar = np.array(tsms[0]['monvars']).mean(0) 
label   = "%s" % (expnames_long[0])  + " (reg avg term)"
ax.plot(mons3,plotvar,label=label,c="k",lw=lw)

# Plot SM
plotvar = np.array(tsms[2]['monvars']).mean(0) 
label   = "%s" % (expnames_long[2])  + " (reg avg term)"
ax.plot(mons3,plotvar,label=label,c="violet",lw=lw)

# Plot CESM1 (Monvar Regavg)
plotvar = ds_regavg_monvar[0].mean(0)
label   = "%s" % (expnames_long[0])  + " (reg avg monvar)"
ax.plot(mons3,plotvar,label=label,c="k",lw=lw,ls='dashed')

# Plot CESM1 (Monvar Regavg)
plotvar = ds_regavg_monvar[2].mean(0)
label   = "%s" % (expnames_long[2])  + " (reg avg monvar)"
ax.plot(mons3,plotvar,label=label,c="violet",lw=lw,ls='dashed')



# # Plot CESM - SST
# plotvar = np.array(tsms[0]['monvars']).mean(0) - np.array(tsms[2]['monvars']).mean(0)
# label   = "%s - %s" % (expnames_long[0], expnames_long[2])
# ax.plot(mons3,plotvar,label=label,c="gray",lw=lw,ls='dashed')


ax.legend(fontsize = 8,ncol=2,bbox_to_anchor=[-.04,.32,1,1])

# Plot for SSS
ax      = axs[1]
ax.set_title("Interannual Variance by Month (SSS)")

# Plot CESM1
plotvar = np.array(tsms[1]['monvars']).mean(0) 
label   = "%s" % (expnames_long[1])  + " (reg avg term)"
ax.plot(mons3,plotvar,label=label,c="k",lw=lw)

# Plot SM
plotvar = np.array(tsms[3]['monvars']).mean(0) 
label   = "%s" % (expnames_long[3])  + " (reg avg term)"
ax.plot(mons3,plotvar,label=label,c="forestgreen",lw=lw)


# Plot CESM1 (Monvar Regavg)
plotvar = ds_regavg_monvar[1].mean(0)
label   = "%s" % (expnames_long[1])  + " (reg avg monvar)"
ax.plot(mons3,plotvar,label=label,c="k",lw=lw,ls='dashed')

# Plot CESM1 (Monvar Regavg)
plotvar = ds_regavg_monvar[3].mean(0)
label   = "%s" % (expnames_long[3])  + " (reg avg monvar)"
ax.plot(mons3,plotvar,label=label,c="forestgreen",lw=lw,ls='dashed')


# # Plot CESM - SST
# plotvar = np.array(tsms[1]['monvars']).mean(0) - np.array(tsms[3]['monvars']).mean(0)
# label   = "%s - %s" % (expnames_long[1], expnames_long[3])
# ax.plot(mons3,plotvar,label=label,c="gray",lw=lw,ls='dashed')

ax.legend(fontsize = 8,ncol=2,bbox_to_anchor=[-.04,.32,1,1])
plt.suptitle("Regional Average Term vs. Regional Average Monthly Variance\n %s" % (bbti))


savename = "%sCESM1_v_SM_%s_%s_SST_SSS_avgeffect.png" % (figpath,comparename,bbname)
plt.savefig(savename,dpi=150,bbox_inches='tight')


#%% Plot comparison (Monvar First)

fig,axs = viz.init_monplot(1,2,figsize=(12,4))

# Plot for SST
ax      = axs[0]
ax.set_title("Interannual Variance by Month (SST)")

# Plot CESM1 (Monvar Regavg)
plotvar = ds_regavg_monvar[0].mean(0)
label   = "%s" % (expnames_long[0])  #+ " (reg avg monvar)"
ax.plot(mons3,plotvar,label=label,c="k",lw=lw,ls='solid')

# Plot CESM1 (Monvar Regavg)
plotvar = ds_regavg_monvar[2].mean(0)
label   = "%s" % (expnames_long[2])  #+ " (reg avg monvar)"
ax.plot(mons3,plotvar,label=label,c="violet",lw=lw,ls='solid')

# CESM - SST
plotvar = ds_regavg_monvar[0].mean(0) - ds_regavg_monvar[2].mean(0)
label   = "%s - %s" % (expnames_long[0], expnames_long[2])
ax.plot(mons3,plotvar,label=label,c="gray",lw=lw,ls='dashed')

# Plot UgeoPrime (Average Monthly Variance)
plotvar = regavg_monvar[0].mean(0) * dtmon**2
label   = ugeos_names[0] #+ " (reg avg monvar)"
ax.plot(mons3,plotvar,label=label,c="navy",lw=lw,ls='dashed')

# Plot UgeoBar (Average Monthly Variance)
plotvar = regavg_monvar[2].mean(1) * dtmon**2
label   = ugeos_names[2] #+ " (reg avg monvar)"
ax.plot(mons3,plotvar,label=label,c="cornflowerblue",lw=lw,ls='dashed')

ax.legend(fontsize = 10,ncol=2,bbox_to_anchor=[-.125,.52,1,1])

# -----------------------------------------------------------------------------

# Plot for SSS
ax      = axs[1]
ax.set_title("Interannual Variance by Month (SSS)")

# Plot CESM1 (Monvar Regavg)
plotvar = ds_regavg_monvar[1].mean(0)
label   = "%s" % (expnames_long[1])  #+ " (reg avg monvar)"
ax.plot(mons3,plotvar,label=label,c="k",lw=lw,ls='solid')

# Plot CESM1 (Monvar Regavg)
plotvar = ds_regavg_monvar[3].mean(0)
label   = "%s" % (expnames_long[3])  #+ " (reg avg monvar)"
ax.plot(mons3,plotvar,label=label,c="forestgreen",lw=lw,ls='solid')


# Plot CESM - SST
plotvar = ds_regavg_monvar[1].mean(0) - ds_regavg_monvar[3].mean(0)
label   = "%s - %s" % (expnames_long[1], expnames_long[3])
ax.plot(mons3,plotvar,label=label,c="gray",lw=lw,ls='dashed')

# Plot UgeoPrime (Average Monthly Variance)
plotvar = regavg_monvar[1].mean(0) * dtmon**2
label   = ugeos_names[1]# + " (reg avg monvar)"
ax.plot(mons3,plotvar,label=label,c="navy",lw=lw,ls='dashed')

# Plot UgeoBar (Average Monthly Variance)
plotvar = regavg_monvar[3].mean(1) * dtmon**2
label   = ugeos_names[3] #+ " (reg avg monvar)"
ax.plot(mons3,plotvar,label=label,c="cornflowerblue",lw=lw,ls='dashed')

ax.legend(fontsize = 10,ncol=2,bbox_to_anchor=[-.125,.52,1,1])
plt.suptitle("Regional Average Term vs. Regional Average Monthly Variance\n %s" % (bbti))


savename = "%sCESM1_v_SM_%s_%s_compare_monvar_regavgmonvar.png" % (figpath,comparename,bbname)
plt.savefig(savename,dpi=150,bbox_inches='tight')

#%% Compute % Explained... (use Monvar)

fig,axs = viz.init_monplot(1,2,figsize=(12,4))

# Plot for SST
ax      = axs[0]
ax.set_title("% of Interannual Variance Explained (SST)")

# Ugeo Prime
cesm_sm_diff   = ds_regavg_monvar[0].mean(0) - ds_regavg_monvar[2].mean(0)
plot_ugeoprime = regavg_monvar[0].mean(0) * dtmon**2
plotvar_prime        = plot_ugeoprime/cesm_sm_diff
label          = ugeos_names[0]
ax.plot(mons3,plotvar_prime*100,label=label,c="navy",lw=lw,ls='solid')

# Ugeo Bar
plot_ugeobar = regavg_monvar[2].mean(1) * dtmon**2
plotvar_bar        = plot_ugeobar/cesm_sm_diff
label          = ugeos_names[2]
ax.plot(mons3,plotvar_bar*100,label=label,c="cornflowerblue",lw=lw,ls='solid')

# Total Explained
plotvar = plotvar_prime + plotvar_bar
label   = "Total Variance Relative to Var(SST) (%)"
ax.plot(mons3,plotvar*100,label=label,c="k",lw=lw,ls='solid')

ax.legend()
ax.set_ylabel("% of Mismatch Explained",fontsize=fsz_tick)

# Plot for SSS
ax      = axs[1]
ax.set_title("% of Interannual Variance Explained (SSS)")

# Ugeo Prime
cesm_sm_diff   = ds_regavg_monvar[1].mean(0) - ds_regavg_monvar[3].mean(0)
plot_ugeoprime = regavg_monvar[1].mean(0) * dtmon**2
plotvar_prime        = plot_ugeoprime/cesm_sm_diff
label          = ugeos_names[1]
ax.plot(mons3,plotvar_prime*100,label=label,c="navy",lw=lw,ls='solid')

# Ugeo Bar
plot_ugeobar = regavg_monvar[3].mean(1) * dtmon**2
plotvar_bar        = plot_ugeobar/cesm_sm_diff
label          = ugeos_names[3]
ax.plot(mons3,plotvar_bar*100,label=label,c="cornflowerblue",lw=lw,ls='solid')

# Total Explained
plotvar = plotvar_prime + plotvar_bar
label   = "Total Variance Relative to Var(SSS) (%)"
ax.plot(mons3,plotvar*100,label=label,c="k",lw=lw,ls='solid')

ax.legend()
#ax.set_ylabel("% of Mismatch Explained",fontsize=fsz_tick)

savename = "%sCESM1_v_SM_%s_%s_ugeo_explain.png" % (figpath,comparename,bbname)
plt.savefig(savename,dpi=150,bbox_inches='tight')


#%%