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


#%%  Indicate Experients (copying upper setion of viz_regional_spectra )

regionset       = "SSSCSU"
comparename     = "SST_SSS_Paper_Draft01_Original"
expnames        = ["SST_CESM","SSS_CESM","SST_Draft03_Rerun_QekCorr","SSS_Draft03_Rerun_QekCorr"]
expnames_long   = ["CESM1 (SST)","CESM1 (SSS)","Stochastic Model (SST)","Stochastic Model (SSS)"]
expnames_short  = ["CESM1_SST","CESM1_SSS","SM_SST","SM_SSS"]
ecols           = ["firebrick","navy","hotpink",'cornflowerblue']
els             = ["solid","solid",'dashed','dashed']
emarkers        = ["o","d","o","d"]


cesm_exps       = ["SST_CESM","SSS_CESM","SST_cesm2_pic","SST_cesm1_pic",
                  "SST_cesm1le_5degbilinear","SSS_cesm1le_5degbilinear",]
#%% Load the stochastic model output (using sm output loader)
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


ugeo_names_2 = [
    r"$\overline{u_{geo}} \cdot \nabla T'$",
    r"$u_{geo}' \cdot \nabla \overline{T}$",
    r"$\overline{u_{geo}} \cdot \nabla S'$",
    r"$u_{geo}' \cdot \nabla \overline{S}$",
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

#sel_box    =  [-40,-30,40,50] # NAC
#bbname     = "NAC"

#sel_box    = [-70,-55,35,40] # Sargasso Sea SSS CSU
#bbname     = "SAR"

sel_box    =  [-40,-25,50,60] # Irminger
bbname     = "IRM"

sel_point  = [-30,50]
bbname     = "SPGPoint"


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
ax              = axs[0]
ax.set_title("% of Interannual Variance Explained (SST)")

# Ugeo Prime
cesm_sm_diff    = ds_regavg_monvar[0].mean(0) - ds_regavg_monvar[2].mean(0)
plot_ugeoprime  = regavg_monvar[0].mean(0) * dtmon**2
plotvar_prime   = plot_ugeoprime/cesm_sm_diff
label           = ugeos_names[0]
ax.plot(mons3,plotvar_prime*100,label=label,c="navy",lw=lw,ls='solid')

# Ugeo Bar
plot_ugeobar    = regavg_monvar[2].mean(1) * dtmon**2
plotvar_bar     = plot_ugeobar/cesm_sm_diff
label           = ugeos_names[2]
ax.plot(mons3,plotvar_bar*100,label=label,c="cornflowerblue",lw=lw,ls='solid')

# Total Explained
plotvar         = plotvar_prime + plotvar_bar
label           = "Total Variance Relative to Var(SST) (%)"
ax.plot(mons3,plotvar*100,label=label,c="k",lw=lw,ls='solid')

ax.legend()

ax.set_ylim([0,110])
ax.set_ylabel("% of Mismatch Explained",fontsize=fsz_tick)

# Plot for SSS
ax              = axs[1]
ax.set_title("% of Interannual Variance Explained (SSS)")

# Ugeo Prime
cesm_sm_diff    = ds_regavg_monvar[1].mean(0) - ds_regavg_monvar[3].mean(0)
plot_ugeoprime  = regavg_monvar[1].mean(0) * dtmon**2
plotvar_prime   = plot_ugeoprime/cesm_sm_diff
label           = ugeos_names[1]
ax.plot(mons3,plotvar_prime*100,label=label,c="navy",lw=lw,ls='solid')

# Ugeo Bar
plot_ugeobar    = regavg_monvar[3].mean(1) * dtmon**2
plotvar_bar     = plot_ugeobar/cesm_sm_diff
label           = ugeos_names[3]
ax.plot(mons3,plotvar_bar*100,label=label,c="cornflowerblue",lw=lw,ls='solid')

# Total Explained
plotvar         = plotvar_prime + plotvar_bar
label           = "Total Variance Relative to Var(SSS) (%)"
ax.plot(mons3,plotvar*100,label=label,c="k",lw=lw,ls='solid')

ax.legend()
#ax.set_ylabel("% of Mismatch Explained",fontsize=fsz_tick)

ax.set_ylim([0,110])

savename        = "%sCESM1_v_SM_%s_%s_ugeo_explain.png" % (figpath,comparename,bbname)
plt.savefig(savename,dpi=150,bbox_inches='tight')

#%% Compute % Explained but use Reg Avg (then MonVar)

fig,axs = viz.init_monplot(1,2,figsize=(12,4))

# Plot for SST
ax              = axs[0]
ax.set_title("% of Interannual Variance Explained (SST)")

# Ugeo Prime
cesm_sm_diff    = np.array(tsms[0]['monvars']).mean(0)  - np.array(tsms[2]['monvars']).mean(0)  #ds_regavg_monvar[0].mean(0) - ds_regavg_monvar[2].mean(0)
plot_ugeoprime  = np.array(tsms_ugeo[0]['monvars']).mean(0) * dtmon**2 #regavg_monvar[0].mean(0) * dtmon**2
plotvar_prime   = plot_ugeoprime/cesm_sm_diff
label           = ugeos_names[0]
ax.plot(mons3,plotvar_prime*100,label=label,c="navy",lw=lw,ls='solid')

# Ugeo Bar
plot_ugeobar    = np.array(tsms_ugeo[2]['monvars']).mean(0) * dtmon**2#regavg_monvar[2].mean(1) * dtmon**2
plotvar_bar     = plot_ugeobar/cesm_sm_diff
label           = ugeos_names[2]
ax.plot(mons3,plotvar_bar*100,label=label,c="cornflowerblue",lw=lw,ls='solid')

# Total Explained
plotvar         = plotvar_prime + plotvar_bar
label           = "Total Variance Relative to Var(SST) (%)"
ax.plot(mons3,plotvar*100,label=label,c="k",lw=lw,ls='solid')

ax.legend()

ax.set_ylim([0,20])
ax.set_ylabel("% of Mismatch Explained",fontsize=fsz_tick)

# Plot for SSS
ax              = axs[1]
ax.set_title("% of Interannual Variance Explained (SSS)")

# Ugeo Prime
cesm_sm_diff    = np.array(tsms[1]['monvars']).mean(0)  - np.array(tsms[3]['monvars']).mean(0)#ds_regavg_monvar[1].mean(0) - ds_regavg_monvar[3].mean(0)
plot_ugeoprime  = np.array(tsms_ugeo[1]['monvars']).mean(0) * dtmon**2 #regavg_monvar[1].mean(0) * dtmon**2
plotvar_prime   = plot_ugeoprime/cesm_sm_diff
label           = ugeos_names[1]
ax.plot(mons3,plotvar_prime*100,label=label,c="navy",lw=lw,ls='solid')

# Ugeo Bar
plot_ugeobar    = np.array(tsms_ugeo[3]['monvars']).mean(0) * dtmon**2
plotvar_bar     = plot_ugeobar/cesm_sm_diff
label           = ugeos_names[3]
ax.plot(mons3,plotvar_bar*100,label=label,c="cornflowerblue",lw=lw,ls='solid')

# Total Explained
plotvar         = plotvar_prime + plotvar_bar
label           = "Total Variance Relative to Var(SSS) (%)"
ax.plot(mons3,plotvar*100,label=label,c="k",lw=lw,ls='solid')

ax.legend()
#ax.set_ylabel("% of Mismatch Explained",fontsize=fsz_tick)

ax.set_ylim([0,20])

savename       = "%sCESM1_v_SM_%s_%s_ugeo_explain_regavg_first.png" % (figpath,comparename,bbname)
plt.savefig(savename,dpi=150,bbox_inches='tight') 


#%% Examine Regional Sensivity of montly variance (see maximum value across a region)

# Start from 1 point and expand a region
# For each expansion
# (1) Compute Regional Average (SST, SSS, ugeoadv terms)
# (2) Compute Variance and Monthly Variance
# (3) Look at CESM Diff and compare the proportion explained by the ugeoadv term

#%% First, let's start by making a locator

origin_point   = [-30,50]

expand_size = 1  # Degrees of expansion
limit       = 20 # Maximum Degree of Expansion
nexpand     = int(limit/expand_size)

specialmark = np.arange(0,limit+5,5)

expand_label = [2*expand_size*ii for ii in range(nexpand)]

bboxes = [[origin_point[0] - expand_size*ii,
          origin_point[0] + expand_size*ii,
          origin_point[1] - expand_size*ii,
          origin_point[1] + expand_size*ii,
          ] for ii in range(nexpand)]

# Make a Locator Plot
fig,ax,mdict = viz.init_orthomap(1,1,bboxplot=bboxplot,figsize=(28,10))
ax           = viz.add_coast_grid(ax,bbox=bboxplot,fill_color="lightgray",fontsize=24)

ax.plot(origin_point[0],origin_point[1],transform=proj,marker="x",markersize=15)

for b,bb in enumerate(bboxes):
    if b in specialmark:
        ls = 'dotted'
        lw = 2.5
    else:
        ls = 'solid'
        lw = 1
    viz.plot_box(bb,ax=ax,linestyle=ls,linewidth=lw)
    
# Plot Currents
qint  = 2
plotu = ds_uvel.UVEL.mean('ens').mean('month').values
plotv = ds_vvel.VVEL.mean('ens').mean('month').values
ax.quiver(tlon[::qint,::qint],tlat[::qint,::qint],plotu[::qint,::qint],plotv[::qint,::qint],
          color='navy',transform=proj,alpha=0.25,zorder=-4)

invars       = ds_all + ds_ugeos
invars_names = expnames_long + ugeos_names 

#%% lets Perform the analysis

totalvars = np.zeros((nexpand,len(invars)))
monvars   = np.zeros((nexpand,len(invars),12))

for ex in tqdm.tqdm(range(nexpand)):
    
    bbin = bboxes[ex]
    
    for vv in range(len(invars)):
        
        
        invar = invars[vv]
        
        # Format dimensions
        if 'ensemble' in list(invar.dims):
            invar = invar.rename(dict(ensemble='ens'))
        if 'run' in list(invar.dims):
            invar = invar.rename(dict(run='ens'))
        
        if bbin[0] - bbin[1] == 0:
            dsreg = proc.selpt_ds(invar,bbin[0],bbin[2])
        else:
            dsreg = proc.sel_region_xr(invar,bbin).mean('lat').mean('lon')
        
        # Copy the total variance
        totalvars[ex,vv] = dsreg.var('time').mean('ens').data.item()
        monvars[ex,vv,:] = dsreg.groupby('time.month').var('time').mean('ens').data
        
#%% Now compute the differences... (this is going to be numbering hell)  
        


cesm_minus_sm_totalvar_sst = totalvars[:,0] - totalvars[:,2]
cesm_minus_sm_totalvar_sss = totalvars[:,1] - totalvars[:,3]


perc_expl_ugeoprime_sst = np.abs((totalvars[:,4] * dtmon**2)/cesm_minus_sm_totalvar_sst)
perc_expl_ugeoprime_sss = np.abs((totalvars[:,5]*  dtmon**2)/cesm_minus_sm_totalvar_sss)

perc_expl_ugeobar_sst   = np.abs((totalvars[:,6]* dtmon**2)/cesm_minus_sm_totalvar_sst)
perc_expl_ugeobar_sss   = np.abs((totalvars[:,7]* dtmon**2)/cesm_minus_sm_totalvar_sss)


#%% Plot Difference in Variance Explained by Expansion Size

fig,ax = plt.subplots(1,1,constrained_layout=True)

ax.plot(expand_label,perc_expl_ugeobar_sst* 100,label="%s, SST" % invars_names[6],marker='x')
ax.plot(expand_label,perc_expl_ugeobar_sss* 100,label="%s, SSS" % invars_names[7],marker='x')

ax.plot(expand_label,perc_expl_ugeoprime_sst* 100,label="%s, SST" % invars_names[4],ls='solid',c='darkblue',marker='x')
ax.plot(expand_label,perc_expl_ugeoprime_sss* 100,label="%s, SSS" % invars_names[5],ls='dotted',c='hotpink',marker='x')
ax.legend()

ax.set_ylabel("% of Discrepancy Explained (Term / [CESM1 - SM])")
ax.set_xlabel("Bounding Box Size (Degrees)")

ax.axhline([0],c='k',lw=0.75)
ax.set_title("Central Point: %s" % (origin_point))

ax.vlines(specialmark,ymin=-1,ymax=125,ls='dotted',colors='k',lw=2.5)

# =================================================
#%% Make Geostrophic Advection Plot (Hypothetical)
# =================================================

pointmode  = True

if pointmode:
    points  = [[-65,36],
               [-34,46],
               [-36,58],
               ]
else:
    # Same Bounding Boxes as the rparams
    bboxes  = [
        [-70,-55,35,40],
        [-40,-30,40,50],
        [-40,-25,50,60],
        ]


pointnames = ["SAR",
              "NAC",
              "IRM"]

locstring_all = [proc.make_locstring(pt[0],pt[1]) for pt in points]

npts    = len(points)

# Make a Locator Plot
fig,ax,mdict = viz.init_orthomap(1,1,bboxplot=bboxplot,figsize=(28,10))
ax           = viz.add_coast_grid(ax,bbox=bboxplot,fill_color="lightgray",fontsize=24)

for nn in range(npts):
    origin_point = points[nn]
    ax.plot(origin_point[0],origin_point[1],transform=proj,marker="d",markersize=15)

invars       = ds_all + ds_ugeos
invars_names = expnames_long + ugeos_names 

#%% lets Perform the analysis

totalvars_pt = np.zeros((npts,len(invars)))
monvars_pt  = np.zeros((npts,len(invars),12))

for ex in tqdm.tqdm(range(npts)):
    
    
    
    for vv in range(len(invars)):
        
        
        invar = invars[vv]
        
        # Format dimensions
        if 'ensemble' in list(invar.dims):
            invar = invar.rename(dict(ensemble='ens'))
        if 'run' in list(invar.dims):
            invar = invar.rename(dict(run='ens'))
        
        if pointmode:
            bbin = points[ex]
            dsreg = proc.selpt_ds(invar,bbin[0],bbin[1])
        else:
            bbin  = bboxes[ex]
            dsreg = proc.sel_region_xr(invar,bbin).mean('lat').mean('lon')

        
        # Copy the total variance
        totalvars_pt[ex,vv] = dsreg.var('time').mean('ens').data.item()
        monvars_pt[ex,vv,:] = dsreg.groupby('time.month').var('time').mean('ens').data

#%% Compute Some Differences

cesm_sm_diff_monvar_sst = monvars_pt[:,0,:] - monvars_pt[:,2,:]
cesm_sm_diff_monvar_sss = monvars_pt[:,1,:] - monvars_pt[:,3,:]

perc_expl_ugeoprime_sst = (monvars_pt[:,4] * dtmon**2)/cesm_sm_diff_monvar_sst
perc_expl_ugeoprime_sss = (monvars_pt[:,5]*  dtmon**2)/cesm_sm_diff_monvar_sss

perc_expl_ugeobar_sst   = (monvars_pt[:,6]* dtmon**2)/cesm_sm_diff_monvar_sst
perc_expl_ugeobar_sss   = (monvars_pt[:,7]* dtmon**2)/cesm_sm_diff_monvar_sss


perc_ugeo_total_sst = perc_expl_ugeoprime_sst + perc_expl_ugeobar_sst
perc_ugeo_total_sss = perc_expl_ugeoprime_sss + perc_expl_ugeobar_sss

#%% Make the monthly plot for the figure but at the individual points

fig,axs = viz.init_monplot(2,3,figsize=(20,11.5))

for vv in range(2): # Loop by Variable
    
    if vv == 0:
        
        vname = "SST ($\degree C$)"
    else:
        vname = "SSS ($psu$)"
    
    for rr in range(3): # Loop by Region
        
        ax = axs[vv,rr]
        
        if vv==0:
            if pointmode:
                ax.set_title("%s (%s)" % (pointnames[rr],locstring_all[rr][1]),fontsize=fsz_title)
            else:
                ax.set_title("%s" % (pointnames[rr]),fontsize=fsz_title)
            
        if rr==0:
            viz.add_ylabel(vname,ax=ax,x=-0.15,fontsize=fsz_title)
            
        # Plot total percentage explained
        
        ax2 = ax.twinx()
        if pointmode:
            ax2.set_ylim([0,200])
        else:
            ax2.set_ylim([0,200])
        if vv == 0:
            plotvar = perc_ugeo_total_sst[rr,:] * 100
        else:
            plotvar = perc_ugeo_total_sss[rr,:] * 100
        plotvar = np.abs(plotvar)
        ax2.bar(mons3,plotvar,zorder=-1,alpha=0.5,color='hotpink',edgecolor="k")
        ax2.axhline([100],ls='dashed',color="k")
        if rr == 2:
            ax2.set_ylabel("% Explained by $U_{geo}$",fontsize=fsz_axis)
            ax2.yaxis.label.set_color('hotpink')
        ax2.spines['right'].set_color('hotpink')
        ax2.tick_params(axis='y', colors='hotpink')
        
        for ii in range(3): # Plot CESM-SM, 
            
            if ii == 0: # Plot CESM1 - SM
                c  = "red"
                nm = "CESM1 - Stochastic Model" 
                if vv == 0:
                    plotvar = cesm_sm_diff_monvar_sst[rr,:]
                else:
                    plotvar = cesm_sm_diff_monvar_sss[rr,:]
            elif ii == 1: # Plot Ugeo Prime
                c  = 'cornflowerblue'
                nm = "$u'$"
                if vv == 0:
                    plotvar = monvars_pt[rr,4,:] * dtmon**2
                elif vv == 1:
                    plotvar = monvars_pt[rr,5,:] * dtmon**2
            elif ii == 2: # Plot Ugeo Bar
                c = 'teal'
                nm = "$\overline{u}$"
                if vv == 0:
                    plotvar = monvars_pt[rr,6,:] * dtmon**2
                elif vv == 1:
                    plotvar = monvars_pt[rr,7,:] * dtmon**2

            ax.plot(mons3,plotvar,c=c,label=nm,lw=2.5,marker='o',zorder=-4)
            
            if vv == 0 and rr == 0:
                ax.legend(fontsize=9)
            
            # Move axis
            ax.set_zorder(ax2.get_zorder()+1)
            ax.patch.set_visible(False) 
            
            ax.tick_params(axis='both', which='major', labelsize=fsz_tick)
            ax2.tick_params(axis='both', which='major', labelsize=fsz_tick)

savename       = "%sUgeo_Contribution_pointmode%i.png" % (figpath,pointmode)
plt.savefig(savename,dpi=150,bbox_inches='tight')
        
#%% Plot Monthly Contribution of Ugeo for SST/SSS, both prime and sbar

fsz_axis  = 28
vnames    = ["SST","SSS"]
ugeo_names = ["ubar","uprime"]
fig,axs,_ = viz.init_orthomap(4,4,bboxplot,figsize=(28,20))

for ax in axs.flatten():
    ax = viz.add_coast_grid(ax,bbox=bboxplot,fill_color="lightgray",fontsize=24)

#irow = 0
for vv in range(2):

    vname = vnames[vv]
    if vv == 0:
        axin = axs[:2,:]
        vmax = 0.5
        cmap = 'cmo.thermal'
        cints = np.arange(0,1.5,0.2)
        ccol = "k"
    else:
        axin = axs[2:,:]
        vmax = 0.075
        cmap = 'cmo.rain'
        cints = np.arange(0,1.5,0.05)
        ccol = 'w'
        
    
    for ui in range(2):
        
        if ui == 0:
            ugeoin = ugeobar_savg
            ulab   = "\overline{u}"
        elif ui == 1:
            ugeoin = ugeoprime_savg
            ulab   = "u'"
            
            
        for sid in range(4):
            
            ax      = axin[ui,sid]
            #ax.set_title("%s,%s,%i" % (vnames[vv],ugeo_names[ui],sid))
            lab     = "%s,$%s$" % (vnames[vv],ulab)
            #print()
            plotvar = np.sqrt(ugeoin[vname].isel(season=sid).mean('ens')) * dtmon * mask
            pcm     = ax.pcolormesh(plotvar.lon,plotvar.lat,plotvar,
                                    transform=proj,vmin=0,vmax=vmax,cmap=cmap)
            
            cl = ax.contour(plotvar.lon,plotvar.lat,plotvar,levels=cints,
                       transform=proj,colors=ccol,linewidths=0.75)
            ax.clabel(cl,fontsize=fsz_tick)
            
            
            # Plot Additional Features (Ice Edge, ETC)
            ax.plot(ds_gs2.lon.mean('mon'),ds_gs2.lat.mean('mon'),transform=proj,lw=2.5,c='cornflowerblue',ls='dashdot')

            # Plot Ice Edge
            ax.contour(icemask.lon,icemask.lat,mask_plot,colors="cyan",linewidths=2.5,
                       transform=proj,levels=[0,1],zorder=-1)
            
            
            
            
            if sid == 0:
                viz.add_ylabel(lab,ax=ax,fontsize=fsz_axis)
            if ui == 0 and vv == 0:
                ax.set_title(plotvar.season.item(),fontsize=fsz_axis)
    cb = fig.colorbar(pcm,ax=axin.flatten(),fraction=0.015,pad=0.01)
    cb.ax.tick_params(labelsize=fsz_tick)
    cb.set_label("$%s$ month$^{-1}$" % vunit,fontsize=fsz_axis)
    
savename       = "%sUgeo_Contribution_Pointwise_Seasonal_Average.png" % (figpath)
plt.savefig(savename,dpi=150,bbox_inches='tight')    
            #irow+=1
            #print(irow)

#%% Quickly Check the seasonal range of each variable

fig,axs,_ = viz.init_orthomap(1,4,bboxplot,figsize=(28,6))

for ax in axs.flatten():
    ax = viz.add_coast_grid(ax,bbox=bboxplot,fill_color="lightgray",fontsize=24)

irow = 0
for vv in range(2):

    vname = vnames[vv]
    if vv == 0:
        vmax = 0.5
        cmap = 'cmo.thermal'
        vunit = '\degree C'
    else:
        vmax = 0.075
        cmap = 'cmo.rain'
        vunit = "psu"
    
    for ui in range(2):
        
        if ui == 0:
            ugeoin = ugeobar_monvar
            ulab   = "\overline{u}"
        elif ui == 1:
            ugeoin = ugeoprime_monvar
            ulab   = "u'"
        
        
        ax      = axs[irow]
        
        invar   = np.sqrt(ugeoin[vname].mean('ens')) * dtmon
        plotvar = (invar.max('month') - invar.min('month')) * mask
        
        
        pcm     = ax.pcolormesh(plotvar.lon,plotvar.lat,plotvar,
                                transform=proj,vmin=0,vmax=vmax,cmap=cmap)
        cb = viz.hcbar(pcm,ax=ax)
        cb.ax.tick_params(labelsize=fsz_tick)
        cb.set_label("$%s$ month$^{-1}$" % vunit,fontsize=fsz_axis)
        lab     = "%s,$%s$" % (vnames[vv],ulab)
        ax.set_title(lab,fontsize=fsz_axis)

        irow +=1

plt.suptitle("Seasonal Range in Geostrophic Advection Terms",fontsize=fsz_axis)

savename       = "%sUgeo_Contribution_Pointwise_Seasonal_Range.png" % (figpath)
plt.savefig(savename,dpi=150,bbox_inches='tight')  
# for vv in range(4):
#     ax = 

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

#%%




#         ax      = axs[irow]
        
#         
#         plotvar = (invar.max('month') - invar.min('month')) * mask
        
        
#         pcm     = ax.pcolormesh(plotvar.lon,plotvar.lat,plotvar,
#                                 transform=proj,vmin=0,vmax=vmax,cmap=cmap)
#         cb = viz.hcbar(pcm,ax=ax)
#         cb.ax.tick_params(labelsize=fsz_tick)
#         cb.set_label("$%s$ month$^{-1}$" % vunit,fontsize=fsz_axis)
#         lab     = "%s,$%s$" % (vnames[vv],ulab)
#         ax.set_title(lab,fontsize=fsz_axis)

    
    
    
    

#%%