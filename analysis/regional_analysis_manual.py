#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Manually do some regional analysis (ACF average)

Created on Thu Jul  4 15:22:32 2024

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
fsz_title                       = 16

rhocrit                         = proc.ttest_rho(0.05,2,86)

proj                            = ccrs.PlateCarree()


#%%  Indicate Experients (copying upper setion of viz_regional_spectra )


# #  Same as comparing lbd_e effect, but with Evaporation forcing corrections
#regionset       = "SSSCSU"
comparename     = "SSS_Paper_Draft01"
expnames        = ["SSS_EOF_LbddCorr_Rerun_lbdE_neg","SSS_EOF_LbddCorr_Rerun","SSS_EOF_LbddCorr_Rerun_NoLbdd","SSS_CESM"]
expnames_long   = ["Stochastic Model (sign corrected + $\lambda^e$)","Stochastic Model","Stochastic Model (No Detrainment Damping)","CESM1"]
expnames_short  = ["SM_lbde_neg","SM_lbde","SM","CESM"]
ecols           = ["magenta","forestgreen","goldenrod","k"]
els             = ['dotted',"solid",'dashed','solid']
emarkers        = ['+',"d","x","o"]

# regionset       = "SSSCSU"
# comparename     = "SST_SSS_Paper_TCM_Coarse"
# expnames        = ["SST_cesm1le_5degbilinear","SSS_cesm1le_5degbilinear","SST_CESM1_5deg_lbddcoarsen_rerun","SSS_CESM1_5deg_lbddcoarsen"]
# expnames_long   = ["CESM1 (SST)","CESM1 (SSS)","Stochastic Model (SST)","Stochastic Model (SSS)"]
# expnames_short  = ["CESM1_SST","CESM1_SSS","SM_SST","SM_SSS"]
# ecols           = ["firebrick","navy","hotpink",'cornflowerblue']
# els             = ["solid","solid",'dashed','dashed']
# emarkers        = ["o","d","o","d"]

# regionset       = "SSSCSU"
# comparename     = "SST_SSS_Paper_Draft01_Original"
# expnames        = ["SST_CESM","SSS_CESM","SST_EOF_LbddCorr_Rerun","SSS_EOF_LbddCorr_Rerun_lbdE_neg"]
# expnames_long   = ["CESM1 (SST)","CESM1 (SSS)","Stochastic Model (SST)","Stochastic Model (SSS)"]
# expnames_short  = ["CESM1_SST","CESM1_SSS","SM_SST","SM_SSS"]
# ecols           = ["firebrick","navy","hotpink",'cornflowerblue']
# els             = ["solid","solid",'dashed','dashed']
# emarkers        = ["o","d","o","d"]

# regionset       = "SSSCSU"
# comparename     = "SST_SSS_Paper_TCM_SM_Coarse_v_Original"
# expnames        = ["SST_EOF_LbddCorr_Rerun","SSS_EOF_LbddCorr_Rerun_lbdE_neg","SST_CESM1_5deg_lbddcoarsen_rerun","SSS_CESM1_5deg_lbddcoarsen"]
# expnames_long   = ["Original (SST)","Original (SSS)","Coarsened (SST)","Coarsened (SSS)"]
# expnames_short  = ["Ori_SST","Ori_SSS","Coarse_SST","Coarse_SSS"]
# ecols           = ["orange","violet","hotpink",'cornflowerblue']
# els             = ["solid","solid",'dashed','dashed']
# emarkers        = ["o","d","o","d"]


# regionset       = "SSSCSU"
# comparename     = "SSS_TCM_lbdE_Effect"
# expnames        = ["SSS_EOF_LbddCorr_Rerun_lbdE_neg","SSS_EOF_LbddCorr_Rerun","SSS_CESM"] 
# expnames_long   = ["Stochastic Model (SST-Evaporation Feedback)","Stochastic Model","CESM",]
# expnames_short  = ["SM_lbde","SM","CESM1"]
# ecols           = ["magenta","forestgreen","k"]
# els             = ["dotted","dashed",'solid']
# emarkers        = ["o","d","x"]

#regionset       = "SSSCSU"
comparename     = "SST_CESM1_v_CESM2_PIC"
#expnames        = ["SST_cesm2_pic_noQek","SST_EOF_LbddCorr_Rerun_NoLbdd_NoQek","SST_cesm2_pic","SST_CESM"]
expnames        = ["SST_cesm2_pic_noQek","SST_cesm1pic_StochmodPaperRun","SST_cesm2_pic","SST_cesm1_pic"]
expnames_long   = ["Stochastic Model (CESM2 Params)","Stochastic Model (CESM1 Params)","CESM2","CESM1"]
expnames_short  = ["SM_2","SM_1","CESM2","CESM1"]
ecols           = ["cornflowerblue","gray","navy","k"]
els             = ['dashed',"dashed",'solid','solid']
emarkers        = ['+',"d","x","o"]

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

#%% Load some variables to plot 

# Load Current
ds_uvel,ds_vvel = dl.load_current()

# load SSS Re-emergence index (for background plot)
ds_rei = dl.load_rei("SSS_CESM",output_path).load().rei

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

#%% Plot Locator and Bounding Box w.r.t. the currents

#sel_box   = [-70,-55,35,40] # Sargasso Sea SSS CSU
#sel_box    =  [-40,-30,40,50] # NAC
#sel_box    =  [-40,-25,50,60] # Irminger

sel_box    = [-50,-25,50,60] # yeager 2012 SPG

bbfn,bbti = proc.make_locstring_bbox(sel_box)

#sel_box = [-40,-25,50,60] 
#sel_box   = [-45,-38,20,25] # Azores High Proximity

qint      = 2

# Restrict REI for plotting
selmons   = [1,2]
iyr       = 0
plot_rei  = ds_rei.isel(mon=selmons,yr=iyr).mean('mon').mean('ens')
rei_cints = np.arange(0,0.55,0.05)
rei_cmap  = 'cmo.deep' 

# Initialize Plot and Map
fig,ax,_    = viz.init_orthomap(1,1,bboxplot,figsize=(18,6.5))
ax          = viz.add_coast_grid(ax,bbox=bboxplot,fill_color="lightgray")


# Plot Currents
plotu = ds_uvel.UVEL.mean('ens').mean('month').values
plotv = ds_vvel.VVEL.mean('ens').mean('month').values
tlon = ds_uvel.TLONG.mean('ens').data
tlat = ds_uvel.TLAT.mean('ens').data

ax.quiver(tlon[::qint,::qint],tlat[::qint,::qint],plotu[::qint,::qint],plotv[::qint,::qint],
          color='navy',transform=proj,alpha=0.75)

l1 = viz.plot_box(sel_box)

# Plot Re-emergence INdex
ax.contourf(plot_rei.lon,plot_rei.lat,plot_rei,cmap='cmo.deep',transform=proj,zorder=-1)


# Plot Gulf Stream Position
ax.plot(ds_gs.lon,ds_gs.lat.mean('ens'),transform=proj,lw=1.75,c="k")


ax.set_title("Bounding Box Test: %s" % (str(sel_box)),fontsize=fsz_title)



#%% Perform Regional Subsetting and Analysis

dsreg     = [proc.sel_region_xr(ds,sel_box) for ds in ds_all]
regavg_ts = [ds.mean('lat').mean('lon').data for ds in dsreg]

tsm_byexp = []
for e in range(nexps):
    ts_in   = regavg_ts[e]
    if len(ts_in.shape) == 1:
        ts_in = ts_in[None,:]
    nrun    = ts_in.shape[0]
    print(nrun)
    
    ts_list = [ts_in[ii,:] for ii in range(nrun)] 
    print(ts_in.shape)
    
    tsm     = scm.compute_sm_metrics(ts_list)
    
    tsm_byexp.append(tsm)


print(tsm.keys())

#%% Visualzie the regional ACF

lags        = np.arange(37)
xtks        = lags[::3]
kmonth      = 1
plot_env    = True
plot_stderr = True

fig,ax= plt.subplots(1,1,constrained_layout=True,figsize=(10,4.5))
ax,_  = viz.init_acplot(kmonth,xtks,lags,ax=ax,title="")

for ex in range(nexps):
    
    acfexp = np.array(tsm_byexp[ex]['acfs'][kmonth]) # Run x Lag
    
    mu    = acfexp.mean(0)
    if plot_stderr:
        sigma   = proc.calc_stderr(acfexp,0)
    else:
        sigma   =  acfexp.std(0) 
    ax.plot(lags,mu,label=expnames_long[ex],
            c=ecols[ex],ls=els[ex],lw=2.5)
    ax.fill_between(lags,mu-sigma,mu+sigma,label="",alpha=0.1,color=ecols[ex])
ax.legend()
ax.set_title("Bounding Box (%s), %s Autocorrelation Function" % (bbti,mons3[kmonth]))
savename = "%sRegional_ACF_%s_%s_mon%02i.png" % (figpath,comparename,bbfn,kmonth+1)
plt.savefig(savename,dpi=150,bbox_inches='tight')

#%% Check the Monthly Variance

fig,axs = viz.init_monplot(1,2,figsize=(12.5,4.5))#plt.subplots(1,1,constrained_layout=True,figsize=(12,4.5))

for ex in range(nexps):
    
    mvplot = np.array(tsm_byexp[ex]['monvars']) # Run x Lag
    exname = expnames_long[ex]
    
    if "SST" in expnames[ex]:
        ax = axs[0]
        
        #vunit = "$\degree C$"
        splabel = "SST ($\degree C^2$"
    else:
        ax = axs[1]
        vunit = "psu"
        splabel = "SSS ($psu^2$"
        
    mu    = mvplot.mean(0)
    if plot_stderr:
        sigma   = proc.calc_stderr(mvplot,0)
    else:
        sigma   =  mvplot.std(0) 
    ax.plot(mons3,mu,label=expnames_long[ex],
            c=ecols[ex],ls=els[ex])
    ax.fill_between(mons3,mu-sigma,mu+sigma,label="",alpha=0.1,color=ecols[ex])
    
    viz.label_sp(splabel,ax=ax,usenumber=True)
    ax.legend(loc='upper right',ncols=1)   
    
savename = "%sRegional_Monvar_%s_%s_mon%02i.png" % (figpath,comparename,bbfn,kmonth+1)
plt.savefig(savename,dpi=150,bbox_inches='tight')


#%% Debug; Check for a residual season cycle and trend

ds   = dsreg[0]
test = ds.isel(run=0).mean('lat').mean('lon')

# ---------------------------------
#%% Examine some pointwise features
# ----------------------------------

ds_ptvar = [ds.groupby('time.month').var('time') for ds in ds_all]

ds_ptvar_all = [ds.var('time') for ds in ds_all]

#%% Poitnwise Variance Difference (By Month)


kmonth = 2

for kmonth in range(12):
    fig,axs,_    = viz.init_orthomap(2,2,bboxplot,figsize=(18,15))
    
    for ax in axs.flatten():
        ax          = viz.add_coast_grid(ax,bbox=bboxplot,fill_color="lightgray")
    
    for ii in range(4):
        ax = axs.flatten()[ii]
        
        if ii == 0: # Plot Stochastic Model Pattern (SST)
            title   = "Variance (SST, Stochastic Model)"
            cmap    = "cmo.thermal"
            vlims   = [0,1]
            plotvar = ds_ptvar[2].mean('run').isel(month=kmonth)
        elif ii == 1: # SST Diff
            title   = "Diff. (Stochastic Model - CESM)"
            cmap    = 'cmo.balance'
            plotvar = ds_ptvar[2].mean('run').isel(month=kmonth) - ds_ptvar[0].mean('ens').isel(month=kmonth)
            vlims   = [-.5,.5]
        elif ii == 2:
            title   = "Variance (SSS, Stochastic Model)"
            cmap    = "cmo.haline"
            plotvar = ds_ptvar[3].mean('run').isel(month=kmonth)
            vlims   = [0,0.025]
        elif ii == 3:
            title   = "Variance Diff. (Stochastic Model - CESM)"
            cmap    = 'cmo.balance'
            plotvar = ds_ptvar[3].mean('run').isel(month=kmonth) - ds_ptvar[1].mean('ens').isel(month=kmonth)
            vlims   = [-.025,0.025]
        
        pcm = ax.pcolormesh(plotvar.lon,plotvar.lat,plotvar * maskin.squeeze(),
                            transform=proj,cmap=cmap,
                            vmin=vlims[0],vmax=vlims[1])
        
        cb = viz.hcbar(pcm,ax=ax)
        ax.set_title(title,fontsize=fsz_title)
            
    plt.suptitle("Variance Difference for %s" % (mons3[kmonth]),fontsize=fsz_title+10)    
    
    savename = "%sPointwise_Monvar_Diff_%s_%s_mon%02i.png" % (figpath,comparename,bbfn,kmonth+1)
    plt.savefig(savename,dpi=150,bbox_inches='tight')
    
#%% Pointwise Variance (All Months)

fig,axs,_    = viz.init_orthomap(2,3,bboxplot,figsize=(22,12.5))

for ax in axs.flatten():
    ax          = viz.add_coast_grid(ax,bbox=bboxplot,fill_color="lightgray")

for ii in range(6):
    
    ax = axs.flatten()[ii]
    if ii == 0: # SST, Stochastic Model
        title   = "var(SST), Stochastic Model"
        cmap    = "cmo.thermal"
        vlims   = [0,1]
        plotvar = ds_ptvar_all[2].mean('run')
    elif ii == 1: # CESM
        title   = "var(SST), CESM1"
        cmap    = "cmo.thermal"
        vlims   = [0,1]
        plotvar = ds_ptvar_all[0].mean('ens')
    elif ii == 2: 
        title   = "Stochastic Model - CESM"
        cmap    = 'cmo.balance'
        plotvar = ds_ptvar_all[2].mean('run') - ds_ptvar_all[0].mean('ens')
        vlims   = [-.5,.5]
    
    if ii == 3: # SSS, Stochastic Model
        title   = "var(SSS), Stochastic Model"
        cmap    = "cmo.haline"
        vlims   = [0,0.025]
        plotvar = ds_ptvar_all[3].mean('run')
    elif ii == 4: # CESM
        title   = "var(SSS), CESM"
        cmap    = "cmo.haline"
        vlims   = [0,0.025]
        plotvar = ds_ptvar_all[1].mean('ens')
    elif ii == 5: 
        title   = "Stochastic Model - CESM"
        cmap    = 'cmo.balance'
        plotvar = ds_ptvar_all[3].mean('run') - ds_ptvar_all[1].mean('ens')
        vlims   = [-.025,0.025]
    
    pcm = ax.pcolormesh(plotvar.lon,plotvar.lat,plotvar * maskin.squeeze(),
                        transform=proj,cmap=cmap,
                        vmin=vlims[0],vmax=vlims[1],zorder=-4)
    
    cb = viz.hcbar(pcm,ax=ax)
    ax.set_title(title,fontsize=fsz_title)
        
savename = "%sPointwise_Monvar_Diff_%s_%s_ALL.png" % (figpath,comparename,bbfn)
plt.savefig(savename,dpi=150,bbox_inches='tight')  

##%%
#%% Part (2): Examine how high-pass filtering over a region impacts the match

#%% Part 1, fit the wintertime decorrelation time
kmonth = 1
acf_in = np.array(tsm_byexp[-1]['acfs'][kmonth]).mean(0) # Run x Lag

expfit_out= proc.expfit(acf_in[::12],lags[::12],len(lags[::12]))
fig,ax= plt.subplots(1,1,constrained_layout=True,figsize=(12,4.5))
ax,_  = viz.init_acplot(kmonth,xtks,lags,ax=ax,title="")

ax.plot(lags,acf_in,label='raw')
efold   = int(np.abs(1/expfit_out['tau_inv']))
ax.plot(lags[::12],expfit_out['acf_fit'],label="$\tau =$ % .2f" % (1/expfit_out['tau_inv']))
ax.legend()
print("Fitted e-folding time (winter) is approximately %.2f months" % (efold))
#%%

hicutoff   = 10#E21#efold
hipass     = lambda x: proc.lp_butter(x,hicutoff,6,btype='highpass') # Make Function
regavg_ts2 = [ds.mean('lat').mean('lon') for ds in dsreg]

# Apply High Pass Filter
ds_hp      = [xr.apply_ufunc(hipass,ds,input_core_dims=[['time']],output_core_dims=[['time']],vectorize=True) for ds in regavg_ts2]
    
tsm_byexp_hp = []
for e in range(nexps):
    ts_in   = ds_hp[e].data
    nrun    = ts_in.shape[0]
    print(nrun)
    
    ts_list = [ts_in[ii,:] for ii in range(nrun)] 
    print(ts_in.shape)
    
    tsm     = scm.compute_sm_metrics(ts_list)
    
    tsm_byexp_hp.append(tsm)

print(tsm.keys())

#%% Compare High Pass Filtered Output:

skipexp = [2]
plot_hp_cesm = True
#np.array(tsm_byexp[0]['monvars']).shape

fig,axs=viz.init_monplot(1,2,figsize=(12,4.5))

for ii in range(2):
    ax = axs[ii]
    if ii == 0:
        tsm_in = tsm_byexp
        title  = "Original"
    else:
        tsm_in = tsm_byexp_hp
        title  = "High Pass Filtered"
    
    for ex in range(nexps):
        
        if ex in skipexp:
            continue
        
        if ex != 3 and plot_hp_cesm is False: # Plot non-filtered for stochastic model
            monvar = np.array(tsm_byexp[ex]['monvars']).mean(0)
        else:
            monvar = np.array(tsm_in[ex]['monvars']).mean(0)
        
        label = "%s, var=%.2e" % (expnames_long[ex],np.nanmean(ds_hp[ex].data.var(1),0))
        ax.plot(mons3,monvar,c=ecols[ex],label=label,ls=els[ex])
    ax.legend()
    ax.set_title(title)
        
#%% Compare persistence

fig,axs= plt.subplots(2,1,constrained_layout=True,figsize=(12,8))

for ii in range(2):
    ax      = axs[ii]
    
    if ii == 0:
        tsm_in = tsm_byexp
        title  = "Original"
    else:
        tsm_in = tsm_byexp_hp
        title  = "High Pass Filtered"
        
    
    ax,_    = viz.init_acplot(kmonth,xtks,lags,ax=ax,title="")
    
    for ex in range(nexps):
        
        if ex != 3 and plot_hp_cesm is False: # Plot non-filtered for stochastic model
            acfexp = np.array(tsm_byexp[ex]['acfs'][kmonth])
        else:
            acfexp = np.array(tsm_in[ex]['acfs'][kmonth])
            
            
        
         # Run x Lag
        ax.plot(lags,acfexp.mean(0),label=expnames_long[ex],
                c=ecols[ex],ls=els[ex])
        
        
#%% Try different hi-pass filters


regavg_ts2 = [ds.mean('lat').mean('lon') for ds in dsreg]


hithres    = [6,12,18,24,36,60]
monvar_bythres = []
for ii in range(len(hithres)):
    
    hicutoff   = hithres[ii]
    
    hipass     = lambda x: proc.lp_butter(x,hicutoff,6,btype='highpass') # Make Function

    # Apply High Pass Filter
    ds_hp      = [xr.apply_ufunc(hipass,ds,input_core_dims=[['time']],output_core_dims=[['time']],vectorize=True) for ds in regavg_ts2]
    
    # Compute Monthly Variance
    ds_monvar = []
    for ds in ds_hp:
        dsmonvar = ds.groupby('time.month').var('time')
        ds_monvar.append(dsmonvar)
    monvar_bythres.append(ds_monvar)
    
#%% Compare how the filterint impacts variance (By Model)

sel_ex  = [0,3]
fig,axs = viz.init_monplot(1,2,figsize=(12,4.5),)

for ex in range(2):
    
    expid   = sel_ex[ex]
    ax      = axs[ex]
    
    for ii in range(len(hithres)):
        plotvar = np.nanmean(monvar_bythres[ii][expid].data,0)
        ax.plot(mons3,plotvar,label="%i Months" % (hithres[ii]))
    ax.set_title(expnames_long[expid])
    ax.legend()
    ax.set_ylim([0,0.0025])
        
        

#%%in terms of persistence, etc

#%% ===========================================================================
#%% ===========================================================================

# #%% Old Version, focused on using ACFs
    
# # CESM2 vs. CESM1 Stochastic Model
# compare_name = "cesm2vcesm1_picsm"

# ncname = "cesm2_pic_0200to2000_TS_ACF_lag00to60_ALL_ensALL.nc"
# vname  = "TS"
# ncpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/proc/"
# ds1    = xr.open_dataset(ncpath+ncname).acf.load().squeeze()


# ncname = "SM_SST_cesm2_pic_noQek_SST_autocorrelation_thresALL_lag00to60.nc"
# vname  = "SST"
# ncpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/proc/"
# ds2    = xr.open_dataset(ncpath+ncname)[vname].load()#.acf.load().squeeze()

# ds_in    = [ds1,ds2]
# ds_in    = proc.resize_ds(ds_in)
# expnames = ["CESM2 PIC","Stochastic Model"]

# #%% Compare wintertime ACF over bounding box
# bbsel   = [-45,-30,50,65]
# selmons = [1,]

# ds_sel  = [proc.sel_region_xr(ds.isel(mons=selmons).mean('mons'),bbsel).mean('lat').mean('lon') for ds in ds_in]

# #%% Plot Mean ACF

# kmonth    = selmons[0]
# lags      = ds_sel[0].lags.data
# xtks      = lags[::3]


# fig       = plt.figure(figsize=(18,6.5))
# gs        = gridspec.GridSpec(4,4)


# # --------------------------------- # Locator
# ax1       = fig.add_subplot(gs[0:3,0],projection=ccrs.PlateCarree())
# ax1       = viz.add_coast_grid(ax1,bbox=bboxplot,fill_color="lightgray")
# ax1.set_title(bbsel)
# ax1 = viz.plot_box(bbsel)

# ax2       = fig.add_subplot(gs[1:3,1:])
# ax2,_     = viz.init_acplot(kmonth,xtks,lags,title="",)


# for ii in range(2):
#     ax2.plot(lags,ds_sel[ii].squeeze().data,label=expnames[ii])

# ax2.legend()

# #%% Section here is copied from regional spectral analysis


#%% Another Scrap Section to Check the trend

sm_sss      = ds_all[0].isel(run=0).sel(lon=-30,lat=46.5,method='nearest').data
sm_sss_dt   = sp.signal.detrend(sm_sss,0)#proc.xrdetrend(sm_sss)

ts_in       = [sm_sss,sm_sss_dt]
tsm_dt      = scm.compute_sm_metrics(ts_in)

#%% Plot the trend

def movmean(ts,N):
    return np.convolve(ts,np.ones(N)/N,mode='valid')

N = 1000
fig,ax = plt.subplots(1,1)
ax.plot(movmean(sm_sss,N),color="gray",label="raw")
ax.plot(movmean(sm_sss_dt,N),color='orange',label="detrend")
ax.legend()

#%% Plot the ACFs

kmonth = 1
lags   = np.arange(37)
xtks   = lags[::3]
acfs   = tsm_dt['acfs'][kmonth]
fig,ax = plt.subplots(1,1,constrained_layout=True,figsize=(12,4))
ax,_   = viz.init_acplot(kmonth,xtks,lags,ax=ax,)


ax.plot(lags,acfs[0],color="gray",label="raw")
ax.plot(lags,acfs[1],color='orange',label="detrend",ls='dashed')

#%% Plot the monhtly variance


mv     = tsm_dt['monvars']

fig,ax = viz.init_monplot(1,1,constrained_layout=True,figsize=(6,4))
#ax,_ = viz.init_acplot(kmonth,xtks,lags,ax=ax,)

ax.plot(mons3,mv[0],color="gray",label="raw")
ax.plot(mons3,mv[1],color='orange',label="detrend",ls='dashed')









