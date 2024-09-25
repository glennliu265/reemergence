#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
viz_pointwise_metrics.py

Add a catchall script to visualize pointwise metrics for the SSS model paper...

(Originally)
Visualize Pointwise Variance

Works with output from pointwise_autocorrelation_smoutput

Loads SST and SSS at once...

Created on Fri Aug 30 14:19:00 2024

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
sys.path.append(cwd + "/..")

# Paths and Load Modules
import reemergence_params as rparams
pathdict = rparams.machine_paths[machine]

sys.path.append(pathdict['amvpath'])
sys.path.append(pathdict['scmpath'])

# Set needed paths
figpath = pathdict['figpath']
input_path = pathdict['input_path']
output_path = pathdict['output_path']
procpath = pathdict['procpath']
rawpath = pathdict['raw_path']


# %% Import Custom Modules

from amv import proc, viz
import scm
import amv.xrfunc as xrf
import amv.loaders as dl

# Import stochastic model scripts

proc.makedir(figpath)

# %%

bboxplot = [-80, 0, 20, 65]
mpl.rcParams['font.family'] = 'Avenir'
mons3 = proc.get_monstr(nletters=3)

fsz_tick = 18
fsz_axis = 20
fsz_title = 16

rhocrit = proc.ttest_rho(0.05, 2, 86)

proj = ccrs.PlateCarree()


# %%  Indicate Experients (copying upper setion of viz_regional_spectra )


# Draft2
regionset       = "SSSCSU"
comparename     = "Paper_Draft02"
expnames        = ["SST_CESM","SSS_CESM","SST_Draft01_Rerun_QekCorr","SSS_Draft01_Rerun_QekCorr"]
expvars         = ["SST","SSS","SST","SSS"]
ecols           = ["firebrick","navy","forestgreen","magenta"]
els             = ["solid",'solid','dashed','dashed']
emarkers        = ["d","x","o","+"]

# Draft3
comparename     = "Draft3"
expnames        = ["SST_CESM","SSS_CESM","SST_Draft03_Rerun_QekCorr","SSS_Draft03_Rerun_QekCorr"]
expvars         = ["SST","SSS","SST","SSS"]
expnames_long   = ["SST (CESM1)","SSS (CESM1)","SST (Stochastic Model)","SSS (Stochastic Model)"]
expnames_short  = ["CESM_SST","CESM_SSS","SM_SST","SM_SSS"]
ecols           = ["firebrick","navy","hotpink","cornflowerblue"]
els             = ["solid",'solid','dashed','dashed']
emarkers        = ["d","x","o","+"]

# Get Point Info
pointset    = "PaperDraft02"
ptdict      = rparams.point_sets[pointset]
ptcoords    = ptdict['bboxes']
ptnames     = ptdict['regions']
ptnames_long = ptdict['regions_long']
ptcols      = ptdict['rcols']
ptsty       = ptdict['rsty']


# # # SST Comparison (Paper Draft, essentially Updated CSU) !!
# regionset       = "SSSCSU"
# comparename     = "SST_Paper_Draft02"
# expnames        = ["SST_Draft01_Rerun_QekCorr","SST_Draft01_Rerun_QekCorr_NoLbdd","SST_CESM"]
# expnames_long   = ["Stochastic Model","Stochastic Model (No $\lambda^d$)","CESM1"]
# expnames_short  = ["SM","SM_NoLbdd","CESM"]
# ecols           = ["forestgreen","goldenrod","k"]
# els             = ["solid",'dashed','solid']
# emarkers        = ["d","x","o"]

#%% IMport other plotting stuff

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

masknc          = "CESM1LE_HTR_limask_pacificmask_enssum_lon-90to20_lat0to90.nc"
dsmask          = xr.open_dataset(maskpath + masknc).MASK.load()

maskin          = dsmask

ds_gs2          = dl.load_gs(load_u2=True)

# Load Gulf Stream
ds_gs2          = dl.load_gs(load_u2=True)

# Load velocities
ds_uvel,ds_vvel = dl.load_current()
tlon            = ds_uvel.TLONG.mean('ens').data
tlat            = ds_uvel.TLAT.mean('ens').data

# Load Land Ice Mask
icemask         = xr.open_dataset(input_path + "masks/CESM1LE_HTR_limask_pacificmask_enssum_lon-90to20_lat0to90.nc")


mask            = icemask.MASK.squeeze()
mask_plot       = xr.where(np.isnan(mask),0,mask)#mask.copy()


mask_reg_sub    = proc.sel_region_xr(mask,bboxplot)
mask_reg_ori    = xr.ones_like(mask) * 0
mask_reg        = mask_reg_ori + mask_reg_sub


mask_apply      = icemask.MASK.squeeze().values

#%% Load the Pointwise Variance (Copied from compare_regional_metrics)

nexps       = len(expnames)

seavar_all  = []
var_all     = []
tsm_all     = []
rssts_all   = []
acfs_all    = []
amv_all     = []
for e in range(nexps):
    
    # Get Experiment information
    expname        = expnames[e]
    
    if "SSS" in expname:
        varname = "SSS"
    elif "SST" in expname:
        varname = "SST"
    metrics_path    = output_path + expname + "/Metrics/"
    
    # Load Pointwise variance
    ds_std = xr.open_dataset(metrics_path+"Pointwise_Variance.nc").load()
    var_all.append(ds_std)
    
    # Load Seasonal variance
    ds_std2 = xr.open_dataset(metrics_path+"Pointwise_Variance_Seasonal.nc").load()
    seavar_all.append(ds_std2)
    
    # # Load Regionally Averaged SSTs
    # ds = xr.open_dataset(metrics_path+"Regional_Averages_%s.nc" % regionset).load()
    # rssts_all.append(ds)
    
    # # Load Regional Metrics
    # ldz = np.load(metrics_path+"Regional_Averages_Metrics_%s.npz" % regionset,allow_pickle=True)
    # tsm_all.append(ldz)
    
    # # Load Pointwise_ACFs
    # ds_acf = xr.open_dataset(metrics_path + "Pointwise_Autocorrelation_thresALL_lag00to60.nc")[varname].load()
    # acfs_all.append(ds_acf)  
    
    # # Load AMV Information
    # ds_amv = xr.open_dataset(metrics_path + "AMV_Patterns_SMPaper.nc").load()
    
#%% Make a shared mask (Due to Ekman Forcing)

ekmask = [xr.where(~np.isnan(var_all[ex][expvars[ex]].mean('run')),1,np.nan) for ex in range(nexps)]
ekmask = xr.concat(ekmask,dim='exp')
ekmask = ekmask.prod('exp',skipna=False)

# Apply mask around corners
rollmask = xr.concat([ekmask.roll(dict(lat=-1)),ekmask.roll(dict(lat=1)),ekmask.roll(dict(lon=-1)),ekmask.roll(dict(lon=1))],dim='rolls')
rollmask = rollmask.prod('rolls',skipna=False)


def make_mask(ds_all,nanval=np.nan):
    if ~np.isnan(nanval):
        ds_all = [xr.where((ds == nanval),np.nan,ds) for ds in ds_all]
    mask    = [xr.where(~np.isnan(ds),1,np.nan) for ds in ds_all]
    mask    = xr.concat(mask,dim='exp')
    mask  = mask.prod('exp',skipna=False)
    return masknc
    


#%% Visualize (Log Ratios)

fsz_axis        = 24
fsz_title       = 28
fsz_tick        = 16

#imsk = icemask.MASK.squeeze()
vnames          = ["SST","SSS"]
plotcurrent     = False

cints           = np.log(np.array([.1,.5,2,10]))
cints           = np.sort(np.append(cints,0))
cints_lab       = [1/10,1/5,1/2,0,2,5,10]

# Visualize Interannual Variability
ii = 0
fig,axs,_       = viz.init_orthomap(1,2,bboxplot,figsize=(20,10))

for vv in range(2):
    
    ax      = axs[vv]
    ax      = viz.add_coast_grid(ax,bbox=bboxplot,fill_color="lightgray",fontsize=fsz_tick)
    
    if vv == 0:
        plotvar  = np.log(var_all[2].mean('run').SST / var_all[0].mean('run').SST) * rollmask
        plotname = "Log ( %s /  %s )"  % (expnames[2],expnames[1])
    elif vv == 1:
        plotvar  = np.log(var_all[3].mean('run').SSS / var_all[1].mean('run').SSS) * rollmask
        plotname = "Log ( %s / %s )"  % (expnames[3],expnames[1])
    
    print(plotname)
    
    pcm = ax.pcolormesh(plotvar.lon,plotvar.lat,plotvar,transform=proj,vmin=-1,vmax=1,cmap="cmo.balance",zorder=-1)
    cl  = ax.contour(plotvar.lon,plotvar.lat,plotvar,transform=proj,levels=cints,colors="dimgray",zorder=2,linewidths=1.5)
    ax.clabel(cl)
    
    cb = viz.hcbar(pcm,ax=ax,fraction=0.05,pad=0.01)
    cb.ax.tick_params(labelsize=fsz_tick)
    cb.set_label("Variance Log Ratio (Stochastic Model / CESM)",fontsize=fsz_axis)
    
    ax.set_title(vnames[vv],fontsize=fsz_title)
    
    # Plot Currents
    if plotcurrent:
        qint  = 2
        plotu = ds_uvel.UVEL.mean('ens').mean('month').values
        plotv = ds_vvel.VVEL.mean('ens').mean('month').values
        ax.quiver(tlon[::qint,::qint],tlat[::qint,::qint],plotu[::qint,::qint],plotv[::qint,::qint],
                  color=[.9,.9,.9],transform=proj,alpha=.67,zorder=1)#scale=1e3)
        
        
#%%



#%% Visualize (Regular Ratio)

fsz_axis        = 24
fsz_title       = 28
fsz_tick        = 16

#imsk = icemask.MASK.squeeze()
vnames          = ["SST","SSS"]
plotcurrent     = False

#cints           = np.log(np.array([.1,.5,2,10]))
cints           = np.arange(0,220,20)#np.array([0.01,0.25,0.50,1.0,1.5,2]) * 100#np.sort(np.append(cints,0))
#cints_lab       = cints*100#[0.01,0.25,0.50,1.0,1.5,2]

# Visualize Interannual Variability
ii = 0
fig,axs,_       = viz.init_orthomap(1,2,bboxplot,figsize=(20,10))

for vv in range(2):
    
    ax      = axs[vv]
    ax      = viz.add_coast_grid(ax,bbox=bboxplot,fill_color="lightgray",fontsize=fsz_tick)
    
    if vv == 0:
        plotvar  = (var_all[2].mean('run').SST / var_all[0].mean('run').SST) * rollmask
        plotname = "Ratio ( %s /  %s )"  % (expnames[2],expnames[1])
    elif vv == 1:
        plotvar  = (var_all[3].mean('run').SSS / var_all[1].mean('run').SSS) * rollmask
        plotname = "Ratio ( %s / %s )"  % (expnames[3],expnames[1])
    
    print(plotname)
    
    pcm = ax.pcolormesh(plotvar.lon,plotvar.lat,plotvar,transform=proj,vmin=0,vmax=2,cmap="cmo.balance",zorder=-1)
    cl  = ax.contour(plotvar.lon,plotvar.lat,plotvar*100,transform=proj,levels=cints,colors="dimgray",zorder=2,linewidths=1.5)
    ax.clabel(cl,fontsize=fsz_tick)
    
    cb = viz.hcbar(pcm,ax=ax,fraction=0.05,pad=0.01)
    cb.ax.tick_params(labelsize=fsz_tick)
    cb.set_label("%s Variance Ratio (Stochastic Model / CESM)" % vnames[vv],fontsize=fsz_axis)
    
    #ax.set_title(vnames[vv],fontsize=fsz_title)
    
    # Plot Currents
    if plotcurrent:
        qint  = 2
        plotu = ds_uvel.UVEL.mean('ens').mean('month').values
        plotv = ds_vvel.VVEL.mean('ens').mean('month').values
        ax.quiver(tlon[::qint,::qint],tlat[::qint,::qint],plotu[::qint,::qint],plotv[::qint,::qint],
                  color=[.9,.9,.9],transform=proj,alpha=.67,zorder=1)#scale=1e3)
        
        #ax.set_title(expnames_long[ex])
        
#         #ax.set_title("%s (%s)" % (vnames[vv],thresnames[th]))
        
#         if vv == 0:
#             ax.set_title(thresnames[th],fontsize=fsz_axis)
#         if th == 0:
#             viz.add_ylabel(vnames[vv],ax=ax,rotation='horizontal',fontsize=fsz_axis)
        
#         if vv == 0: # Log Ratio (SSTs)
#             plotvar = np.log(specsum_exp[1].isel(thres=thres).mean('ens')/specsum_exp[0].isel(thres=thres).mean('ens'))
#         else:
#             plotvar = np.log(specsum_exp[3].isel(thres=thres).mean('ens')/specsum_exp[2].isel(thres=thres).mean('ens'))
        
#         pcm = ax.pcolormesh(plotvar.lon,plotvar.lat,plotvar*imsk,transform=proj,vmin=-2.5,vmax=2.5,cmap="cmo.balance",zorder=-1)
#         cl  = ax.contour(plotvar.lon,plotvar.lat,plotvar*imsk,transform=proj,levels=cints,colors="dimgray",zorder=2,linewidths=1.5)
#         ax.clabel(cl,fmt="%.2f",fontsize=fsz_tick)
        
    # Plot Gulf Stream Position
    ax.plot(ds_gs2.lon.mean('mon'),ds_gs2.lat.mean('mon'),transform=proj,lw=1.75,c='k',ls='dashdot')
    
    # Plot Ice Edge
    ax.contour(icemask.lon,icemask.lat,mask_plot,colors="cyan",linewidths=2.5,
                transform=proj,levels=[0,1],zorder=-1)
    
    viz.label_sp(ii,alpha=0.75,ax=ax,fontsize=fsz_title,x=0.05)
    ii+=1
        

    
#         # Plot Regions
#         for ir in range(nregs):
#             rr   = regplot[ir]
#             rbbx = bboxes[rr]
            
#             ls_in = rsty[rr]
#             if ir == 2:
#                 ls_in = 'dashed'
            
#             viz.plot_box(rbbx,ax=ax,color=rcols[rr],linestyle=ls_in,leglab=regions_long[rr],linewidth=1.5,return_line=True)

        
#         #cb  = viz.hcbar(pcm,ax=ax)
#         viz.label_sp(ii,alpha=0.75,ax=ax,fontsize=fsz_title,y=1.08,x=-.02)
#         ii+=1
        
# cb = viz.hcbar(pcm,ax=axs.flatten(),fraction=0.025)
# cb.set_label("Log(Stochastic Model / CESM1)",fontsize=fsz_axis)
# cb.ax.tick_params(labelsize=fsz_tick)
# #plt.suptitle("Log Ratio (SM/CESM)")
    
# #figname = "%sVariance_Specsum_LogRatio.png" % (figpath)
# figname = "%sLogratio_Spectra.png" % (figpath)

# if plotcurrent:
#     figname = proc.addstrtoext(figname,"_withcurrent",)
# plt.savefig(figname,dpi=150,bbox_inches='tight')
    
savename = "%sSST_SSS_Variance_Ratio_%s.png" % (figpath,comparename,)
plt.savefig(savename,dpi=150,bbox_inches='tight')

#%% Section 2: Pointwise Cross Correlation
"""

Works with output from 
    - calc_significance_pointwise
    - pointwise_crosscorrelation
    
"""

# CESM names
cesm_path       = procpath
cesm_cc_name    = "CESM1_1920to2005_SST_SSS_crosscorrelation_nomasklag1_nroll0_lag00to60_ALL_ensALL.nc"
cesm_sig_name   = "CESM1_1920to2005_SST_SSS_crosscorrelation_nomasklag1_nroll0_Significance_mciter1000_usemon1_tails2.nc"

# SM names
sm_path         = procpath
# sm_cc_name      = "SM_SST_SSS_PaperDraft02_lag00to60_ALL_ensALL.nc"
# sm_sig_name     = "SM_SST_SSS_PaperDraft02_Significance_mciter1000_usemon1_tails2.nc"
sm_cc_name      = "SM_SST_SSS_PaperDraft03_lag00to60_ALL_ensALL.nc"
sm_sig_name     = "SM_SST_SSS_PaperDraft03_Significance_mciter1000_usemon1_tails2.nc"


# CESM HPF
cesm_cc_name_hpf    = "CESM1_1920to2005_SST_SSS_crosscorrelation_nomasklag1_nroll0_hpf012mons_lag00to60_ALL_ensALL.nc"
cesm_sig_name_hpf   = ""

# SM HPF
sm_cc_name_hpf      = "SM_SST_SSS_PaperDraft02_hpf012mons_lag00to60_ALL_ensALL.nc"
sm_sig_name_hpf     = ""

#%% Load the files
ilag            =  0

# Load CESM1
st              = time.time()
cesm_cc         = xr.open_dataset(cesm_path + cesm_cc_name).acf.isel(thres=0,lags=ilag).load()
cesm_sig        = xr.open_dataset(cesm_path + cesm_sig_name).thresholds.load() # (tails: 2, thres: 3, lat: 48, lon: 65)
print("Loaded CESM1 in %.2fs" % (time.time()-st))

# Load Stochastic Model
st              = time.time()
sm_cc           = xr.open_dataset(sm_path + sm_cc_name).acf.isel(thres=0,lags=ilag).load()
sm_sig          = xr.open_dataset(sm_path + sm_sig_name).thresholds.load()
print("Loaded Stochastic Model in %.2fs" % (time.time()-st))


# Load CESM1 (hpf)
st              = time.time()
cesm_cc_hpf     = xr.open_dataset(cesm_path + cesm_cc_name_hpf).acf.isel(thres=0,lags=ilag).load()
print("Loaded CESM1 in %.2fs" % (time.time()-st))

# Load SM (hpf)
st              = time.time()
sm_cc_hpf       = xr.open_dataset(sm_path + sm_cc_name_hpf).acf.isel(thres=0,lags=ilag).load()
print("Loaded Stochastic Model in %.2fs" % (time.time()-st))

#%% Compare the two cross-correlations

siglvl          = 0.05
plotnames       = ["CESM1-LE","Stochastic Model"]
plot_ccs        = [cesm_cc.mean('mons').mean('ens'),sm_cc.mean('mons').mean('ens')]
plot_sig        = [cesm_sig.sel(thres=siglvl),sm_sig.sel(thres=siglvl),]


compare_ccs     = [cesm_cc,sm_cc,cesm_cc_hpf,sm_cc_hpf]
compare_ccs     = [ds.mean('mons').mean('ens') for ds in compare_ccs]
comparenames    = ["CESM1-LE","Stochastic Model","CESM1-LE (High Pass Filter)","Stochastic Model (High Pass Filter)"]

#%% Plot/Compare Pointwise Cross Correlation

plot_point      = True
pmesh           = False
cints           = np.arange(-1,1.1,.1)
fsz_axis        = 24
fsz_title       = 28
fsz_tick        = 22
lw_plot         = 4.5

fig,axs,_       = viz.init_orthomap(1,2,bboxplot,figsize=(20,10))
ii = 0
for vv in range(2):
    
    ax      = axs[vv]
    ax      = viz.add_coast_grid(ax,bbox=bboxplot,fill_color="lightgray",fontsize=fsz_tick)
    
    
    plotvar = plot_ccs[vv] * mask_reg
    
    if pmesh:
        pcm     = ax.pcolormesh(plotvar.lon,plotvar.lat,plotvar.T,cmap='cmo.balance',
                            transform=proj,vmin=-1,vmax=1)
    else:
        pcm     = ax.contourf(plotvar.lon,plotvar.lat,plotvar.T,cmap='cmo.balance',
                            transform=proj,levels=cints)
        cl      = ax.contour(plotvar.lon,plotvar.lat,plotvar.T,colors="lightgray",
                            transform=proj,levels=cints,linewidths=0.65)
        ax.clabel(cl,fontsize=fsz_tick)
    
    # Plot the significance
    upper = plot_sig[vv].isel(tails=1)
    lower = plot_sig[vv].isel(tails=0)
    
    masksig = (plotvar > upper) | (plotvar < lower)
    viz.plot_mask(masksig.lon,masksig.lat,masksig,reverse=True,
                  proj=proj,ax=ax,geoaxes=True,
                  color='dimgray',markersize=1.5)
    
    ax.set_title(plotnames[vv],fontsize=fsz_title)
    
    # Add Other Features
    # Plot Gulf Stream Position
    ax.plot(ds_gs2.lon.mean('mon'),ds_gs2.lat.mean('mon'),transform=proj,lw=lw_plot,c='cornflowerblue',ls='dashdot')

    # Plot Ice Edge
    ax.contour(icemask.lon,icemask.lat,mask_plot,colors="cyan",linewidths=lw_plot,
               transform=proj,levels=[0,1],zorder=-1)
    
    # Plot Points
    if plot_point:
        nregs = len(ptnames)
        for ir in range(nregs):
            pxy   = ptcoords[ir]
            ax.plot(pxy[0],pxy[1],transform=proj,markersize=45,markeredgewidth=.5,c=ptcols[ir],
                    marker='*',markeredgecolor='k')
        
    
    viz.label_sp(ii,alpha=0.75,ax=ax,fontsize=fsz_title,y=1.08,x=-.02)
    ii +=1
    
# Add Colorbar
cb = viz.hcbar(pcm,ax=axs.flatten(),fraction=0.055,pad=0.01)
cb.ax.tick_params(labelsize=fsz_tick)
cb.set_label("SST-SSS Cross Correlation",fontsize=fsz_axis)

# Add Other Plots
savename = "%sSST_SSS_CESM_vs_SM_CrossCorr_Avg_AllMonths.png" % (figpath)
plt.savefig(savename,dpi=150,bbox_inches='tight')

#%% Make Comparison Plot

cylabs          = ['Raw',"High-Pass Filter"]
pmesh           = True
fig,axs,_       = viz.init_orthomap(2,2,bboxplot,figsize=(20,14.5))

# Set up Axes
ii = 0
for cc in range(2):
    for ex in range(2):
        
        ax      = axs[cc,ex]
        ax      = viz.add_coast_grid(ax,bbox=bboxplot,fill_color="lightgray",fontsize=fsz_tick)
        
        if ex == 0:
            viz.add_ylabel(cylabs[cc],fontsize=fsz_title,ax=ax,x=-0.05)
        if cc == 0:
            ax.set_title(plotnames[ex],fontsize=fsz_title)
        
        
        plotvar = compare_ccs[ii] * mask_reg
        
        if pmesh:
            pcm     = ax.pcolormesh(plotvar.lon,plotvar.lat,plotvar.T,cmap='cmo.balance',
                                transform=proj,vmin=-1,vmax=1)
        else:
            pcm     = ax.contourf(plotvar.lon,plotvar.lat,plotvar.T,cmap='cmo.balance',
                                transform=proj,levels=cints)
            cl      = ax.contour(plotvar.lon,plotvar.lat,plotvar.T,colors="lightgray",
                                transform=proj,levels=cints,linewidths=0.65)
            ax.clabel(cl,fontsize=fsz_tick)
        
        
        ii += 1
        
        
#%% Compare the seasonal fluctuation in cross-correlation

plot_cc_savg    = [cesm_cc,sm_cc]
plot_cc_savg    = [proc.calc_savg(ds.rename(dict(mons='mon')),ds=True) for ds in plot_cc_savg]

fig,axs,_       = viz.init_orthomap(2,4,bboxplot,figsize=(28,10))


for ex in range(2):
    
    if ex == 0:
        c = 'k'
        lab = "CESM"
    else:
        c = 'salmon'
        lab = "Stochastic Model"
        
    
    for sid in range(4):
        
        
        ax = axs[ex,sid]
        ax = viz.add_coast_grid(ax,bbox=bboxplot,fill_color="lightgray",fontsize=fsz_tick)
        
        plotvar = plot_cc_savg[ex].isel(season=sid).mean('ens')
        
        
        if ex == 0:
            ax.set_title(plotvar.season.item(),fontsize=fsz_axis)
        if sid == 0:
            viz.add_ylabel(lab,fontsize=fsz_axis,ax=ax)
            
            
        
        if pmesh:
            pcm     = ax.pcolormesh(plotvar.lon,plotvar.lat,plotvar.T,cmap='cmo.balance',
                                transform=proj,vmin=-1,vmax=1)
        else:
            pcm     = ax.contourf(plotvar.lon,plotvar.lat,plotvar.T,cmap='cmo.balance',
                                transform=proj,levels=cints)
            cl      = ax.contour(plotvar.lon,plotvar.lat,plotvar.T,colors="lightgray",
                                transform=proj,levels=cints,linewidths=0.65)
            ax.clabel(cl,fontsize=fsz_tick)
    
    
savename = "%sSST_SSS_Cross_Correlation_CESM_v_SM_AllSeasons.png" % (figpath)
plt.savefig(savename,dpi=150)
#%% Plot seasonal evolution of momthyl cross correlation


plot_cc_all = [cesm_cc,sm_cc]
fig,axs      = viz.init_monplot(1,3,figsize=(16,4.5))


for ipt in range(3):
    ax = axs[ipt]
    lonf,latf = ptcoords[ipt]
    locfn,loctitle = proc.make_locstring(lonf,latf)
    ax.set_title(loctitle)
    ax.set_ylim([-.75,.75])
    ax.axhline([0],lw=.75,c="k",ls='dashed')
    
    if ipt == 0:
        ax.set_ylabel("SST-SSS Cross Correlation")
    
    
    for ex in range(2):
        
        if ex == 0:
            c = 'k'
            lab = "CESM"
        else:
            c = 'salmon'
            lab = "Stochastic Model"
        
        plotvar = proc.selpt_ds(plot_cc_all[ex],lonf,latf)
        nens,_ = plotvar.shape
        
        mu    = plotvar.mean('ens').data
        sigma = proc.calc_stderr(plotvar,0) #plotvar.std('ens').data
        
        ax.plot(mons3,mu,color=c,label=lab,lw=2)
        ax.fill_between(mons3,mu-sigma,mu+sigma,color=c,alpha=0.15,)
        
savename = "%sSST_SSS_Cross_Correlation_CESM_v_SM_Paper_Points.png" % (figpath)
plt.savefig(savename,dpi=150)    

#%% Visualize pointwise crosscorrelation at 3 locations

fig,axs = plt.subplots(1,3,constrained_layout=True)


#%%




