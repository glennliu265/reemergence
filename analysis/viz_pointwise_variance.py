#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Visualize Pointwise Variance

Works with output from pointwise_autocorrelation_smoutput

Loads SST and SSS at once...

Created on Fri Aug 30 14:19:00 2024

@author: gliu

"""



from amv import proc, viz
import scm
import amv.xrfunc as xrf
import amv.loaders as dl
import reemergence_params as rparams
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

# Import AMV Calculation

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
regionset       = "SSSCSU"
comparename     = "Paper_Draft02"
expnames        = ["SST_CESM","SSS_CESM","SST_Draft01_Rerun_QekCorr","SSS_Draft01_Rerun_QekCorr"]
expvars         = ["SST","SSS","SST","SSS"]
ecols           = ["firebrick","navy","forestgreen","magenta"]
els             = ["solid",'solid','dashed','dashed']
emarkers        = ["d","x","o","+"]


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

masknc = "CESM1LE_HTR_limask_pacificmask_enssum_lon-90to20_lat0to90.nc"
dsmask = xr.open_dataset(maskpath + masknc).MASK.load()

maskin = dsmask


ds_gs2          = dl.load_gs(load_u2=True)

# Load Gulf Stream
ds_gs2  = dl.load_gs(load_u2=True)

# Load velocities
ds_uvel,ds_vvel = dl.load_current()
tlon  = ds_uvel.TLONG.mean('ens').data
tlat  = ds_uvel.TLAT.mean('ens').data

# Load Land Ice Mask
icemask     = xr.open_dataset(input_path + "masks/CESM1LE_HTR_limask_pacificmask_enssum_lon-90to20_lat0to90.nc")


mask        = icemask.MASK.squeeze()
mask_plot   = xr.where(np.isnan(mask),0,mask)#mask.copy()


mask_reg_sub    = proc.sel_region_xr(mask,bboxplot)
mask_reg_ori    = xr.ones_like(mask) * 0
mask_reg        = mask_reg_ori + mask_reg_sub


mask_apply  = icemask.MASK.squeeze().values

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


#%% Visualize (Log Ratios)

fsz_axis        = 24
fsz_title       = 28
fsz_tick        = 16

#imsk = icemask.MASK.squeeze()
vnames          = ["SST","SSS"]
plotcurrent     = True

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
    cb.set_label("%s Variance Ratio (Stochastic Model / CESM)" % varname,fontsize=fsz_axis)
    
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

