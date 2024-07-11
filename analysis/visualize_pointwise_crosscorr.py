#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Visualize
Created on Tue Mar 19 09:31:24 2024

@author: gliu
"""
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import sys
from tqdm import tqdm
import copy
import glob
import cartopy.crs as ccrs

import matplotlib as mpl
mpl.rcParams['font.family'] = 'JetBrains Mono'

#%% Import Custom Modules
amvpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/03_Scripts/" # amv module
scmpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/03_Scripts/stochmod/model/"

sys.path.append(amvpath)
sys.path.append(scmpath)

from amv import proc,viz
import scm
import amv.loaders as dl
import yo_box as ybx

#%% Figure Path

datpath     = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/proc/"
figpath     = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/02_Figures/20240614/"
# proc.makedir(figpath)


#%% Import data
#savename_out = "%s%s_%s_%s_ensALL.nc" % (outpath,outname_data,lagname,thresholds_name)


#ncname      = datpath + "CESM1_1920to2005_SSTvSSS_lag00to60_ALL_ensALL.nc"
ncname      = datpath + "SM_SST_SSS_lbdE_neg_crosscorrelation_nomasklag1_nroll0_lag00to60_ALL_ensALL.nc"
ds          = xr.open_dataset(ncname).load()


acfs        = ds.acf
acfsmean    = acfs.mean('ens')

#%% Load R1 and compute effective degrees of freedom

# Load Lag 1 from ACF calculations
r1names = ["SST","SSS"]
r1ncs   = ["CESM1_1920to2005_SSTACF_lag00to60_ALL_ensALL.nc","CESM1_1920to2005_SSSACF_lag00to60_ALL_ensALL.nc"]
autocorrs = []
for ii in range(2):
    ds_var = xr.open_dataset(datpath + r1ncs[ii])
    ds_var = ds_var.acf.isel(lags=1).load() # ('ens', 'lon', 'lat', 'mons', 'thres')
    autocorrs.append(ds_var)
    
    
# Compute effective DOF
dof    = 86 # Lag Correlations were calculated monthly, so 86 years
dofeff = proc.calc_dof(autocorrs[0],ts1=autocorrs[1],calc_r1=False,ntotal=dof) 

# Compute critical rho
p      = 0.05
tails  = 2 
rhocrit,ttest_str = proc.ttest_rho(p,tails,dofeff,return_str=True)

#%% Set names for land ice mask (this is manual, and works just on Astraeus :(...!)

# Copied from viz_metrics

lipath           = "/Users/gliu/Downloads/02_Research/01_Projects/04_Predict_AMV/03_Scripts/CESM_data/Masks/"
liname           = "CESM1LE_HTR_limask_pacificmask_enssum_lon-90to20_lat0to90.nc"

# Load Land Ice Mask
ds_mask          = xr.open_dataset(lipath+liname).MASK.squeeze().load()

# Edit
plotmask         = ds_mask.values.copy()
plotmask[np.isnan(plotmask)] = 0.

maskcoast        = ds_mask.values.copy()
maskcoast        = np.roll(maskcoast,1,axis=0) * np.roll(maskcoast,-1,axis=0) * np.roll(maskcoast,1,axis=1) * np.roll(maskcoast,-1,axis=1)



#%% 


def plot_ensmean(x,ds,dim,ax=None,c="k",
                 lw=1,):
    if ax is None:
        ax = plt.gca()
    
    mu    = ds.mean(dim)
    sigma = ds.std(dim)
    
    ax.plot(x,mu,c=c,lw=lw)
    ax.fill_between(x,mu-sigma,mu+sigma)
    

im   = 1
nens = 42


esel = 36


xtks = np.arange(0,63,3)
dspt = ds.sel(lon=-30,lat=50,method='nearest')


fig,ax = plt.subplots(1,1,constrained_layout=True,figsize=(12,4.5))


# Plot for each ensemble member
for e in range(nens):
    pv = dspt.acf.isel(mons=im,thres=0,ens=e)
    ax.plot(pv.lags,pv,alpha=0.2)

ax,_ = viz.init_acplot(im,xtks,pv.lags)

# Plot the Ensemble Mean
mu   = dspt.acf.isel(mons=im,thres=0).mean('ens')#ens=e)
ax.plot(pv.lags,mu,c="k",label="Ens. Mean")

# Plot the 
pv = dspt.acf.isel(mons=im,thres=0,ens=esel)
ax.plot(pv.lags,pv,c="red",label="Ens %02i" % (esel+1))
ax.legend()

ax.axhline([0],ls='solid',lw=0.55,c="k")

#%% Plotting Info -----------------------------

# Information
mpl.rcParams['font.family'] = 'JetBrains Mono'
bboxplot                    = [-80,0,20,65]
proj                        = ccrs.PlateCarree()
mons3                       = proc.get_monstr()
fsz_title                   = 16
fsz_axis                    = 14

# Other Plotting Parameters, specific to this script
lon                         = ds.lon
lat                         = ds.lat

#%% Visualize cross correlation from a selected base month


# Do selection
kmonth      = 1
cints       = np.arange(-.75,.80,0.05)
plotmesh    = False

# Make the plot
fig,axs,_ = viz.init_orthomap(3,4,bboxplot,figsize=(16,10))

for ll in range(12):
    
    ax = axs.flatten()[ll]
    ax = viz.add_coast_grid(ax,bbox=bboxplot,fill_color="lightgray")
    
    im = (kmonth+ll)%12
    ax.set_title("SSS Lag %i (%s)" % (ll,mons3[im]),fontsize=fsz_title)
    
    print(im)
    
    plotvar    = (acfsmean.isel(mons=kmonth,lags=ll).squeeze().T) * maskcoast
    
    rhocrit_in = np.nanmean(rhocrit[:,:,:,im,0],0).T * maskcoast
    
    viz.plot_mask(lon,lat,(plotvar > rhocrit_in).T,reverse=True,proj=proj,ax=ax,
                  color="lightgray",geoaxes=True,markersize=.75)
    
    
    if plotmesh:
        pcm = ax.pcolormesh(lon,lat,plotvar,vmin=cints[0],vmax=cints[-1],cmap='cmo.balance',transform=proj)
    else:
        pcm = ax.contourf(lon,lat,plotvar,levels=cints,
                          cmap='cmo.balance',transform=proj,extend='both')
        
        #cl = ax.contour(l)
cb = fig.colorbar(pcm,ax=axs.flatten(),fraction=0.025,pad=0.05,orientation='horizontal')
cb.set_label("Correlation with %s SST" % (mons3[kmonth]),fontsize=fsz_title)

savename = "%sSST_SSS_Pointwise_Crosscorr_mon%02i.png" % (figpath,kmonth+1)
plt.savefig(savename,dpi=150,bbox_inches='tight')
    
    
#%% Visualize Ens Mean DOF

def init_map(bboxplot):
    fig,ax,_ = viz.init_orthomap(1,1,bboxplot,figsize=(8,8))
    ax       = viz.add_coast_grid(ax,bbox=bboxplot,fill_color="lightgray")
    return fig,ax

cint   = np.arange(0,12,1)
kmonth = 1

fig,ax = init_map(bboxplot)
pv     = np.nanmean(dofeff[:,:,:,kmonth,0],(0)).T * maskcoast

#pcm = ax.pcolormesh(lon,lat,pv,cmap='cmo.balance',transform=proj)
pcm     = ax.contourf(lon,lat,pv,cmap='cmo.dense',transform=proj,levels=cint,extend='both')
cl      = ax.contour(lon,lat,pv,colors='k',transform=proj,levels=cint,linewidths=0.75)
ax.clabel(cl,levels=cint[::2])

cb = fig.colorbar(pcm,ax=ax,fraction=0.025,pad=0.05,orientation='horizontal')
cb.set_label("Effective Degrees of Freedom (%s)" % (mons3[kmonth]),fontsize=fsz_title)

savename = "%sSST_SSS_DOFEff_mon%02i.png" % (figpath,kmonth+1)
plt.savefig(savename,dpi=150,bbox_inches='tight')

#%%  Visualize critical rho

# Viz options ------------------
kmonth   = 1
cint     = np.arange(0,1.1,0.1)
# ------------------------------

fig,ax = init_map(bboxplot)

pv     = np.nanmean(rhocrit[:,:,:,kmonth,0],(0)).T * maskcoast

#pcm = ax.pcolormesh(lon,lat,pv,cmap='cmo.balance',transform=proj)
pcm    = ax.contourf(lon,lat,pv,cmap='cmo.dense',transform=proj,levels=cint,extend='both')
cl     = ax.contour(lon,lat,pv,colors='k',transform=proj,levels=cint,linewidths=0.75)
ax.clabel(cl,levels=cint[::2])

cb = fig.colorbar(pcm,ax=ax,fraction=0.025,pad=0.05,orientation='horizontal')
cb.set_label(r"Critical $\rho$ (%s)" % (mons3[kmonth]),fontsize=fsz_title)

savename = "%sSST_SSS_RhoCrit_mon%02i.png" % (figpath,kmonth+1)
plt.savefig(savename,dpi=150,bbox_inches='tight')

#%% Visualize R1 for SST and SSS

# Viz options ------------------
kmonth    = 1
cint      = np.arange(0.5,1.05,0.05)
# ------------------------------

fig,axs,_ = viz.init_orthomap(1,3,bboxplot,figsize=(16,8))

for aa in range(3):
    
    ax = axs.flatten()[aa]
    ax = viz.add_coast_grid(ax,bbox=bboxplot,fill_color="lightgray")
    
    if aa < 2:
        title = "%s Lag 1 Autocorr." % (r1names[aa])
        pv    = np.nanmean(autocorrs[aa][:,:,:,kmonth,0],(0))
    else:
        pv    = np.nanmean((np.abs(autocorrs[0]*autocorrs[1])[:,:,:,kmonth,0]),0)
        title = r"$\rho1_{SST}$ * $\rho1_{SSS}$"
    pv     = pv.T * maskcoast
    ax.set_title(title)
    
    
    pcm    = ax.contourf(lon,lat,pv,cmap='cmo.dense',transform=proj,levels=cint,extend='both')
    cl     = ax.contour(lon,lat,pv,colors='k',transform=proj,levels=cint,linewidths=0.75)
    ax.clabel(cl,levels=cint[::2])
    
cb = fig.colorbar(pcm,ax=ax,fraction=0.025,pad=0.05,orientation='vertical')
cb.set_label(r"%s Correlation" % (mons3[kmonth]),fontsize=fsz_title)

savename = "%sSST_SSS_r1_mon%02i.png" % (figpath,kmonth+1)
plt.savefig(savename,dpi=150,bbox_inches='tight')

#%% Load BSF

bsf      = dl.load_bsf()
bsf_savg = proc.calc_savg(bsf.BSF,ds=True)
bsf_savg,_ =proc.resize_ds([bsf_savg,acfsmean])
#%% Visualize SST-SSS (instantaneous) cross correlation seasonal

crosscorr_savg = proc.calc_savg(acfsmean.rename({'mons':'mon'}),ds=True)

#%% 

cints_bsf      = np.arange(-50,60,10)
cint           = np.arange(-1,1.1,0.1)
selcontour     = [-.5,0,0.5,]
fig,axs,_      = viz.init_orthomap(1,4,bboxplot,figsize=(16,8))

for ss in range(4):
    ax = axs[ss]
    ax = viz.add_coast_grid(ax,bbox=bboxplot,fill_color="lightgray")
    
    plotvar = crosscorr_savg.isel(lags=0,season=ss).squeeze().T * maskcoast
    pcm    = ax.contourf(lon,lat,plotvar,cmap='cmo.balance',transform=proj,levels=cint)
    cl = ax.contour(lon,lat,plotvar,colors="k",transform=proj,levels=selcontour,extend='both')
    ax.clabel(cl)
    ax.set_title(plotvar.season.values.item(),fontsize=fsz_title)
    
    # Plot BSF
    ax.contour(lon,lat,bsf_savg.isel(season=ss),colors="navy",linewidths=0.75,levels=cints_bsf,transform=proj,alpha=0.75)
    
cb = fig.colorbar(pcm,ax=axs.flatten(),fraction=0.01,pad=0.01)
cb.set_label(r"Cross Correlation",fontsize=fsz_title)

savename = "%sSST_SSS_lag0corr_ensAvg.png" % (figpath)
plt.savefig(savename,dpi=150,bbox_inches='tight')



#%% Load and visualize hi-pass filtered data

ncname = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/proc/CESM1/NATL_proc/filtered/CESM1_HTR_SST_SSS_NATL_crosscorr_hpf03mon.nc"
dshpf  = xr.open_dataset(ncname).isel(month=1).load()


#%%
cint = np.arange(-1,1.1,0.1)
cmap='cmo.balance'

plotvar = dshpf.corr.mean('ens')

fig,ax = init_map(bboxplot)
pv     = plotvar #* maskcoast

#pcm = ax.pcolormesh(lon,lat,pv,cmap='cmo.balance',transform=proj)
pcm     = ax.contourf(pv.lon,pv.lat,pv,cmap=cmap,transform=proj,levels=cint,extend='both')
cl      = ax.contour(pv.lon,pv.lat,pv,colors='k',transform=proj,levels=cint,linewidths=0.75)
ax.clabel(cl,levels=cint[::2])

cb = fig.colorbar(pcm,ax=ax,fraction=0.025,pad=0.05,orientation='horizontal')
cb.set_label("SST-SSS Correlation",fontsize=fsz_title)

ax.set_title("Instantaneous SST-SSS correlation (All Months) \n 3-month High Pass Filter",fontsize=fsz_title)
savename = "%sSST_SSS_CrossCorr_AllMon_3monHPF.png" % (figpath)
plt.savefig(savename,dpi=150,bbox_inches='tight')


