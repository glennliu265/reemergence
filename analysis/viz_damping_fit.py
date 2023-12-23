#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualize Damping Fit

Compare fitted damping timescales computed in [estimate_damping_fit.py]
between SST and SSS

Copied upper section of aforementioned script on 2023.11.22

Created on Wed Nov 22 16:00:20 2023

@author: gliu
"""

import xarray as xr
import numpy as np
import scipy as sp
import sys
import matplotlib.pyplot as plt
from tqdm import tqdm
import cartopy.crs as ccrs

#%% User Edits

figpath     = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/02_Figures/20231218/"
datpath_ac  = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/proc/"
lonf        = -30
latf        = 50

varnames    = ("SST","SSS")
recalculate = False

# Define which lagmaxes to fit exponential function over
lagmaxes    = [1,2,3]#[7,13,37]

# Information for loading covariance-estimated HFF
datpath_damping = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/01_Data/model_input/"


#%% Import Custom Modules

sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/03_Scripts/")
sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/03_Scripts/stochmod/model/")
from amv import proc,viz
import scm

proc.makedir(figpath)

# %% Load Damping Data

savename = "%sCESM1_LENS_SST_SSS_lbd_exponential_fit_lagmax1to3.nc" % datpath_ac
exists   = proc.checkfile(savename)
da       = xr.open_dataset(savename)
lbd_fit  = da.lbd.values

lon = da.lon.values
lat = da.lat.values

# ------------------------------------------------------
#%% Examine the map of estimated damping for each month
# ------------------------------------------------------
v   = 1
lm  = 2
e   = 0
im  = 0

bboxplot = [-80,0,0,65]
clvls  = [3,6,12,24]
plot_timescale=True

for im in tqdm(range(12)):
    fig,ax = viz.geosubplots(1,1,figsize=(8,6))
    
    ax = viz.add_coast_grid(ax,bboxplot,fill_color="k")
    plotvar = 1/lbd_fit[v,lm,e,im,:,:]*-1
    pcm     = ax.pcolormesh(lon,lat,plotvar,vmin=0,vmax=24,cmap="inferno_r")
    cl      = ax.contour(lon,lat,plotvar,levels=clvls,colors="w")
    ax.clabel(cl,fontsize=12)
    cb = fig.colorbar(pcm,ax=ax,fraction=0.02,pad=0.01)
    cb = cb.set_label("Damping Timescale $\lambda_a^{-1}$ (months)")
    
    title = "%s Damping Timescale (Fit over first %i lags)\n Mon %i | Ens %02i" % (varnames[v],lagmaxes[lm]-1,im+1,e+1,)
    ax.set_title(title,fontsize=16)
    savename = "%sExpFit_Damping_Map_%s_lagmax%02i_month%02i_ens%02i.png" % (figpath,varnames[v],lagmaxes[lm]-1,im+1,e+1,)
    plt.savefig(savename,dpi=150,bbox_inches="tight",)

# ------------------------------
#%% Make seasonally averaged maps

lm = 1
e  = "mean"

savgs,snames = proc.calc_savg(lbd_fit,axis=3,return_str=True) # [var x lagmax x ens x lat x lon]

# Take ensemble mean
savgs_mean = [np.mean(s,(2))[:,-1,...] for s in savgs]

#%% SEASONALLY AVERAGED PLOTS FOR SSS AND SST

bboxplot       = [-80,0,10,65]
fig,axs        = viz.geosubplots(2,4,figsize=(16,5.5),constrained_layout=True)
cmap           = 'cmo.deep'

plot_timescale = False

if plot_timescale:
    clvls  = [3,6,12,18,24]
    vlms   = [0,24]
    clbl    = "Damping Timescale $\lambda_a^{-1}$ (months)"
else:
    vlms   = [0,0.75]
    clbl    = "Damping (1/month)"

for v in range(2):
    for s in range(4):
        ax = axs[v,s]
        
        # Labeling + Setup ------
        blabel=[0,0,0,0]
        if s == 0:
            blabel[0] = 1
        if v == 1:
            blabel[-1] = 1
        ax = viz.add_coast_grid(ax,bboxplot,fill_color="k",blabels=blabel,fontsize=14)
        if v == 0:
            ax.set_title(snames[s],fontsize=16)
        if s == 0:
            ax.text(-0.22, 0.55, varnames[v], va='bottom', ha='center',rotation='vertical',
                    rotation_mode='anchor',transform=ax.transAxes,fontsize=16)
            
        # Plotting
        
        
        if plot_timescale:
            plotvar = 1/savgs_mean[s][v,...]*-1
            pcm     = ax.pcolormesh(lon,lat,plotvar,vmin=vlms[0],vmax=vlms[1],cmap=cmap)
            cl      = ax.contour(lon,lat,plotvar,levels=clvls,colors="w",linewidths=0.75)
            ax.clabel(cl,fontsize=12)
            
        else:
            plotvar = savgs_mean[s][v,...]*-1
            pcm     = ax.pcolormesh(lon,lat,plotvar,cmap=cmap,vmin=vlms[0],vmax=vlms[1])
            #fig.colorbar(pcm,ax=ax)
            
        
        #fig.colorbar(pcm,ax=ax,orientation='horizontal')
cb = fig.colorbar(pcm,ax=axs.flatten(),pad=0.01,fraction=0.020)
cb = cb.set_label(clbl,fontsize=14)

savename = "%sExpFit_Damping_Map_lagmax%02i_Seasonal_Ensemble_Avg_plottimescale%i.png" % (figpath,lagmaxes[lm]-1,plot_timescale)
plt.savefig(savename,dpi=150,bbox_inches="tight",)

#%% Plot Actual Damping values


#%% Load Explicitly Estimated Atmospheric Damping for comparison --------------------------

# Load damping
cv_damping = np.load(datpath_damping+"FULL_HTR_NHFLX_Damping_monwin3_sig020_dof082_mode4.npy") # [nlon, nlat, 12] (ensemble average, a bit old...)
mask       = np.load(datpath_damping+"FULL_PIC_NHFLX_Damping_monwin3_sig005_dof1893_mode5_mask_lag1_ensorem0.npy") # [nlon, nlat, 12]
mld        = np.load(datpath_damping+"FULL_HTR_HMXL_hclim.npy") # [Lon x Lat x 12]

# Load Latlon
lon180,lat180    = scm.load_latlon()

# Subset to Region
preprocvars = [cv_damping,mask,mld]
bbox_est    = [lon[0],lon[-1],lat[0],lat[-1]] # Bounding Box of estimate above
cropvars = []
for v in preprocvars:
    
    varr,lonr,latr=proc.sel_region(v,lon180,lat180,bbox_est)
    cropvars.append(varr.transpose(1,0,2)) # [Lat x Lon x 12]


# Convert damping to timescale
dt  = 3600*24*30
cp  = 3996
rho = 1026
cv_damping,mask,mld=cropvars
lbd_est = (cv_damping * mask * dt) / (rho * cp * mld )#mld.mean(2)[...,None])
lbd_est = lbd_est.transpose(2,0,1) # [mon x lat x lon]
#%% Take Ensemble mean of lbd_fit


lbd_fit_ensmean = lbd_fit.mean(2)  # [var x lagmax x mon x lat x lon]

# Lets first look at SST damping timescales and their differences
sst_lbd_diff = lbd_fit_ensmean[0,:,:,:,:] - lbd_est[None,:,:,:]

#%% Make an Ice Mask
ice_mask = lbd_est.sum(0)
ice_mask[~np.isnan(ice_mask)] = 1 


#%% Test Plot

plt.pcolormesh(-1/sst_lbd_diff[0,0,:,:],vmin=0,vmax=10),plt.colorbar()

#%% Look at the seasonalaverages

savgs_fit,snames = proc.calc_savg(lbd_fit_ensmean,axis=2,return_str=True) #[seas][var x lagmax x lat x lon]
savgs_est,_      = proc.calc_savg(lbd_est,axis=0,return_str=True) # [seas][lat x lon]

#%% Plot seasonal average of $\lambda_a$ in the style of SST/SSS damping above

bboxplot       = [-80,0,10,65]
cmap           = 'cmo.deep'
plot_timescale = False

if plot_timescale:
    clvls  = [3,6,12,18,24,36]
    vlms   = [0,24]
    clbl    = "Damping Timescale $\lambda_a^{-1}$ \n (months)"
else:
    vlms   = [0,0.75]
    clbl    = "Damping (1/month)"

fig,axs = viz.geosubplots(1,4,figsize=(16,8))

for s in range(4):
    ax = axs[s]
    blabel = [0,0,0,1]
    if s == 0:
        blabel[0] = 1
    ax = viz.add_coast_grid(ax,bboxplot,fill_color="k",blabels=blabel,fontsize=14)
    
    if plot_timescale:
        plotvar = 1/savgs_est[s][...]
    else:
        plotvar = savgs_est[s][...]
    pcm     = ax.pcolormesh(lon,lat,plotvar,vmin=vlms[0],vmax=vlms[1],cmap=cmap)
    cl      = ax.contour(lon,lat,plotvar,levels=clvls,colors="w",linewidths=0.75)
    ax.clabel(cl,fontsize=12)
    
    if s == 0:
        ax.text(-0.15, 0.55,"$\lambda^a$", va='bottom', ha='center',rotation='vertical',
                rotation_mode='anchor',transform=ax.transAxes,fontsize=16)
    
cb = fig.colorbar(pcm,ax=axs.flatten(),pad=0.01,fraction=0.01)
cb = cb.set_label(clbl,fontsize=14)

savename = "%sLbda_Damping_Map_lagmax%02i_Seasonal_Ensemble_Avg_plottimescale%i.png" % (figpath,lagmaxes[lm]-1,plot_timescale)
plt.savefig(savename,dpi=150,bbox_inches="tight",)

# --------------------------
#%% Visualize difference of lbd_SST - lbd_a and lbd_SSS
# --------------------------

fig,axs        = viz.geosubplots(2,4,figsize=(16,5.5),constrained_layout=True)
cmap           = 'cmo.deep'
plot_timescale = False

if plot_timescale:
    clvls  = [3,6,12,18,24]
    vlms   = [0,24]
    clbl    = "Damping Timescale $\lambda_a^{-1}$ (months)"
else:
    vlms   = [0,0.75]
    clbl    = "Damping (1/month)"

for v in range(2):
    for s in range(4):
        ax = axs[v,s]
        
        # Labeling + Setup ------
        blabel=[0,0,0,0]
        if s == 0:
            blabel[0] = 1
        if v == 1:
            blabel[-1] = 1
        ax = viz.add_coast_grid(ax,bboxplot,fill_color="k",blabels=blabel,fontsize=14)
        if v == 0:
            ax.set_title(snames[s],fontsize=16)

            
        # Plotting
        if v == 0: # Plot lbd_SSS
            plotvar = savgs[s][1,lm,:,:,:].mean(0)*-1 * ice_mask# Take Ens Avg
            name = "$\lambda^{SSS}$"
        else: # Plot lbd_SST - lbda
            plotvar = savgs[s][0,lm,:,:,:].mean(0)*-1 - savgs_est[s]
            name = "$\lambda^{SST}-\lambda^a$"
        if s == 0:
            ax.text(-0.22, 0.55, name, va='bottom', ha='center',rotation='vertical',
                    rotation_mode='anchor',transform=ax.transAxes,fontsize=16)
            
        
        
        if plot_timescale:
            plotvar = 1/plotvar
            pcm     = ax.pcolormesh(lon,lat,plotvar,vmin=vlms[0],vmax=vlms[1],cmap=cmap)
            cl      = ax.contour(lon,lat,plotvar,levels=clvls,colors="w",linewidths=0.75)
            ax.clabel(cl,fontsize=12)
            
        else:
            plotvar = plotvar
            pcm     = ax.pcolormesh(lon,lat,plotvar,cmap=cmap,vmin=vlms[0],vmax=vlms[1])
            #fig.colorbar(pcm,ax=ax)
            
        
        #fig.colorbar(pcm,ax=ax,orientation='horizontal')
cb = fig.colorbar(pcm,ax=axs.flatten(),pad=0.01,fraction=0.020)
cb = cb.set_label(clbl,fontsize=14)

savename = "%sLbdO_Estimate_Damping_Map_lagmax%02i_Seasonal_Ensemble_Avg_plottimescale%i.png" % (figpath,lagmaxes[lm]-1,plot_timescale)
plt.savefig(savename,dpi=150,bbox_inches="tight",)


#%% Make the plot

import cmocean as cmo

ivar    = 0
ilagmax = 1
clvls   = [-24,-12,-6,-3,0,3,6,12,24]

fig,axs = viz.geosubplots(3,4,figsize=(16,8))
bboxplot = [-80,0,10,62]

for v in range(3):
    
    for s in range(4):
        ax = axs[v,s]
        
        # Select Plotting Variable
        if v == 0: # Plot Fit
            savg_in = -savgs_fit[s][ivar,ilagmax,:,:]
            vlms = [0,36]
            lab  = "Exp. Fit (%i-mon)" % (lagmaxes[ilagmax]-1)
            cmap = "cmo.deep"
        elif v == 1: # Plot estimate
            savg_in = savgs_est[s][:,:] 
            vlms = [0,36]
            lab  = "Cov. Est."
            cmap = "cmo.deep"
        else: # Fit minus estimate
            savg_in =  (1/savgs_est[s][:,:]) - (-1/savgs_fit[s][ivar,ilagmax,:,:])
            vlms =[-18,18]
            lab  = "Est. - Fit"
            cmap = "cmo.balance"
        
        # Labeling + Setup ------
        blabel=[0,0,0,0]
        if s == 0:
            blabel[0] = 1
        if v == 2:
            blabel[-1] = 1
        ax = viz.add_coast_grid(ax,bboxplot,fill_color="k",blabels=blabel,fontsize=14)
        if v == 0:
            ax.set_title(snames[s],fontsize=16)
        if s == 0:
            ax.text(-0.22, 0.55, lab, va='bottom', ha='center',rotation='vertical',
                    rotation_mode='anchor',transform=ax.transAxes,fontsize=16)
            
        # Plotting
        if v < 2:
            plotvar = 1/savg_in
        else:
            plotvar = savg_in
        if v < 2:
            pcm     = ax.pcolormesh(lon,lat,plotvar,vmin=vlms[0],vmax=vlms[1],cmap=cmap)
        else:
            pcmdiff = ax.pcolormesh(lon,lat,plotvar,vmin=vlms[0],vmax=vlms[1],cmap=cmap)
        cl      = ax.contour(lon,lat,plotvar,levels=clvls,colors="k",linewidths=0.75)
        ax.clabel(cl,fontsize=14)
        
        #fig.colorbar(pcm,ax=ax,orientation='horizontal')
        
cb = fig.colorbar(pcm,ax=axs[:2,:].flatten(),pad=0.01,fraction=0.025)
cb = cb.set_label("Damping Timescale $\lambda_a^{-1}$ (months)",fontsize=14)

cbdiff = fig.colorbar(pcmdiff,ax=axs[2,-1],pad=0.03,fraction=0.045)
cbdiff = cbdiff.set_label("Timescale Diff (months)",fontsize=12)

savename = "%sExpFit_Damping_Map_lagmax%02i_Seasonal_Ensemble_Avg_comparison.png" % (figpath,lagmaxes[ilagmax]-1,)
plt.savefig(savename,dpi=150,bbox_inches="tight",)

#%% Plot actual damping values

ivar    = 0
ilagmax = 1
clvls   = [3,6,12,24]

fig,axs = viz.geosubplots(3,4,figsize=(15,8))

for v in range(3):
    
    for s in range(4):
        ax = axs[v,s]
        
        # Select Plotting Variable
        if v == 0: # Plot Fit
            savg_in = savgs_fit[s][ivar,ilagmax,:,:]
            #vlms = [0,24]
            lab  = "Exp. Fit"
            cmap = "inferno_r"
        elif v == 1: # Plot estimate
            savg_in = savgs_est[s][:,:] * -1
            #vlms = [0,24]
            lab  = "Cov. Est."
            cmap = "inferno_r"
        else: # Fit minus estimate
            savg_in =  savgs_est[s][:,:] *-1 - savgs_fit[s][ivar,ilagmax,:,:]
            #vlms =[-12,12]
            lab  = "Est. - Fit"
            cmap = "RdBu_r"
        
        # Labeling + Setup ------
        blabel=[0,0,0,0]
        if s == 0:
            blabel[0] = 1
        if v == 2:
            blabel[-1] = 1
        ax = viz.add_coast_grid(ax,bboxplot,fill_color="k",blabels=blabel,fontsize=14)
        if v == 0:
            ax.set_title(snames[s],fontsize=16)
        if s == 0:
            ax.text(-0.22, 0.55, lab, va='bottom', ha='center',rotation='vertical',
                    rotation_mode='anchor',transform=ax.transAxes,fontsize=16)
            
        # Plotting
        if v < 2:
            plotvar = savg_in*-1
        else:
            plotvar = savg_in
        if v < 2:
            #pcm     = ax.pcolormesh(lon,lat,plotvar,vmin=vlms[0],vmax=vlms[1],cmap=cmap)
            pcm     = ax.pcolormesh(lon,lat,plotvar,cmap=cmap)
        else:
            #pcmdiff = ax.pcolormesh(lon,lat,plotvar,vmin=vlms[0],vmax=vlms[1],cmap=cmap)
            pcmdiff = ax.pcolormesh(lon,lat,plotvar,cmap=cmap)
        cl      = ax.contour(lon,lat,plotvar,levels=clvls,colors="w",linewidths=0.75)
        ax.clabel(cl,fontsize=14)
        
        #fig.colorbar(pcm,ax=ax,orientation='horizontal')
        
cb = fig.colorbar(pcm,ax=axs[:2,:].flatten(),pad=0.01,fraction=0.025)
cb = cb.set_label("Damping $\lambda_a^{-1}$ (months)",fontsize=14)

cbdiff = fig.colorbar(pcmdiff,ax=axs[2,-1],pad=0.03,fraction=0.045)
cbdiff = cbdiff.set_label("Timescale Diff (months)",fontsize=12)

savename = "%sExpFit_Damping_Map_lagmax%02i_Seasonal_Ensemble_Avg_comparison_dampingval.png" % (figpath,lagmaxes[ilagmax]-1,)
plt.savefig(savename,dpi=150,bbox_inches="tight",)


#%%
