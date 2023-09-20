#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Estimate Damping Parameter Using and Exponential Fit

Loop for the whole basin, copied from [point_case_study]

Created on Wed Sep  6 11:11:51 2023

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

figpath     = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/02_Figures/20230907/"
datpath_ac  = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/proc/"
lonf        = -30
latf        = 50

varnames    = ("SST","SSS")
recalculate = False

# Define which lagmaxes to fit exponential function over
lagmaxes    = [7,13,37]

# Information for loading covariance-estimated HFF
datpath_damping = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/01_Data/model_input/"


#%% Import Custom Modules

sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/03_Scripts/")
sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/03_Scripts/stochmod/model/")
from amv import proc,viz
import scm


#%% Read in datasets
ds_all = []
ac_all = []
for v in range(2):
    ds = xr.open_dataset("%sHTR-FULL_%s_autocorrelation_thres0.nc" % (datpath_ac,varnames[v]))
    ds  = ds.sel(thres="ALL").load()# [ens lag mon]
    ds_all.append(ds)
    ac_all.append(ds[varnames[v]].values) 
#%% Get some parameters

nens,nlags,nmon,nlat,nlon = ac_all[0].shape
lags                      = ds_all[0].lag.values



lon = ds_all[0].lon.values
lat = ds_all[0].lat.values


# Debug
e = 0
v = 0
a = 0
o = 0

#%% Check to see if file exists

savename = "%sCESM1_LENS_SST_SSS_lbd_exponential_fit.nc" % datpath_ac
exists = proc.checkfile(savename)
if (not exists) and recalculate:

    #% Fit The exponential (a dumb loop...) ----------------------------------
    expf3     = lambda t,b: np.exp(b*t)         # No c and A
    
    lm        = len(lagmaxes)
    
    lbd_fit   = np.zeros((2,lm,nens,nmon,nlat,nlon)) * np.nan # [variable,]
    funcin    = expf3
    problem_y = []
    for v in range(2):
        for e in range(nens):
            for a in tqdm(range(nlat)):
                for o in range(nlon):
                    
                    acpt = ac_all[v][e,:,:,a,o] # Lag x Month
                    
                    # Skip Land Points
                    if np.all(np.isnan(acpt)):
                        continue
                    
                    for im in range(nmon):
                        for l in range(lm):
                            lagmax = lagmaxes[l]
                            x = lags[:lagmax]
                            y = acpt[:lagmax,im]
                            
                            try:
                                popt, pcov = sp.optimize.curve_fit(funcin, x[:lagmax], y[:lagmax])
                            except:
                                print("Issue with ilat %i ilon %i"% (a,o))
                                problem_y.append(y)
                                continue
                            lbd_fit[v,l,e,im,a,o] = popt[0]

    #% Save the exponential Fit
    
    coords_dict = {
        'vars'   : list(varnames),
        'lag_max': lagmaxes,
        'ens'    : ds_all[0].ens.values,
        'mon'    : ds_all[0].mon.values,
        'lat'    : ds_all[0].lat.values,
        'lon'    : ds_all[0].lon.values
        }
    
    da       = xr.DataArray(lbd_fit,coords=coords_dict,dims=coords_dict,name="lbd")
    
    
    da.to_netcdf(savename,encoding={'lbd': {'zlib': True}})
else:
    da = xr.open_dataset(savename)
    lbd_fit = da.lbd.values
    

#%% Examine the map of estimated damping
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

#%% Do seasonally averaged maps

lm = 2
e  = "mean"

savgs,snames = proc.calc_savg(lbd_fit,axis=3,return_str=True) # [var x lagmax x ens x lat x lon]

# Take ensemble mean
savgs_mean = [np.mean(s,(2))[:,-1,...] for s in savgs]

#%% Make the plot
fig,axs = viz.geosubplots(2,4,figsize=(15,6))

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
        plotvar = 1/savgs_mean[s][v,...]*-1
        pcm     = ax.pcolormesh(lon,lat,plotvar,vmin=0,vmax=24,cmap="inferno_r")
        cl      = ax.contour(lon,lat,plotvar,levels=clvls,colors="w",linewidths=0.75)
        ax.clabel(cl,fontsize=14)
        #fig.colorbar(pcm,ax=ax,orientation='horizontal')
cb = fig.colorbar(pcm,ax=axs.flatten(),pad=0.01,fraction=0.025)
cb = cb.set_label("Damping Timescale $\lambda_a^{-1}$ (months)",fontsize=14)

savename = "%sExpFit_Damping_Map_lagmax%02i_Seasonal_Ensemble_Avg.png" % (figpath,lagmaxes[lm]-1,)
plt.savefig(savename,dpi=150,bbox_inches="tight",)

#%% Load Explicitly Estimated Damping for comparison --------------------------

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
lbd_est = (cv_damping * mask * dt) / (rho * cp *mld)
lbd_est = lbd_est.transpose(2,0,1) # [mon x lat x lon]
#%% Take Ensemble mean of lbd_fit


lbd_fit_ensmean = lbd_fit.mean(2)  # [var x lagmax x mon x lat x lon]

# Lets first look at SST damping timescales and their differences
sst_lbd_diff = lbd_fit_ensmean[0,:,:,:,:] - lbd_est[None,:,:,:]


#%%


plt.pcolormesh(-1/sst_lbd_diff[0,0,:,:],vmin=0,vmax=10),plt.colorbar()


#%% Look at the seasonalaverages

savgs_fit,snames = proc.calc_savg(lbd_fit_ensmean,axis=2,return_str=True) #[seas][var x lagmax x lat x lon]
savgs_est,_      = proc.calc_savg(lbd_est,axis=0,return_str=True) # [seas][lat x lon]


#%% Make the plot

ivar    = 0
ilagmax = 0
clvls   = [-24,-12,-6,-3,0,3,6,12,24]

fig,axs = viz.geosubplots(3,4,figsize=(15,8))

for v in range(3):
    
    for s in range(4):
        ax = axs[v,s]
        
        # Select Plotting Variable
        if v == 0: # Plot Fit
            savg_in = savgs_fit[s][ivar,ilagmax,:,:]
            vlms = [0,24]
            lab  = "Exp. Fit"
            cmap = "inferno_r"
        elif v == 1: # Plot estimate
            savg_in = savgs_est[s][:,:] * -1
            vlms = [0,24]
            lab  = "Cov. Est."
            cmap = "inferno_r"
        else: # Fit minus estimate
            savg_in =  1/savgs_est[s][:,:] *-1 - 1/savgs_fit[s][ivar,ilagmax,:,:]
            vlms =[-12,12]
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
            plotvar = 1/savg_in*-1
        else:
            plotvar = savg_in#1/savg_in
        if v < 2:
            pcm     = ax.pcolormesh(lon,lat,plotvar,vmin=vlms[0],vmax=vlms[1],cmap=cmap)
        else:
            pcmdiff = ax.pcolormesh(lon,lat,plotvar,vmin=vlms[0],vmax=vlms[1],cmap=cmap)
        cl      = ax.contour(lon,lat,plotvar,levels=clvls,colors="w",linewidths=0.75)
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
ilagmax = 0
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