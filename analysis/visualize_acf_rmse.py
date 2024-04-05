#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Visualize/Sort Points by RMSE of ACFs between Stochastic Model
and CESM1 Outputs.

Works with output from the following scripts: 
    - [pointwise_autocorrelation_smoutput.py] (Calculates pointwise ACF from stochastic model (sm) output)
    - []

Inputs:
------------------------
    
    varname : dims                              - units                 - processing script

Outputs: 
------------------------

    varname : dims                              - units 

What does this script do?

Created on Tue Mar  5 10:36:18 2024

@author: gliu
"""

import numpy as np
import xarray as xr
import sys
import time
import matplotlib.pyplot as plt
import matplotlib as mpl
import cartopy.crs as ccrs
import os
from tqdm import tqdm

# ----------------------------------
#%% Import custom modules and paths
# ----------------------------------

# Indicate the Machine!
machine = "Astraeus"

# First Load the Parameter File
sys.path.append("../")
import reemergence_params as rparams

# Paths and Load Modules
pathdict   = rparams.machine_paths[machine]

sys.path.append(pathdict['amvpath'])
sys.path.append(pathdict['scmpath'])
from amv import proc,viz
import scm
import amv.loaders as dl
import yo_box as ybx

# Set needed paths
figpath     = pathdict['figpath']
proc.makedir(figpath)
input_path  = pathdict['input_path']
output_path = pathdict['output_path']
procpath    = pathdict['procpath']

# ----------------------------------
#%% User Edits
# ----------------------------------

# Indicate the experiment




# Set Names for the Stochastic Model Experiment and CESM

# SSS Comparison
expname         = "SSS_EOF_Qek_LbddEnsMean"
fn_sm           = "SM_SSS_EOF_Qek_LbddEnsMean_SSS_autocorrelation_thresALL_lag00to60.nc"
fn_cesm         = "HTR-FULL_SSS_autocorrelation_thres0_lag00to60.npz"
vname           = "SSS"

# SST Comparison
# expname         = "SSS_EOF_Qek_pilot_corrected"
# fn_sm           = "SM_SSS_EOF_Qek_pilot_corrected_SSS_autocorrelation_thresALL_lag00to60.nc"
# fn_cesm         = "HTR-FULL_SSS_autocorrelation_thres0_lag00to60.npz"
# vname           = "SSS"


# Load the parameter dictionary
expparams_raw   = np.load("%s%s/Input/expparams.npz" % (output_path,expname),allow_pickle=True)

# Set names for land ice mask (this is manual, and works just on Astraeus :(...!)
lipath          = "/Users/gliu/Downloads/02_Research/01_Projects/04_Predict_AMV/03_Scripts/CESM_data/Masks/"
liname          = "CESM1LE_HTR_limask_pacificmask_enssum_lon-90to20_lat0to90.nc"


# ----------------------------------
#%% Load the output
# ----------------------------------


ds_sm = xr.open_dataset(procpath+fn_sm).load()

ld_cesm = np.load(procpath+fn_cesm,allow_pickle=True) # (65, 69, 42, 12, 3, 61)
print(ld_cesm.files) # lon x lat x ens x mon x thres x lags

# Place CESM data into DataArray (this step will eventually not be needed)
ccoords = dict(lon=ld_cesm['lon'],
               lat=ld_cesm['lat'],
               ens=np.arange(1,43,1),
               mons=np.arange(1,13,1),
               lags=np.arange(0,61,1)
               )
ds_cesm = xr.DataArray(ld_cesm['acs'][:,:,:,:,-1,:],coords=ccoords,dims=ccoords)

# Load Land Ice Mask
ds_mask          = xr.open_dataset(lipath+liname).MASK.squeeze().load()

# Resize DS
ds_list     = [ds_sm,ds_cesm,ds_mask]
ds_list_rsz = proc.resize_ds(ds_list)
ds_sm,ds_cesm,ds_mask=ds_list_rsz

# Apply Mask
# ds_sm = ds_sm #* ds_mask
# ds_cesm = ds_cesm #* ds_mask

# Edit
plotmask = ds_mask.values.copy()
plotmask[np.isnan(plotmask)] = 0.

maskcoast = ds_mask.values.copy()
maskcoast = np.roll(maskcoast,1,axis=0) * np.roll(maskcoast,-1,axis=0) * np.roll(maskcoast,1,axis=1) * np.roll(maskcoast,-1,axis=1)

# ----------------------------------
#%% Compute RMSE
# ----------------------------------

# Load out ACFs
ac_cesm      = ds_cesm.mean('ens')       # {lon x lat x mon x lag}
ac_sm        = ds_sm['SSS'].squeeze()    # {lon x lat x mon x lag}

# Compute the RMSE
rmse_mon     = np.sqrt(((ac_cesm - ac_sm)**2).mean('lags')) # [lon x lat x mon]
abserr_bylag = np.abs(ac_cesm - ac_sm)

# ----------------------------------------
#%% Do preliminary visualizations of ACFs
# ----------------------------------------

ds = ds_sm # Indicate ds containing lat/lon

# Plotting Params
mpl.rcParams['font.family'] = 'JetBrains Mono'
bboxplot                    = [-80,0,20,65]
proj                        = ccrs.PlateCarree()
lon                         = ds.lon.values
lat                         = ds.lat.values
mons3                       = proc.get_monstr()

plotmon                     = np.roll(np.arange(12),1)

fsz_title                   = 26
fsz_axis                    = 22
fsz_lbl                     = 10

def init_monplot():
    plotmon       = np.roll(np.arange(12),1)
    fig,axs,mdict = viz.init_orthomap(4,3,bboxplot=bboxplot,figsize=(18,18))
    for a,ax in enumerate(axs.flatten()):
        im = plotmon[a]
        ax = viz.add_coast_grid(ax,bbox=bboxplot,fill_color="lightgray")
        ax.set_title(mons3[im],fontsize=fsz_axis)
    return fig,axs

#%% Plot Monthly RMSE in ACF

vname          = 'ACF RMSE'
vname_long     = "RMSE (CESM - Stochastic Model)"
vlabel         = "$RMSE$ (correlation)"
plotcontour    = False
vlms           = [0,0.75]
cints_sp       = None# np.arange(0,66,12)#None#np.arange(200,1500,100)# None
cmap           = 'cmo.amp'

# Get variable, lat, lon
selvar      = rmse_mon

# Preprocessing

fig,axs = init_monplot()
for aa in range(12):
    ax      = axs.flatten()[aa]
    im      = plotmon[aa]
    plotvar = selvar.isel(mons=im).T 
    
    # Just Plot the contour with a colorbar for each one
    if vlms is None:
        pcm = ax.pcolormesh(lon,lat,plotvar,transform=proj,cmap=cmap,zorder=-3)
        fig.colorbar(pcm,ax=ax)
    else:
        pcm = ax.pcolormesh(lon,lat,plotvar,transform=proj,
                            cmap=cmap,vmin=vlms[0],vmax=vlms[1],zorder=-3)
        
    cl = ax.contour(lon,lat,plotmask,colors="k",linestyles='dotted',linewidths=.95,
                    levels=[0,1],transform=proj,zorder=-2)

if vlms is not None:
    cb = fig.colorbar(pcm,ax=axs.flatten(),
                      orientation='horizontal',pad=0.02,fraction=0.025)
    cb.set_label(vlabel)

plt.suptitle("%s" % (vname_long),fontsize=fsz_title)
savename = "%s%s_ACF_RMSE_seasonrow.png" % (figpath,expname)
plt.savefig(savename,dpi=150,bbox_inches='tight')


#%% Lets Examine things at a point to see how they look
kmonth         = 0

#for kmonth in range(12):

xticks         = np.arange(0,66,6)
lags           = ds_sm.lags
lonf           = -30
latf           = 50

locfn,loctitle = proc.make_locstring(lonf,latf,lon360=True)

title          = "SSS ACF (Lag 0 = Feb) @ %s\n RMSE = %.2f" % (loctitle,rmse_mon.sel(lon=lonf,lat=latf,method='nearest').isel(mons=kmonth))
fig,ax         = plt.subplots(1,1,figsize=(8.5,4),constrained_layout=True)
ax,_           = viz.init_acplot(kmonth,xticks,lags,title=title,ax=ax)
ax             = viz.add_ticks(ax=ax,)

ax.plot(lags,ac_sm.sel(lon=lonf,lat=latf,method='nearest').isel(mons=kmonth),
        color='orange',lw=2.5,label="Stochastic Model")
ax.plot(lags,ac_cesm.sel(lon=lonf,lat=latf,method='nearest').isel(mons=kmonth),
        color='k',lw=2.5,label="CESM1 Historical (Ens. Avg.)")

ax.legend()

#%% Focus on points where values are min/max for a particular month

# Calculations
plotvar   = rmse_mon.isel(mons=kmonth).T
plotvar   = xr.where(plotvar==0.,np.nan,plotvar) # Set Points to Zero
plotvar   = plotvar * maskcoast #* ds_mask  # Apply MASK to ice points

plotvarnp = plotvar.values
idsort1d  = np.argsort(plotvarnp.flatten())
idsort2d  = np.unravel_index(idsort1d,plotvarnp.shape)

# Remove Nans
sorted_values = []
for ii in tqdm(range(len(idsort1d))):
    sorted_values.append(plotvar[idsort2d[0][ii],idsort2d[1][ii]].values.item())
idnan = np.where(np.isnan(sorted_values))[0][0]
idsort2d = [idsort2d[0][:idnan],idsort2d[1][:idnan]]

#%%

topn = 100

# Intialize Map and plot values
fig,ax,_ = viz.init_orthomap(1,1,bboxplot=bboxplot,figsize=(12,10))
ax       = viz.add_coast_grid(ax,bbox=bboxplot,fill_color="lightgray")
pcm      = ax.pcolormesh(lon,lat,plotvar,transform=proj,
                    cmap=cmap,vmin=vlms[0],vmax=vlms[1],zorder=-3)


# Plot top and bottom n points
# Top N
acf_top = []
acf_bot = []
for ii in range(topn):
    if ii == 0:
        labels=["Lowest %i RMSE" % topn,'Highest %i RMSE' % topn]
    else:
        labels=["",]*2
    
    ax.scatter(lon[idsort2d[1][ii]],lat[idsort2d[0][ii]],marker="o",color="cornflowerblue",transform=proj,label=labels[0])
    ax.scatter(lon[idsort2d[1][-(ii+1)]],lat[idsort2d[0][-(ii+1)]],marker="x",color="yellow",transform=proj,label=labels[1])
    
    # Get ACFs 
    
ax.legend()

# Colorbar Stuff
cb       = fig.colorbar(pcm,ax=ax,
                  orientation='horizontal',pad=0.02,fraction=0.025)
cb.set_label(vlabel)
ax.set_title("%s ACF RMSE (CESM1 - Stochastic Model)" % (mons3[kmonth]),fontsize=fsz_title)

savename = "%s%s_ACF_RMSE_topbot%03i_mon%02i.png" % (figpath,expname,topn,kmonth+1)
plt.savefig(savename,dpi=150,bbox_inches='tight')
#def get_ids()

#%% Plot the ACFs of these points

plottop            = True
topn               = 5

for plottop in [True,False]:
    
    cmapn   = plt.get_cmap('Accent',topn)
    cmapn   = [mpl.colors.rgb2hex(cmapn(i)) for i in range(topn)]
    if plottop:
        title          = "SSS ACF, Top %i RMSE (CESM=solid, Stochastic Model=dashed)" % (topn)
    else:
        title          = 'SSS ACF, Bot. %i RMSE (CESM=solid, Stochastic Model=dashed)' % (topn)
     
    fig,ax         = plt.subplots(1,1,figsize=(8.5,4.5),constrained_layout=True)
    ax,_           = viz.init_acplot(kmonth,xticks,lags,title=title,ax=ax)
    ax             = viz.add_ticks(ax=ax,)
    
    for ii in range(topn):
        
        if plottop:
            ii_in = ii
        else:
            ii_in = -(ii+1)
        
        klon = idsort2d[1][ii_in]
        klat = idsort2d[0][ii_in]
        
        _,loctitle=proc.make_locstring(lon[klon],lat[klat])
        
        ax.plot(lags,ac_sm.isel(mons=kmonth,lon=klon,lat=klat),c=cmapn[ii],alpha=1,label=loctitle,ls='dashed')
        ax.plot(lags,ac_cesm.isel(mons=kmonth,lon=klon,lat=klat),c=cmapn[ii],alpha=0.75)
    ax.legend(ncol=2)
    
    
    savename = "%s%s_ACF_RMSE_topbot%03i_mon%02i_plottop%i.png" % (figpath,expname,topn,kmonth+1,plottop)
    plt.savefig(savename,dpi=150,bbox_inches='tight')


#%% Plot the corresponding locator plot

# Intialize Map and plot values
fig,ax,_ = viz.init_orthomap(1,1,bboxplot=bboxplot,figsize=(12,10))
ax       = viz.add_coast_grid(ax,bbox=bboxplot,fill_color="lightgray")
pcm      = ax.pcolormesh(lon,lat,plotvar,transform=proj,
                    cmap=cmap,vmin=vlms[0],vmax=vlms[1],zorder=-3)


# Plot top and bottom n points
# Top N
acf_top = []
acf_bot = []
for ii in range(topn):
    if ii == 0:
        labels=["Lowest %i RMSE" % topn,'Highest %i RMSE' % topn]
    else:
        labels=["",]*2
    
    ax.scatter(lon[idsort2d[1][ii]],lat[idsort2d[0][ii]],marker="o",s=100,color=cmapn[ii],transform=proj,label=labels[0])
    ax.scatter(lon[idsort2d[1][-(ii+1)]],lat[idsort2d[0][-(ii+1)]],s=100,marker="x",color=cmapn[ii],transform=proj,label=labels[1])
    
    # Get ACFs 

ax.legend()

# Colorbar Stuff
cb       = fig.colorbar(pcm,ax=ax,
                  orientation='horizontal',pad=0.02,fraction=0.025)
cb.set_label(vlabel)
ax.set_title("%s ACF RMSE (CESM1 - Stochastic Model)" % (mons3[kmonth]),fontsize=fsz_title)

savename = "%s%s_ACF_RMSE_topbot%03i_mon%02i_locator.png" % (figpath,expname,topn,kmonth+1)
plt.savefig(savename,dpi=150,bbox_inches='tight')

    # if ii == 0:
    #     labels=["Top %i RMSE" % topn,'Bottom %i RMSE' % topn]
    # else:
    #     labels=["",]*2

# -------------------------------
#%% Analyze T2 for each of these
# -------------------------------

# 
t2_cesm = proc.calc_T2(ac_cesm.values,3)
t2_sm   = proc.calc_T2(ac_sm.values,3)



#%% Make seasonal plots of T2


plotcontour    = False

cints_sp       = None# np.arange(0,66,12)#None#np.arange(200,1500,100)# None
vname          = 'T2'  
vlabel         = "$T_2$ (months)"

for ii in range(3):
    if ii == 0:
        selvar         = t2_cesm
        vname_long     = "Decorrelation Timescale ($T_2$, CESM1)"
        cmap           = 'cmo.dense'
        vlms           = [0,48]#,[0,5]

    elif ii == 1:
        selvar         = t2_sm
        vname_long     = "Decorrelation Timescale ($T_2$, Stochastic Model)"
        cmap           = 'cmo.dense'
        vlms           = [0,48]#,[0,5]
    elif ii == 2:
        selvar         = t2_cesm-t2_sm
        vname_long     = "$\Delta \, T_2$, (CESM1 - Stochastic Model)"
        cmap           = 'cmo.balance'
        vlms           = [-12,12]#,[0,5]
        
    
    # Get variable, lat, lon
    
    
    # Preprocessing
    
    fig,axs = init_monplot()
    for aa in range(12):
        ax      = axs.flatten()[aa]
        im      = plotmon[aa]
        plotvar = selvar[:,:,im].T#.isel(mons=im).T 
        
        # Just Plot the contour with a colorbar for each one
        if vlms is None:
            pcm = ax.pcolormesh(lon,lat,plotvar,transform=proj,cmap=cmap,zorder=-3)
            fig.colorbar(pcm,ax=ax)
        else:
            pcm = ax.pcolormesh(lon,lat,plotvar,transform=proj,
                                cmap=cmap,vmin=vlms[0],vmax=vlms[1],zorder=-3)
            
        cl = ax.contour(lon,lat,plotmask,colors="k",linestyles='dotted',linewidths=.95,
                        levels=[0,1],transform=proj,zorder=-2)
    
    if vlms is not None:
        cb = fig.colorbar(pcm,ax=axs.flatten(),
                          orientation='horizontal',pad=0.02,fraction=0.025)
        cb.set_label(vlabel)
    
    plt.suptitle("%s" % (vname_long),fontsize=fsz_title)
    savename = "%s%s_ACF_RMSE_seasonrow.png" % (figpath,expname)
    plt.savefig(savename,dpi=150,bbox_inches='tight')


#%% Plot correlation with dataarray averaged over a section


