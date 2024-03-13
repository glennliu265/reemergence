#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

OSM 2024 Plots for T2

Created on Thu Feb  8 01:55:53 2024

@author: gliu

"""

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import sys
from tqdm import tqdm
import copy
import glob

import matplotlib as mpl

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
figpath     = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/01_hfdamping/02_Figures/20240216/"
proc.makedir(figpath)



#%% Set Experiment Names

expnames = [
    "SM_SST_OSM_Tddamp",
    "SM_SST_OSM_Tddamp_noentrain",
    "SM_SSS_OSM_Tddamp",
    "SM_SSS_OSM_Tddamp_noentrain"]
vnames   = ["SST",
            "SST",
            "SSS",
            "SSS"]

expnames_long = [
    "Stochastic Model, Entraining",
    "Stochastic Model, Non-entraining",
    "Stochastic Model, Entraining",
    "Stochastic Model, Non-entraining"
    
    ]
nexps   = len(expnames)
ldnames = [ "%s%s_%s_autocorrelation_thresALL_lag00to60.nc" % (datpath,expnames[ex],vnames[ex],) for ex in range(nexps)]


cesm_files = ["HTR-FULL_SST_autocorrelation_thres0_lag00to60.npz","HTR-FULL_SSS_autocorrelation_thres0_lag00to60.npz"]

ldsst = np.load(datpath+cesm_files[0],allow_pickle=True)
ldsss = np.load(datpath+cesm_files[1],allow_pickle=True)
print(ldsst.files)
print(ldsss.files)

#%%  Load data for stochastic model

ds_sm   = [xr.open_dataset(ldnames[nn])[vnames[nn]] for nn in range(nexps)]
acfs_sm = [ds.values.squeeze() for ds in ds_sm] # [lon x lat x kmonth x thres x lag]

t2_sm   = [proc.calc_T2(acf,axis=3) for acf in acfs_sm] # [(65, 69, 12)]


#%% Load data for CESM1

ld_cesm = [np.load(datpath+cesm_files[ii],allow_pickle=True) for ii in range(2)]
acfs_cesm = [ld['acs'][:,:,:,:,-1,:] for ld in ld_cesm] # [ lon x lat x ens x mon x thres x lag], subset threshold -1

t2cesm = [proc.calc_T2(acf,axis=-1) for acf in acfs_cesm]


# Place into dataarray
ld     = ld_cesm[0]
coords = dict(lon=ld['lon'],lat=ld['lat'],ens=np.arange(1,43,1),mons=np.arange(1,13,1),lags=ld['lags'])

acfscesm = [xr.DataArray(vv,coords=coords,dims=coords) for vv in acfs_cesm]

#%% Make Land-Ice Mask

# Make Land Ice Mask
msk = ds_sm[0].prod(('mons','thres','lags')).values
mskcontour = msk.copy()
msk[msk==1] = np.nan
msk[~np.isnan(msk)] = 1
plt.pcolormesh(msk.T)

# Apply Land Mask
bbsim  = [-80,0,0,65] # Simulation Box
maskpath = "/Users/gliu/Downloads/02_Research/01_Projects/04_Predict_AMV/03_Scripts/CESM_data/Masks/"
masknc   = "CESM1LE_HTR_landmask_allens.nc"
maskds   = xr.open_dataset(maskpath+masknc)

mask180 = proc.format_ds(maskds)
mask180 = proc.sel_region_xr(mask180,bbsim).LANDMASK.values#.prod('time') # Time is actually ensemble

maskreg = mask180.prod(0)




#%% Set some plotting parameters

# Plotting Params by Variable
vnames = ["SST","SSS"]
vcmaps = ["cmo.dense","cmo.deep"]
levels_byvar = [np.arange(0,21,1),np.arange(0,32,2)]

bbplot = [-80,0,15,65]

mons3  = proc.get_monstr()
mpl.rcParams['font.family'] = 'Avenir'
fsz_tick=18
fsz_axis=20
fsz_title=24
fsz_ct = 16 # Fontsize of counturs

figsize1 = (10,8.5)

idwint = [0,1,11]

ds = ds_sm[0]
lon = ds.lon.values
lat = ds.lat.values

lc = 'midnightblue'

# Make a regional mask, also mask out regions south of the bounding box
#maskreg = proc.sel_region_xr(mask180,bbsim).LANDMASK
southid = np.where(lat<bbplot[2]-1)[0]
# Manual fix under certain level
maskreg[southid,:] = np.nan

#%% Plot CESM Results (OSM 2024 Poster PLots!)

plot_title   = False
fsc          = (10,8.5) #(10,7.5)
cesm_wintmap = []

for ii in range(2):
    fig,ax,mdict=viz.init_orthomap(1,1,bbplot,figsize=fsc,centlat=45,)
    ax = viz.add_coast_grid(ax,bbox=bbplot,fill_color='lightgray',fontsize=20)
    
    if plot_title:
        ax.set_title(vnames[ii] + " (CESM1)",fontsize=fsz_title)
    
    cmap = vcmaps[ii]
    
    plotvar = t2cesm[ii].mean(2)[:,:,idwint].mean(-1).T  * maskreg#* maskreg#* msk.T
    plotvar[plotvar==1.] = np.nan
    pcm     = ax.contourf(ds.lon,ds.lat,plotvar,cmap=cmap,transform=mdict['noProj'],levels=levels_byvar[ii],zorder=-1)
    cl      = ax.contour(lon,lat,plotvar,colors='darkslategray',linewidths=.5,linestyles='solid',levels=levels_byvar[ii],transform=mdict['noProj'])
    ax.clabel(cl,levels_byvar[ii][::2],
              fontsize=fsz_ct,colors="k",use_clabeltext=True)
    
    clm = ax.contour(lon,lat,mskcontour.T,levels=[0,1],linestyles='dashed',linewidths=0.75,transform=mdict['noProj'],colors="w")
    
    # Add Colorbar
    cb = fig.colorbar(pcm,ax=ax,fraction=0.045,orientation='horizontal',pad=0.01,)#size=16)
    cb.ax.tick_params(labelsize=fsz_tick)
    cb.set_label("Persistence Timescale [Months]",fontsize=fsz_axis)
    
    savename = "%sCESM1_%s_T2_wintAvg_EnsAvg.png" % (figpath,vnames[ii],)
    plt.savefig(savename,dpi=200,bbox_inches='tight',transparent=True)
    
    cesm_wintmap.append(plotvar)
    

#%% Plot mean winter T2 for the stochastic model


levels = np.arange(0,25,1)

for ex in range(nexps):
    if ex < 2:
        ii = 0
    else:
        ii = 1
    
    fig,ax,mdict=viz.init_orthomap(1,1,bbplot,figsize=fsc)
    
    ax = viz.add_coast_grid(ax,bbox=bbplot,fill_color='lightgray')    
    
    plotvar = t2_sm[ex][:,:,idwint].mean(-1).T
    if ii == 0: # Explicit LEvels for SST
    
        pcm     = ax.contourf(ds.lon,ds.lat,plotvar,cmap=vcmaps[ii],transform=mdict['noProj'],levels=levels_byvar[ii],extend='both')
        cl      = ax.contour(lon,lat,plotvar,colors='darkslategray',linewidths=.5,linestyles='solid',levels=levels_byvar[ii],transform=mdict['noProj'])
        ax.clabel(cl,levels_byvar[ii][::2],fontsize=fsz_tick)
        
    else:
        
        pcm     = ax.contourf(ds.lon,ds.lat,plotvar,cmap=vcmaps[ii],transform=mdict['noProj'],extend='both')
        cl      = ax.contour(lon,lat,plotvar,colors='darkslategray',linewidths=.5,linestyles='solid',transform=mdict['noProj'])
        ax.clabel(cl,fontsize=fsz_tick)
    
    ax.set_title("%s (%s)" % (vnames[ii],expnames_long[ex]),fontsize=fsz_title)
    
    # Add Colorbar
    cb = fig.colorbar(pcm,ax=ax,fraction=0.025,orientation='horizontal',pad=0.01,)#size=16)
    cb.ax.tick_params(labelsize=fsz_tick)
    cb.set_label("Timescale (Month)",fontsize=fsz_axis)
    
    savename = "%sSM_%s_T2_wintAvg_EnsAvg.png" % (figpath,expnames[ex],)
    plt.savefig(savename,dpi=150,bbox_inches='tight',transparent=True)

#%% Examine Difference in T2 (adding Entrainment)


maskregdiff = maskreg.T[...,None]

t2_diff = []
t2_diff.append(t2_sm[0] * maskregdiff  - t2_sm[1]* maskregdiff ) # SST
t2_diff.append(t2_sm[2] * maskregdiff - t2_sm[3]* maskregdiff ) # SSS
t2_diff.append(t2_sm[0] * maskregdiff - t2cesm[0].mean(2)* maskregdiff  )
t2_diff.append(t2_sm[2] * maskregdiff  - t2cesm[1].mean(2)* maskregdiff )

diffnames = ["SST (entrain - no entrain)",
             "SSS (entrain - no entrain)",
             "SST (Stochastic Model - CESM1)",
             "SSS (Stochastic Model - CESM1)"]


#%% Diff Plot 1 (Entrain vs Non Entrain, SST)


dd = 0
ii = 0
plot_title=False

if ii == 0:
    
    levels = np.arange(-15,16,1)
else:
    levels = np.arange(-30,33,3)
cmap = 'cmo.balance'

fig,ax,mdict=viz.init_orthomap(1,1,bbplot,figsize=fsc,centlat=45)

ax = viz.add_coast_grid(ax,bbox=bbplot,fill_color='lightgray',fontsize=20)    


plotvar = t2_diff[dd][:,:,idwint].mean(-1).T  * msk.T

plotvar[plotvar==0.] = np.nan
pcm     = ax.contourf(ds.lon,ds.lat,plotvar ,cmap=cmap,transform=mdict['noProj'],
                      levels=levels,extend='both',zorder=-1)
cl      = ax.contour(lon,lat,plotvar,colors='darkslategray',linewidths=.5,linestyles='solid',levels=levels,transform=mdict['noProj'])

ax.clabel(cl,levels[::2],
              fontsize=fsz_ct,colors="k",use_clabeltext=True)
if plot_title:
    ax.set_title("%s" % (diffnames[dd]),fontsize=fsz_title)

# Add Colorbar
cb = fig.colorbar(pcm,ax=ax,fraction=0.045,orientation='horizontal',pad=0.01,)#size=16)
cb.ax.tick_params(labelsize=fsz_tick)
cb.set_label("Timescale Difference [Months]",fontsize=fsz_axis)

savename = "%sT2_diff_wintAvg_diffnum%0i_EnsAvg.png" % (figpath,dd)
plt.savefig(savename,dpi=150,bbox_inches='tight',transparent=True)


#%%

selmon = [0,1,11]
bbplot = [-80,0,20,65]
for dd in range(4):
    
    fig,ax,mdict=viz.init_orthomap(1,1,bbplot,figsize=(8,4))
    ax = viz.add_coast_grid(ax,bbox=bbplot,fill_color='lightgray')    
    
    plotvar = t2_diff[dd][:,:,selmon].mean(-1).T
    
    pcm = ax.contourf(ds.lon,ds.lat,plotvar,cmap='cmo.balance',transform=mdict['noProj'])
    #pcm = ax.contourf(ds.lon,ds.lat,plotvar,cmap='cmo.balance',transform=mdict['noProj'])
    ax.set_title(diffnames[dd])
    
    fig.colorbar(pcm,ax=ax)

#%% Plot/Examine ACFs over a region. First, lets select a region

ii = 1

regsel = [-45,-25,50,60]

fig,ax,mdict=viz.init_orthomap(1,1,bbplot,figsize=fsc)
ax = viz.add_coast_grid(ax,bbox=bbplot,fill_color='lightgray')
ax.set_title(vnames[ii] + " (CESM1)",fontsize=fsz_title)


cmap = vcmaps[ii]

plotvar = cesm_wintmap[ii]
pcm     = ax.contourf(ds.lon,ds.lat,plotvar,cmap=cmap,transform=mdict['noProj'],levels=levels_byvar[ii],zorder=-1)
cl      = ax.contour(lon,lat,plotvar,colors='darkslategray',linewidths=.5,linestyles='solid',levels=levels_byvar[ii],transform=mdict['noProj'])
ax.clabel(cl,levels_byvar[ii][::2],fontsize=fsz_tick)

clm = ax.contour(lon,lat,mskcontour.T,levels=[0,1],linestyles='dashed',linewidths=0.75,transform=mdict['noProj'],colors="w")

# Add Colorbar
cb = fig.colorbar(pcm,ax=ax,fraction=0.025,orientation='horizontal',pad=0.01,)#size=16)
cb.ax.tick_params(labelsize=fsz_tick)
cb.set_label("Timescale (Month)",fontsize=fsz_axis)

ln = viz.plot_box(regsel,ax=ax,proj=mdict['noProj'])


#%% Select and plot ACFs for a region (SPG)

kmonth = 1

dsregsm = [proc.sel_region_xr(ds,regsel).isel(thres=0,mons=kmonth) for ds in ds_sm]
dsregc  = [proc.sel_region_xr(ds,regsel).mean('ens').isel(mons=kmonth) for ds in acfscesm]


#ds_in = [acfs[0][:,:,kmonth,:] for acfs in acfs_sm] # Select month

# savename = "%sCESM1_%s_T2_wintAvg_EnsAvg.png" % (figpath,vnames[ii],)
# plt.savefig(savename,dpi=150,bbox_inches='tight',transparent=True)

#cesm_wintmap.append(plotvar)

dsreg = dsregsm[0]

lonr = dsreg.lon.values
latr = dsreg.lat.values

nlonr = len(lonr)
nlatr = len(latr)

#%% SST comparison over region

xtksl   = np.arange(0,37,3)
lags3   = np.arange(0,37,1)
lags6   = ld['lags']


plot_ds = [dsregsm[0],dsregsm[1],dsregc[0]] # 
pdcolor = ["orange","magenta","k"]
pdnames = ["Stochastic Model (entrain)","Stochastic Model (non-entraining)","CESM1"]


fig,ax = plt.subplots(1,1,figsize=(8,3.5),constrained_layout=True)
ax,_   = viz.init_acplot(kmonth,xtksl,lags3,title="")



# Plot Individual points
for aa in range(nlatr):
    for oo in range(nlonr):
        for ii in range(3):
            plotvar = plot_ds[ii].isel(lon=oo,lat=aa)
            ax.plot(lags6,plotvar,alpha=0.01,label="",color=pdcolor[ii],zorder=-1)
        
# Now plot the regional average
acfmean = []
for ii in range(3):
    plotvar = plot_ds[ii].mean(('lon','lat'))
    ax.plot(lags6,plotvar,alpha=1,label=pdnames[ii] + ", Region Avg.",color=pdcolor[ii],zorder=9)
    acfmean.append(plotvar)
ax.legend(fontsize=12,framealpha=0)
ax.tick_params(axis='both', which='major', labelsize=12)
#ax.tick_params(bottom=True,top=True,left=True,right=True,which='both')

savename = "%sSM_v_CESM_SST_ACF_EnsAvg_SPGBox.png" % (figpath,)
plt.savefig(savename,dpi=150,bbox_inches='tight',transparent=True)


#%% SSS comparison over region

xtksl   = np.arange(0,37,3)
lags3   = np.arange(0,37,1)
lags6   = ld['lags']


plot_ds = [dsregsm[2],dsregsm[3],dsregc[1]] # 
pdcolor = ["orange","magenta","k"]
pdnames = ["Stochastic Model (entrain)","Stochastic Model (non-entraining)","CESM1"]


fig,ax = plt.subplots(1,1,figsize=(8,3.5),constrained_layout=True)
ax,_   = viz.init_acplot(kmonth,xtksl,lags,title="")



# Plot Individual points
for aa in range(nlatr):
    for oo in range(nlonr):
        for ii in range(3):
            plotvar = plot_ds[ii].isel(lon=oo,lat=aa)
            ax.plot(lags6,plotvar,alpha=0.01,label="",color=pdcolor[ii],zorder=-1)
        
# Now plot the regional average
acfmean = []
for ii in range(3):
    plotvar = plot_ds[ii].mean(('lon','lat'))
    ax.plot(lags6,plotvar,alpha=1,label=pdnames[ii] + ", Region Avg.",color=pdcolor[ii],zorder=9)
    acfmean.append(plotvar)
#ax.legend(fontsize=12,framealpha=0)
ax.tick_params(axis='both', which='major', labelsize=12)


savename = "%sSM_v_CESM_SSS_ACF_EnsAvg_SPGBox.png" % (figpath,)
plt.savefig(savename,dpi=150,bbox_inches='tight',transparent=True)


#%% WIll COPY ABOVE
#%% STG Region

regsel = [-70,-55,30,40]

#

#%% Select and plot ACFs for a region (SPG)

kmonth = 1

dsregsm = [proc.sel_region_xr(ds,regsel).isel(thres=0,mons=kmonth) for ds in ds_sm]
dsregc  = [proc.sel_region_xr(ds,regsel).mean('ens').isel(mons=kmonth) for ds in acfscesm]


#ds_in = [acfs[0][:,:,kmonth,:] for acfs in acfs_sm] # Select month

# savename = "%sCESM1_%s_T2_wintAvg_EnsAvg.png" % (figpath,vnames[ii],)
# plt.savefig(savename,dpi=150,bbox_inches='tight',transparent=True)

#cesm_wintmap.append(plotvar)

dsreg = dsregsm[0]

lonr = dsreg.lon.values
latr = dsreg.lat.values

nlonr = len(lonr)
nlatr = len(latr)

#%% SST comparison over region

xtksl   = np.arange(0,37,3)
lags3   = np.arange(0,37,1)
lags6   = ld['lags']


plot_ds = [dsregsm[0],dsregsm[1],dsregc[0]] # 
pdcolor = ["orange","magenta","k"]
pdnames = ["Stochastic Model (entrain)","Stochastic Model (non-entraining)","CESM1"]


fig,ax = plt.subplots(1,1,figsize=(8,3.5),constrained_layout=True)
ax,_   = viz.init_acplot(kmonth,xtksl,lags,title="")



# Plot Individual points
for aa in range(nlatr):
    for oo in range(nlonr):
        for ii in range(3):
            plotvar = plot_ds[ii].isel(lon=oo,lat=aa)
            ax.plot(lags6,plotvar,alpha=0.01,label="",color=pdcolor[ii],zorder=-1)
        
# Now plot the regional average
acfmean = []
for ii in range(3):
    plotvar = plot_ds[ii].mean(('lon','lat'))
    ax.plot(lags6,plotvar,alpha=1,label=pdnames[ii] + ", Region Avg.",color=pdcolor[ii],zorder=9)
    acfmean.append(plotvar)
ax.legend(fontsize=12)

ax.legend(fontsize=12,framealpha=0)
ax.tick_params(axis='both', which='major', labelsize=12)


savename = "%sSM_v_CESM_SST_ACF_EnsAvg_STGBox.png" % (figpath,)
plt.savefig(savename,dpi=150,bbox_inches='tight',transparent=True)


#%% SSS comparison over region

xtksl   = np.arange(0,37,3)
lags3   = np.arange(0,37,1)
lags6   = ld['lags']


plot_ds = [dsregsm[2],dsregsm[3],dsregc[1]] # 
pdcolor = ["orange","magenta","k"]
pdnames = ["Stochastic Model (entrain)","Stochastic Model (non-entraining)","CESM1"]


fig,ax = plt.subplots(1,1,figsize=(8,3.5),constrained_layout=True)
ax,_   = viz.init_acplot(kmonth,xtksl,lags,title="")



# Plot Individual points
for aa in range(nlatr):
    for oo in range(nlonr):
        for ii in range(3):
            plotvar = plot_ds[ii].isel(lon=oo,lat=aa)
            ax.plot(lags6,plotvar,alpha=0.01,label="",color=pdcolor[ii],zorder=-1)
        
# Now plot the regional average
acfmean = []
for ii in range(3):
    plotvar = plot_ds[ii].mean(('lon','lat'))
    ax.plot(lags6,plotvar,alpha=1,label=pdnames[ii] + ", Region Avg.",color=pdcolor[ii],zorder=9)
    acfmean.append(plotvar)
#ax.legend(fontsize=12)
#ax.legend(fontsize=12,framealpha=0)
ax.tick_params(axis='both', which='major', labelsize=12)


savename = "%sSM_v_CESM_SSS_ACF_EnsAvg_STGBox.png" % (figpath,)
plt.savefig(savename,dpi=150,bbox_inches='tight',transparent=True)




