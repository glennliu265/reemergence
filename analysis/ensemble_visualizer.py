#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Make some animations for the thesis

Created on Thu Nov 21 11:16:18 2024

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

import matplotlib.ticker as mticker


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

# Set Mode

darkmode = True
if darkmode:
    dfcol = "w"
    bgcol = np.array([15,15,15])/256
    transparent = True
    plt.style.use('dark_background')
    mpl.rcParams['font.family']     = 'Avenir'
else:
    dfcol = "k"
    bgcol = "w"
    transparent = False
    plt.style.use('default')


bboxplot                        = [-80,0,20,65]
mpl.rcParams['font.family']     = 'Avenir'
mons3                           = proc.get_monstr(nletters=3)

fsz_tick                        = 18
fsz_axis                        = 20
fsz_title                       = 32

rhocrit                         = proc.ttest_rho(0.05,2,86)
proj                            = ccrs.PlateCarree()

# Get Region Info
regionset = "SSSCSU"
rdict                       = rparams.region_sets[regionset]
regions                     = rdict['regions']
bboxes                      = rdict['bboxes']
rcols                       = rdict['rcols']
rsty                        = rdict['rsty']
regions_long                = rdict['regions_long']
nregs                       = len(bboxes)

regions_long = ('Sargasso Sea', 'N. Atl. Current',  'Irminger Sea')

#%% Load Land Ice Mask
#bboxplot   = bbplot
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

#%% Load 


# first ensemble member
amvbbox      = [-80,0,0,60]

ncname       = "b.e11.B20TRC5CNBDRD.f09_g16.001.cam.h0.TS.185001-200512.nc"
ds_ens1      = xr.open_dataset(rawpath + ncname).load()

#% Load in SST (All Ensemble)
ds_sst         = xr.open_dataset(rawpath + 'CESM1LE_SST_NAtl_19200101_20050101_bilinear.nc').SST.load()
ds_scycle      = ds_sst.groupby('time.month').mean('time')
ds_ensavg      = ds_sst.mean('ensemble')

#%% Preprocess

# Full Ens 1 ---------
# Edit formatting. crop, mask
ds_ens1               = proc.format_ds(ds_ens1.TS)
ds_ens1               = proc.fix_febstart(ds_ens1)
ds_ens1_natl          = proc.sel_region_xr(ds_ens1,amvbbox)


# Deseason
dsa_ens1_natl         = ds_ens1_natl.groupby('time.month') - ds_scycle.isel(ensemble=0)
dsa_ens1_natl_masked  = dsa_ens1_natl * mask

# Take Area Average
ens1_nasst   = proc.area_avg_cosweight(dsa_ens1_natl_masked)
time1850     = proc.cftime2str(ds_ens1.time.data)
timeyr       = [t[:4] for t in time1850]
timemon      = [t[:7] for t in time1850]
itime1850    = np.arange(len(time1850))

#%%

# Select the region
dsa_sst        = proc.xrdeseason(ds_sst)
dsa_sst_masked = dsa_sst * mask

dsa_sst_masked_natl = proc.sel_region_xr(dsa_sst_masked,amvbbox)

ensall_nasst   = proc.area_avg_cosweight(dsa_sst_masked_natl)

time1920       = proc.cftime2str(ds_sst.time.data)
itime1920      = itime1850[-1032:]

#%% Check the plot

fig,ax = plt.subplots(1,1,constrained_layout=True,figsize=(12,4.5))

ax.plot(itime1850,ens1_nasst,label="Raw Ens1")
ax.plot(itime1920,ensall_nasst.isel(ensemble=0),label="Crop Ens 1")
ax.legend()

ax.set_xticks(itime1850[::120],labels=timeyr[::120])

#ax.set_xlim([750,1000])


#%% Plot the large ensemble timeseries (full, static)

# Copied format from predict_amv/Analysis/plot_AMV_hadISST.py
fsz_title = 22
fsz_axis  = 18
fsz_ticks = 16

fig,ax = plt.subplots(1,1,constrained_layout=True,figsize=(12,4))

ax.plot(itime1850,ens1_nasst,label="Ensemble 1",color='palegoldenrod',alpha=1,zorder=-1,lw=2)

for e in np.arange(1,42):
    ax.plot(itime1920,ensall_nasst.isel(ensemble=e),
            label="",alpha=0.25,color='lightgray')

#ax.plot(itime1920,ensall_nasst.isel(ensemble=0),label="Crop Ens 1")
#ax.legend()

# Adjust X-axis
ax.set_xticks(itime1850[::12],labels=timeyr[::12],fontsize=fsz_ticks)
ax.set_xlim([0,1872])
ax.set_xlabel("Time (Year)",fontsize=fsz_axis)
ax.xaxis.set_minor_locator(mticker.AutoMinorLocator())

# Adjust Y-axis
ax.set_ylim([-0.75,0.75])
ax.set_yticks(np.arange(-0.5,.75,0.25))
ax.tick_params(labelsize=fsz_ticks)
ax.set_ylabel("Sea Surface Temperature [$\degree$C]",fontsize=fsz_axis)
ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())

#viz.add_ticks(ax=ax,facecolor='none')

# Additional Formatting
ax.spines[['right', 'top']].set_visible(False)

# Add horizontal zero line
#ax.axhline([0],ls='dashed',lw=0.75,c=dfcol)
ax.set_xlim([840-12,840+60])
ax.axvline([840],color='w',lw=0.75,ls='dotted')

ax.set_facecolor(bgcol)
fig.set_facecolor(bgcol)

#%% Animate it...

tstart = 840 - 12 # 1919
tend   = 840 + 60 # 1925


tstops = np.arange(tstart,tend+1)
nframes = len(tstops)
for ff in range(nframes):
    tt     = tstops[ff]
    tt1920 = tt - 840
    
    fig,ax = plt.subplots(1,1,constrained_layout=True,figsize=(12,4))
    
    # Set Ensemble 1
    iplot = itime1850[:tt]
    plotvar = ens1_nasst.data[:tt]
    
    ax.plot(iplot,plotvar,label="Ensemble 1",color='palegoldenrod',alpha=1,zorder=-1,lw=2)
    ax.plot(iplot[-1],plotvar[-1],alpha=1,markersize=10,zorder=-1,
            color='palegoldenrod',marker="o",markerfacecolor="none")
    
    if tt1920 >=1:
        for e in np.arange(1,42):
            iplot   = itime1920[:tt1920]
            plotvar = ensall_nasst.isel(ensemble=e).data[:tt1920]
            
            ax.plot(iplot,plotvar,
                    label="",alpha=0.25,color='lightgray')
            
            ax.plot(iplot[-1],plotvar[-1],alpha=0.25,markersize=5,
                    color='lightgray',marker="o",markerfacecolor="none")
    
    #ax.plot(itime1920,ensall_nasst.isel(ensemble=0),label="Crop Ens 1")
    #ax.legend()
    
    # Adjust X-axis
    ax.set_xticks(itime1850[::12],labels=timeyr[::12],fontsize=fsz_ticks)
    ax.set_xlim([0,1872])
    ax.set_xlabel("Time (Year)",fontsize=fsz_axis)
    ax.xaxis.set_minor_locator(mticker.AutoMinorLocator())
    
    # Adjust Y-axis
    ax.set_ylim([-0.75,0.75])
    ax.set_yticks(np.arange(-0.5,.75,0.25))
    ax.tick_params(labelsize=fsz_ticks)
    ax.set_ylabel("Sea Surface Temperature [$\degree$C]",fontsize=fsz_axis)
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())
    
    #viz.add_ticks(ax=ax,facecolor='none')
    
    # Additional Formatting
    ax.spines[['right', 'top']].set_visible(False)
    
    # Add horizontal zero line
    #ax.axhline([0],ls='dashed',lw=0.75,c=dfcol)
    ax.set_xlim([840-12,840+60])
    
    # Add line at 1920
    if tt >= 840:
        ax.axvline([840],color='w',lw=0.75,ls='dotted')
        
    # Set Facecolor
    ax.set_facecolor(bgcol)
    fig.set_facecolor(bgcol)
    #plt.figure(bgcol)
        
    
    savename = "%sCESM1_NASST_ENSwise_1920_frame%03i.png" % (figpath,ff)
    plt.savefig(savename,dpi=150,bbox_inches='tight',facecolor=bgcol)
    

    if ff > 2:
        continue

#%% Check Differences

ens1full = dsa_ens1_natl_masked.sel(time='1920-01-01')
ens1crop = dsa_sst_masked.isel(ensemble=0,time=0)
diff = ens1full - ens1crop
diff.plot()

#%% Make 


#%% Make a stochastic model simulation
ntime   = 1000
lbd     = 0.95
sigma   = 0.1
noisets =np.random.normal(0,1*sigma,ntime,)

lbd_b   = 0.99

rednoise = np.zeros(ntime)
rednoise_b = rednoise.copy()
for t in np.arange(1,ntime):
    
    rednoise[t] = rednoise[t-1] * lbd + noisets[t]
    rednoise_b[t] = rednoise_b[t-1] * lbd_b + noisets[t]


#%% Static Noise Example

tt   = 12
trange = np.arange(ntime)

#atmforce_color = np.array([141,0,255])/256 # Dark Purple
atmforce_color = np.array([193,67,254])/256 # Dark Purple
ocnresp_color  = np.array([255,80,122])/256

ttplot = trange[1::10]
nframes = len(ttplot)
for ff in range(nframes):

    print(ff)
    
    tend   = ttplot[ff]
    
    fig,axs = plt.subplots(2,1,constrained_layout=True,figsize=(12,5.5))
    
    # Plot Forcing
    plotvar = noisets[:tend]
    plottime = trange[:tend]
    axs[0].plot(plottime,plotvar,lw=0.75,c=atmforce_color,label="White Noise")
    axs[0].plot(plottime[-1],plotvar[-1],c=atmforce_color,label="",marker="o",markerfacecolor="none",markersize=5)
    axs[0].set_ylabel("Forcing [$\degree$C/sec]",fontsize=fsz_ticks)
    
    # Plot 
    plotvar = rednoise[:tend]
    plottime = trange[:tend]
    axs[1].plot(plottime,plotvar,c=ocnresp_color,lw=1,label="Stochastic Model Output")
    axs[1].plot(plottime[-1],plotvar[-1],c=ocnresp_color,label="",marker="o",markerfacecolor="none",markersize=5)
    #ax.plot(rednoise_b,c='cornflowerblue',lw=1,label="$\lambda$=%.2f" % lbd_b)
    
    for ax in axs:
        #ax.legend(fontsize=fsz_tick-4)
        ax.set_xlim([0,1000])
        ax.set_ylim([-1.5,1.5])
        ax.tick_params(labelsize=fsz_tick)
        #ax.set_xlabel("Simulation Timestep",fontsize=fsz_axis)
        #ax.set_ylabel("Value",fontsize=fsz_axis)
        ax.spines[['right', 'top']].set_visible(False)
        ax.set_facecolor(bgcol)
        
    axs[1].set_xlabel("Timestep",fontsize=fsz_ticks)
    axs[1].set_ylabel("SST [$\degree$C]",fontsize=fsz_ticks)
    fig.set_facecolor(bgcol)
    
    savename = "%sStochastic_Model_Example_frame%03i.png" % (figpath,ff)
    plt.savefig(savename,dpi=150,bbox_inches='tight',facecolor=bgcol)
    
    
    




