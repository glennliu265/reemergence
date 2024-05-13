#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Visualize Ice Edge/Fraction
Also visualize vertical profiles of SST/SSS at the ice edge, over
target points

Copied upper section from visualize_rei_acf.py


Created on Fri May 10 10:26:45 2024

@author: gliu

"""


import xarray as xr
import numpy as np
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import cartopy.crs as ccrs

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

#%% Load some information

# Indicate files containing ACFs
cesm_name   = "CESM1_1920to2005_%sACF_lag00to60_ALL_ensALL.nc"
vnames      = ["SST","SSS"]
#%% Load ACFs and REI

acfs_byvar  = []
rei_byvar   = []
for vv in range(2):
    ds = xr.open_dataset(procpath + cesm_name % vnames[vv]).acf.squeeze()
    acfs_byvar.append(ds)
    
    dsrei = xr.open_dataset("%s%s_CESM/Metrics/REI_Pointwise.nc" % (output_path,vnames[vv])).rei.load()
    rei_byvar.append(dsrei)
    
#%% Load mixed layer depth

ds_h    = xr.open_dataset(input_path + "mld/CESM1_HTR_FULL_HMXL_NAtl.nc").h.load()
#id_hmax = np.argmax(ds_h.mean(0),0)

cints_mld = np.arange(500,2100,100)
#%% Load BSF and Ice Mask (copied from compare_detrainment_damping)

bsf      = dl.load_bsf()

# Load Land Ice Mask
icemask  = xr.open_dataset(input_path + "masks/CESM1LE_HTR_limask_pacificmask_enssum_lon-90to20_lat0to90.nc")

# Resize
#bsf,icemask,_    = proc.resize_ds([bsf,icemask,acfs_in_rsz[0]])
bsf_savg = proc.calc_savg_mon(bsf)

#
mask = icemask.MASK.squeeze()
mask_plot = xr.where(np.isnan(mask),0,mask)#mask.copy()

mask_apply = icemask.MASK.squeeze().values
#mask_plot[np.isnan(mask)] = 0


#%% Indicate Plotting Parameters (taken from visualize_rem_cmip6)


bboxplot                    = [-80,0,10,65]
mpl.rcParams['font.family'] = 'Avenir'
mons3                       = proc.get_monstr(nletters=3)
fsz_tick                    = 18
fsz_axis                    = 14
fsz_title                   = 16
rhocrit                     = proc.ttest_rho(0.05,2,86)

proj                        = ccrs.PlateCarree()
#%% Load the ice fraction

# Ice Fraction
icenc = "%sCESM1LE_ICEFRAC_NAtl_19200101_20050101_bilinear.nc" % (rawpath)
iceds = xr.open_dataset(icenc).ICEFRAC.load()

# Compute mean seasonal cycle
icecycle = iceds.groupby('time.month').mean('time')

#%% Plot Re-emergence

rei_sss   = rei_byvar[1].isel(mon=[11,0,1,2]).mean('mon').mean('yr').mean('ens')
cints_rei = np.arange(0,0.55,0.05)
cints_rei_max = np.arange(0.40,0.62,0.02)

#%% Configure an ice edge plot

bboxice = [-70,-10,55,70]

for im in range(12):
    
    rei_sss   = rei_byvar[1].isel(mon=im).mean('yr').mean('ens')
    
    fig,ax,_ = viz.init_orthomap(1,1,bboxice,figsize=(12,4))
    ax       = viz.add_coast_grid(ax,bbox=bboxice,fill_color='lightgray')
    
    pv       =icecycle.isel(month=im).mean('ensemble')
    pcm      = ax.pcolormesh(pv.lon,pv.lat,pv,cmap='cmo.ice',transform=proj,zorder=-1)
    
    cb=viz.hcbar(pcm,ax=ax,fraction=0.045)
    cb.set_label("Ice Fraction (%)",fontsize=fsz_axis)
    ax.set_title("%s Ice Fraction" % (mons3[im]),fontsize=fsz_title)
    
    ax.contour(icemask.lon,icemask.lat,mask_plot,colors="red",linewidths=1.5,levels=[0,1],zorder=1,transform=proj,label="Ice Mask Edge")
    
    cl=ax.contour(rei_sss.lon,rei_sss.lat,rei_sss,transform=proj,levels=cints_rei)
    ax.clabel(cl)
    savename = "%sREI_SSS_mon%02i.png" % (figpath,im+1)
    plt.savefig(savename,dpi=150,bbox_inches="tight")

#%% Same as above but all in the same plot


fig,axs,_ = viz.init_orthomap(4,3,bboxice,figsize=(16,12))
for ii in range(12):
    ax        = axs.flatten()[ii]
    im        =  np.roll(np.arange(12),1)[ii]
    
    rei_sss   = rei_byvar[1].isel(mon=im).mean('yr').mean('ens')
    
    
    ax       = viz.add_coast_grid(ax,bbox=bboxice,fill_color='lightgray')
    
    pv        = icecycle.isel(month=im).mean('ensemble')
    pcm      = ax.pcolormesh(pv.lon,pv.lat,pv,cmap='cmo.ice',transform=proj,zorder=-1)
    
    ax.set_title("%s" % (mons3[im]),fontsize=fsz_title)
    
    ax.contour(icemask.lon,icemask.lat,mask_plot,colors="red",linewidths=1.5,levels=[0,1],zorder=1,transform=proj,label="Ice Mask Edge")
    
    cl=ax.contour(rei_sss.lon,rei_sss.lat,rei_sss,transform=proj,levels=cints_rei)
    ax.clabel(cl)
cb = viz.hcbar(pcm,ax=axs.flatten(),fraction=0.025,pad=0.02)
savename = "%sREI_SSS_monALL.png" % (figpath)
plt.savefig(savename,dpi=150,bbox_inches="tight")

#%% Visualize the seasonal cycle of the mixed layer depth in these regions as well

fig,axs,_ = viz.init_orthomap(4,3,bboxice,figsize=(16,12))
for ii in range(12):
    ax       = axs.flatten()[ii]
    im       =  np.roll(np.arange(12),1)[ii]
    
    rei_sss  = rei_byvar[1].isel(mon=im).mean('yr').mean('ens')
    
    ax       = viz.add_coast_grid(ax,bbox=bboxice,fill_color='lightgray')
    
    # Plot Ice Concentration
    pv       = icecycle.isel(month=im).mean('ensemble')
    pcm      = ax.pcolormesh(pv.lon,pv.lat,pv,cmap='cmo.ice',transform=proj,zorder=-1)
    
    ax.set_title("%s" % (mons3[im]),fontsize=fsz_title)
    
    ax.contour(icemask.lon,icemask.lat,mask_plot,colors="red",linewidths=1.5,levels=[0,1],zorder=1,transform=proj,label="Ice Mask Edge")
    
    plotcl  = ds_h.isel(mon=im).mean('ens')
    cl      = ax.contour(plotcl.lon,plotcl.lat,plotcl,linewidths=0.75,
                         transform=proj,levels=cints_mld,colors="skyblue")
    ax.clabel(cl)
cb = viz.hcbar(pcm,ax=axs.flatten(),fraction=0.025,pad=0.02)
savename = "%sMLD_SSS_monALL.png" % (figpath)
plt.savefig(savename,dpi=150,bbox_inches="tight")

#%% Plot Point selection

centerpoints = ([-37,62],[-55,59],)

sel_mons  = [1,2]

fig,axs,_ = viz.init_orthomap(1,2,bboxice,figsize=(12,6),constrained_layout=True)

for ii in range(2):
    ax       = axs[ii]
    im       = sel_mons[ii]
    rei_sss  = rei_byvar[1].isel(mon=im).mean('yr').mean('ens')
    
    ax       = viz.add_coast_grid(ax,bbox=bboxice,fill_color='lightgray')
    
    # Plot Ice Concentration
    pv       = icecycle.isel(month=im).mean('ensemble')
    pcm      = ax.pcolormesh(pv.lon,pv.lat,pv,cmap='cmo.ice',transform=proj,zorder=-1)
    
    # Plot RE Index
    cl=ax.contour(rei_sss.lon,rei_sss.lat,rei_sss,transform=proj,levels=cints_rei,linewidths=0.9,cmap='cmo.deep_r')
    ax.clabel(cl)
    
    # Pot ice Edge
    ax.contour(icemask.lon,icemask.lat,mask_plot,colors="red",linewidths=1.5,linestyles='dotted',
               levels=[0,1],zorder=1,transform=proj,label="Ice Mask Edge")

    
    # Plot the points
    cp = centerpoints[ii]
    locfn,loctitle=proc.make_locstring(cp[0],cp[1])
    ax.plot(cp[0],cp[1],transform=proj,marker='o',markersize=5,color='yellow')
    ax.plot(cp[0]+3,cp[1],transform=proj,marker='o',markersize=3,color='yellow')
    ax.plot(cp[0]-3,cp[1],transform=proj,marker='o',markersize=3,color='yellow')
    ax.plot(cp[0],cp[1]+3,transform=proj,marker='o',markersize=3,color='yellow')
    ax.plot(cp[0],cp[1]-3,transform=proj,marker='o',markersize=3,color='yellow')

    ax.set_title("%s - %s" % (mons3[im],loctitle),fontsize=fsz_title)
cb = viz.hcbar(pcm,ax=axs.flatten(),fraction=0.025,pad=0.02)
savename = "%sProfile_Analysis_locator.png" % (figpath)
plt.savefig(savename,dpi=150,bbox_inches="tight")

#%% Pull in profile data

profpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/ptdata/profile_analysis/"
ncnames  = ["IrmingerEns01_CrossPoints.nc","LabradorEns01_CrossPoints.nc"]
cpnames  = ["Irminger","Labrador"]

vv = 0

ds_savgs = []
for vv in range(2):

    ds      = xr.open_dataset(profpath + ncnames[vv])
    ds_savg = ds.groupby('time.month').mean('time').sel(z_t=slice(0*100,1000*100))
    ds_savgs.append(ds_savg)
    
    

mon = np.arange(1,13,1)
z   = ds_savg.z_t


#%%



fig,axs = plt.subplots(3,3,constrained_layout=True,figsize=(4,16))

# North Grid
ax = axs[0,1]
dsplot = ds_savg.sel(dir="N").SALT.T
pcm = ax.pcolormesh(mon,z,dsplot)


# Center Grid
ax = axs[1,1]
dsplot = ds_savg.sel(dir="Center").SALT.T
pcm = ax.pcolormesh(mon,z,dsplot)

# West
ax = axs[1,0]
dsplot = ds_savg.sel(dir="W").SALT.T
pcm = ax.pcolormesh(mon,z,dsplot)


# East
ax = axs[1,2]
dsplot = ds_savg.sel(dir="E").SALT.T
pcm = ax.pcolormesh(mon,z,dsplot)

# South
ax = axs[2,1]
dsplot = ds_savg.sel(dir="S").SALT.T
pcm = ax.pcolormesh(mon,z,dsplot)

#%% Just Plot Profiles 1 Direction at a time

# <|><|><|><|><|><|><|><|><|><|><|><|><|><|><|><|><|><|><|><|><|>
# <|><|><|><|><|><|><|><|><|><|><|><|><|><|><|><|><|><|><|><|><|>
# <|><|><|><|><|><|><|><|><|><|><|><|><|><|><|><|><|><|><|><|><|>
# <|><|><|><|><|><|><|><|><|><|><|><|><|><|><|><|><|><|><|><|><|>
# <|><|><|><|><|><|><|><|><|><|><|><|><|><|><|><|><|><|><|><|><|>
# <|><|><|><|><|><|><|><|><|><|><|><|><|><|><|><|><|><|><|><|><|>
cc   = 0
xtks = np.arange(1,13,1)

if cc == 0:
    cints_salt = np.arange(34,36,0.1)
    cints_temp = np.arange(0,10,0.4)
elif cc == 1:
    cints_salt = np.arange(34,35.6,0.1)
    cints_temp = np.arange(0,10,0.4)

di       = 'Center'
dirnames = ds_savgs[0].dir.values

for di in dirnames:
    
    
    
    fig,ax  = plt.subplots(1,1,constrained_layout=True,figsize=(8,6))
    
    # Plot the Salinity
    pv      = ds_savgs[cc].sel(dir=di).SALT.T
    pcm     = ax.contourf(mon,z/100,pv,cmap='cmo.haline',levels=cints_salt,extend='both')
    
    # Plot the Temp
    pv      = ds_savgs[cc].sel(dir=di).TEMP.T
    cl      = ax.contour(mon,z/100,pv,cmap='cmo.thermal',levels=cints_temp,extend='both')
    ax.clabel(cl)
    
    # # Plot the Temp
    # pv2      = ds_savg.sel(dids_r=di).TEMP.T
    # pcm2     = ax.contourf(mon,z/100,pv,cmap='cmo.ter')
    
    ax.set_ylim([0,800])
    plt.gca().invert_yaxis()
    fig.colorbar(pcm,ax=ax)
    
    # Plot Mean Mixed Layer Depth
    plotmld = ds_h.isel(ens=0).sel(lon=pv.TLONG-360,lat=pv.TLAT,method='nearest')
    ax.plot(mon,plotmld,ls='dashed',color='violet',label="HMXL")
    ax.set_ylabel("Depth (meters)")
    ax.set_xticks(xtks,labels=mons3)
    ax.grid(True,ls='dotted',c='w',alpha=0.5)
    
    # Plot the ice fraction
    plotice = icecycle.isel(ensemble=0).sel(lon=pv.TLONG-360,lat=pv.TLAT,method='nearest')
    ax2 = ax.twinx()
    ax2.plot(mon,plotice,label="Ice Fraction",alpha=1,color="k",marker="x")
    ax2.set_ylim([0,0.5])
    ax2.set_ylabel("Ice Fraction")
    
    ax2.legend()
    
    ax.set_title("%s (%s) : Lon %.2f, Lat %.2f" % (cpnames[cc],di,pv.TLONG,pv.TLAT))
    savename = "%sVertical_Profiles_Scycle_%s_%s.png" % (figpath,cpnames[cc],di)
    
    plt.savefig(savename,dpi=150,bbox_inches='tight')
    
    
    
#%%

#%%




