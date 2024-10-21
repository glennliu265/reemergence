#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

Compare currents computed by the followingscripts

    - calc_ekman_advection_htr.py (instantaneous Ekman velocities)
    - calc_geostrophic_current.py (instantaneous Geostrophic velocities)
    - regrid_POP_1level.py (regridded UVEl and VVEL)

Created on Wed Jul 31 09:43:17 2024

@author: gliu
"""

import xarray as xr
import numpy as np
import matplotlib as mpl
import scipy as sp

import matplotlib.pyplot as plt
import sys
import glob
import os

import tqdm
import time

import cartopy.crs as ccrs

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
pathdict    = rparams.machine_paths[machine]

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

#%% Plotting Information

bboxplot                    = [-80,0,20,65]
mpl.rcParams['font.family'] = 'Avenir'
mons3                       = proc.get_monstr(nletters=3)
fsz_tick                    = 18
fsz_axis                    = 22
fsz_title                   = 28
proj                        = ccrs.PlateCarree()

#%% Load the currents

# Load the geostrophic currents
st          = time.time()
nc_ugeo     = "CESM1LE_ugeo_NAtl_19200101_20050101_bilinear.nc"
path_ugeo   = rawpath
ds_ugeo     = xr.open_dataset(path_ugeo + nc_ugeo).load()
print("Loaded ugeo in %.2fs" % (time.time()-st))

# Load the ekman currents
st          = time.time()
nc_uek      = "CESM1LE_uek_NAtl_19200101_20050101_bilinear.nc"
ds_uek      = xr.open_dataset(path_ugeo + nc_uek).load()
print("Loaded uek in %.2fs" % (time.time()-st))

# Load total surface velocity
st          = time.time()
nc_uvel     = "UVEL_NATL_AllEns_regridNN.nc"
nc_vvel     = "VVEL_NATL_AllEns_regridNN.nc"
ds_uvel     = xr.open_dataset(path_ugeo + nc_uvel).load()
ds_vvel     = xr.open_dataset(path_ugeo + nc_vvel).load()
print("Loaded uvels in %.2fs" % (time.time()-st))

#%% Calculate the modulus and do additional setup

# Compute u.v vel, convert to m/s
ds_uvel     = xr.merge([ds_uvel.UVEL/100,ds_vvel.VVEL/100])

# Convert u_ek 
ds_uek['u_ek'] = ds_uek.u_ek * 100 * -1
ds_uek['v_ek'] = ds_uek.v_ek * 100 * -1

#ds_uek      = ds_uek * 100 # Multiply by 100

# Compute amplitudes
ugeo_mod    = (ds_ugeo.ug **2 + ds_ugeo.vg ** 2) ** 0.5
uek_mod     = (ds_uek.u_ek**2 + ds_uek.v_ek ** 2) ** 0.5
uvel_mod    = (ds_uvel.UVEL**2 + ds_uvel.VVEL **2) ** 0.5

#%% Load Land/Ice Mask

# Load Land Ice Mask
icemask     = xr.open_dataset(input_path + "masks/CESM1LE_HTR_limask_pacificmask_enssum_lon-90to20_lat0to90.nc")

mask        = icemask.MASK.squeeze()
mask_plot   = xr.where(np.isnan(mask),0,mask)#mask.copy()

mask_apply  = icemask.MASK.squeeze().values
#mask_plot[np.isnan(mask)] = 0

# Load Gulf Stream
ds_gs           = dl.load_gs()
ds_gs           = ds_gs.sel(lon=slice(-90,-50))
ds_gs2          = dl.load_gs(load_u2=True)

#%% Make some convenience functions

def plot_vel(plotu,plotv,qint,ax,proj=ccrs.PlateCarree(),scale=1,c='k'):
    lon     = plotu.lon.data
    lat     = plotu.lat.data
    qv      = ax.quiver(lon[::qint],lat[::qint],
                        plotu.data[::qint,::qint],plotv.data[::qint,::qint],
                        transform=proj,scale=scale,color=c)
    return qv
    
    

#%% Plot the currents at 1 instant in time

t       = 0
e       = 0
qint    = 2

# Initialize figure
fig,ax,_    = viz.init_orthomap(1,1,bboxplot,figsize=(24,14.5),)
ax          = viz.add_coast_grid(ax,bbox=bboxplot,fill_color="lightgray",fontsize=24)

# Plot each of the currents ----

# Total Velocite
plotu   = ds_uvel.UVEL.isel(ens=e,time=t) * mask
plotv   = ds_uvel.VVEL.isel(ens=e,time=t) * mask
scale   = 5
qv_uvel = plot_vel(plotu,plotv,qint,ax,scale=scale,c='k')

# Uek
plotu   = ds_uek.u_ek.isel(ens=e,time=t) * mask
plotv   = ds_uek.v_ek.isel(ens=e,time=t) * mask
scale   = 0.8#0.005
qv_uek  = plot_vel(plotu,plotv,qint,ax,scale=scale,c='blue')

# Ugeo
plotu   = ds_ugeo.ug.isel(ens=e,time=t) * mask
plotv   = ds_ugeo.vg.isel(ens=e,time=t) * mask
scale   = 5
qv_uek  = plot_vel(plotu,plotv,qint,ax,scale=scale,c='green')

#%% Plot the amplitude at a time instance


#dtmon= 3600*24*30

fig,axs,_    = viz.init_orthomap(1,3,bboxplot,figsize=(24,7),)

for ax in axs:
    ax = viz.add_coast_grid(ax,bbox=bboxplot,fill_color="lightgray",fontsize=24)

# Total Velocite --------------------------------------------------------------
ax      = axs[0]
ax.set_title("Total Current [$m/s$]",fontsize=fsz_title)
plotu   = ds_uvel.UVEL.isel(ens=e,time=t) * mask
plotv   = ds_uvel.VVEL.isel(ens=e,time=t) * mask
scale   = 5
qv_uvel = plot_vel(plotu,plotv,qint,ax,scale=scale,c='k')
qk_uvel  = ax.quiverkey(qv_uvel,.9,1.,.5,r"0.5 $\frac{m}{s}$",fontproperties=dict(size=fsz_tick))


# Plot Magnitude
plotvar = uvel_mod.isel(ens=e,time=t) * mask
pcm     = ax.pcolormesh(plotvar.lon,plotvar.lat,plotvar,cmap='cmo.tempo',transform=proj,zorder=-1)
cb      = viz.hcbar(pcm,ax=ax,fraction=0.045)
cb.ax.tick_params(labelsize=fsz_tick)

# Geostrophic Advection -------------------------------------------------------
ax      = axs[1]
ax.set_title("Geostrophic Current [$m/s$]",fontsize=fsz_title)
plotu   = ds_ugeo.ug.isel(ens=e,time=t) * mask
plotv   = ds_ugeo.vg.isel(ens=e,time=t) * mask
scale   = 5
qv_ugeo  = plot_vel(plotu,plotv,qint,ax,scale=scale,c='green')
qk_ugeo  = ax.quiverkey(qv_ugeo,.9,1.,.5,r"0.5 $\frac{m}{s}$",fontproperties=dict(size=fsz_tick))

# Plot Magnitude
plotvar = ugeo_mod.isel(ens=e,time=t) * mask
pcm     = ax.pcolormesh(plotvar.lon,plotvar.lat,plotvar,cmap='cmo.speed',transform=proj,zorder=-1)
cb      = viz.hcbar(pcm,ax=ax,fraction=0.045)
cb.ax.tick_params(labelsize=fsz_tick)

# Ekman -----------------------------------------------------------------------
ax      = axs[2]
ax.set_title("Ekman Current [$m/s$]",fontsize=fsz_title)
plotu   = ds_uek.u_ek.isel(ens=e,time=t) * mask
plotv   = ds_uek.v_ek.isel(ens=e,time=t) * mask
scale   = 1
qv_uek  = plot_vel(plotu,plotv,qint,ax,scale=scale,c='blue')
qk_uek  = ax.quiverkey(qv_uek,.9,1.,.1,r"0.1 $\frac{m}{s}$",fontproperties=dict(size=fsz_tick))

# Plot Magnitude
plotvar = uek_mod.isel(ens=e,time=t) * mask #* dtmon
pcm     = ax.pcolormesh(plotvar.lon,plotvar.lat,plotvar,cmap='cmo.ice_r',transform=proj,zorder=-1)
cb      = viz.hcbar(pcm,ax=ax,fraction=0.045)
cb.ax.tick_params(labelsize=fsz_tick)

plt.suptitle("Currents, Ens. %02i, t=%s" % (e+1,proc.noleap_tostr(ds_ugeo.time.isel(time=t))),
             fontsize=fsz_title)

savename = "%sCurrent_Comparison_Ens%02i_t%03i.png" % (figpath,e+1,t)
plt.savefig(savename,dpi=150,bbox_inches='tight')

#  ----------------------------------------------------------------------------
#%% Plot Time Mean (same structure essentially as above)
#  ----------------------------------------------------------------------------


#dtmon= 3600*24*30

fig,axs,_    = viz.init_orthomap(1,3,bboxplot,figsize=(24,7),)

for ax in axs:
    ax = viz.add_coast_grid(ax,bbox=bboxplot,fill_color="lightgray",fontsize=24)
    
    # Plot Extra Things --------
    # # Plot Gulf Stream Position

    ax.plot(ds_gs2.lon.mean('mon'),ds_gs2.lat.mean('mon'),transform=proj,lw=2.5,c='red',ls='dashdot')

    # Plot Ice Edge
    ax.contour(icemask.lon,icemask.lat,mask_plot,colors="cyan",linewidths=2,
               transform=proj,levels=[0,1],zorder=-1)

# Total Velocite --------------------------------------------------------------
ax      = axs[0]
ax.set_title("Total Current [$m/s$]",fontsize=fsz_title)
plotu   = ds_uvel.UVEL.mean('time').mean('ens') * mask
plotv   = ds_uvel.VVEL.mean('time').mean('ens') * mask
scale   = 5
qv_uvel = plot_vel(plotu,plotv,qint,ax,scale=scale,c='k')
qk_uvel  = ax.quiverkey(qv_uvel,.9,1.,.5,r"0.5 $\frac{m}{s}$",fontproperties=dict(size=fsz_tick))


# Plot Magnitude
plotvar = uvel_mod.mean('time').mean('ens') * mask
pcm     = ax.pcolormesh(plotvar.lon,plotvar.lat,plotvar,cmap='cmo.tempo',transform=proj,zorder=-1,vmin=0,vmax=0.3)
cb      = viz.hcbar(pcm,ax=ax,fraction=0.045)
cb.ax.tick_params(labelsize=fsz_tick)

# Geostrophic Advection -------------------------------------------------------
ax      = axs[1]
ax.set_title("Geostrophic Current [$m/s$]",fontsize=fsz_title)
plotu   = ds_ugeo.ug.mean('time').mean('ens') * mask
plotv   = ds_ugeo.vg.mean('time').mean('ens') * mask
scale   = 5
qv_ugeo  = plot_vel(plotu,plotv,qint,ax,scale=scale,c='green')
qk_ugeo  = ax.quiverkey(qv_ugeo,.9,1.,.5,r"0.5 $\frac{m}{s}$",fontproperties=dict(size=fsz_tick))

# Plot Magnitude
plotvar = ugeo_mod.mean('time').mean('ens') * mask
pcm     = ax.pcolormesh(plotvar.lon,plotvar.lat,plotvar,cmap='cmo.speed',transform=proj,zorder=-1,vmin=0,vmax=0.3)
cb      = viz.hcbar(pcm,ax=ax,fraction=0.045)
cb.ax.tick_params(labelsize=fsz_tick)

# Ekman -----------------------------------------------------------------------
ax      = axs[2]
ax.set_title("Ekman Current [$m/s$]",fontsize=fsz_title)
plotu   = ds_uek.u_ek.mean('time').mean('ens') * mask
plotv   = ds_uek.v_ek.mean('time').mean('ens') * mask
scale   = 1
qv_uek  = plot_vel(plotu,plotv,qint,ax,scale=scale,c='blue')
qk_uek  = ax.quiverkey(qv_uek,.9,1.,.1,r"0.1 $\frac{m}{s}$",fontproperties=dict(size=fsz_tick))

# Plot Magnitude
plotvar = uek_mod.mean('time').mean('ens') * mask
pcm     = ax.pcolormesh(plotvar.lon,plotvar.lat,plotvar,cmap='cmo.ice_r',transform=proj,zorder=-1,vmin=0,vmax=0.1)
cb      = viz.hcbar(pcm,ax=ax,fraction=0.045)
cb.ax.tick_params(labelsize=fsz_tick)




plt.suptitle("Currents, Ens. Avg., Time Mean",
             fontsize=fsz_title)

savename = "%sCurrent_Comparison_EnsAvg_tmean.png" % (figpath)
plt.savefig(savename,dpi=150,bbox_inches='tight')

#%% Set up Arrays for looping

# Get the data and preprocess
ds_all   = [uvel_mod,ugeo_mod,uek_mod]
expnames = ["$u_{total}$","$u_{geo}$","$u_{ek}$"]
expcols  = ["k","green","blue"]
expcols_light = ["gray","limegreen","cornflowerblue"]

def preproc_ds(ds):
    dsa = proc.xrdeseason(ds)#ds - ds.groupby('time.month').mean('time')
    dsa = dsa - dsa.mean('ens')
    return dsa

#%% Look at the spectra at a point

lonf           = -30
latf           = 50

#bbox_sel       = [-40,-30,40,50] # Set to None to Do Point Analysis
#bbox_name      = "NAC"

#bbox_sel    =  [-40,-25,50,60] # Irminger
#bbox_name   = "IRM"

bbox_sel   = [-70,-55,35,40] # Sargasso Sea
bbox_name  = "SAR"

if bbox_sel is None:
    print("Selecting a point")
    locfn,loctitle = proc.make_locstring(lonf,latf)
    ds_pt       = [ds.sel(lon=lonf,lat=latf,method='nearest') for ds in ds_all]


else:
    print("Computing regional average")
    locfn,loctitle = proc.make_locstring_bbox(bbox_sel)
    
    ds_pt       = [proc.sel_region_xr(ds,bbox_sel).mean('lat').mean('lon') for ds in ds_all]
    
    locfn       = "%s_%s"   % (bbox_name,locfn)
    loctitle    = "%s (%s)" % (bbox_name,loctitle)
    
dsa_pt  = [preproc_ds(ds) for ds in ds_pt]
#arr_pt  = [ds.data.flatten() for ds in dsa_pt]
arr_pt  = [ds.data for ds in dsa_pt]

# Zero out 1 value
arr_pt[0][32,219] = 0
#[print(np.any(np.isnan(ds))) for ds in arr_pt]





#%% Compute the power spectra
nsmooth     = 2
pct         = 0.10
loopdim     = 0 # Ens
dtmon       = 3600*24*30

specdicts   = []
for ii in range(3):
    in_ts    = arr_pt[ii]
    specdict = scm.quick_spectrum(in_ts,nsmooth,pct,dt=dtmon,dim=loopdim,return_dict=True,make_arr=True)
    specdicts.append(specdict)

#%% Make the plot

xlims       = [1/(86*12*dtmon),1/(2*dtmon)]
xtks_per    = np.array([50,25,10,5,2,1,0.5]) # in Years
xtks_freq   = 1/(xtks_per * 12 * dtmon)

fig,ax      = plt.subplots(1,1,constrained_layout=True,figsize=(12,4.5))

# -----------------------------------------------------------------------------

for ii in range(3):
    
    specdict = specdicts[ii]
    
    ename       = expnames[ii]
    ecol        = expcols[ii]
    ecollight   = expcols_light[ii]
    
    evar        = arr_pt[ii].std(1).mean(0)
    lab         = "%s ($\sigma$=%.2e m/s)" % (ename,evar)
    
    # Plot each Ensemble
    for e in range(42):
        plotspec = specdict['specs'][e,:]
        plotfreq = specdict['freqs'][e,:]
        ax.loglog(plotfreq,plotspec,alpha=0.05,label="",c=ecollight)
     
    # PLot Ens Avg
    plotspec = specdict['specs'].mean(0)
    plotfreq = specdict['freqs'].mean(0)
    ax.loglog(plotfreq,plotspec,alpha=1,label=lab,color=ecol)
    
    # Plot AR(1) Null Hypothesis
    plotcc0 = specdict['CCs'][:,:,0].mean(0)
    plotcc1 = specdict['CCs'][:,:,1].mean(0)
    ax.loglog(plotfreq,plotcc1,alpha=1,label="",color=ecol,lw=0.75,ls='dashed')
    ax.loglog(plotfreq,plotcc0,alpha=1,label="",color=ecol,lw=0.75)
    #ax.loglog(plotfreq,plotspec,alpha=1,label="CESM1 Ens. Avg",color="navy")
    #ax.axvline([1/(2*dtmon)],label="NQ")

# Draw some lines
ax.axvline([1/(6*dtmon)],label="Semiannual",color='lightgray',ls='dotted')
ax.axvline([1/(12*dtmon)],label="Annual",color='gray',ls='dashed')
ax.axvline([1/(10*12*dtmon)],label="Decadal",color='dimgray',ls='dashdot')
ax.axvline([1/(50*12*dtmon)],label="50-yr",color='k',ls='solid')
ax.legend(ncol=3,fontsize=fsz_tick-6,loc='lower center')

# Label Frequency Axis
ax.set_xlabel("Frequency (Cycles/Sec)",fontsize=fsz_tick)
ax.set_xlim(xlims)

# Label y-axis
ax.set_ylabel("Power ([m/s]$^2$/cps)",fontsize=fsz_tick)

ax2 = ax.twiny()
ax2.set_xlim(xlims)
ax2.set_xscale('log')
ax2.set_xticks(xtks_freq,labels=xtks_per)
ax2.set_xlabel("Period (Years)",fontsize=fsz_tick)
ax.set_title("Power Spectra for Currents @ %s" % loctitle,fontsize=fsz_axis)

savename = "%sCESM1_HTR_EnsAvg_current_comparison_spectra_%s_nsmooth%04i.png" % (figpath,locfn,nsmooth)
plt.savefig(savename,dpi=200,bbox_inches='tight')

#%% Examine Log Ratio of the currents

ugeo_var = ugeo_mod.var('time')
uvel_var = uvel_mod.var('time')
uek_var  = uek_mod.var('time')

#%% Examine Percentages

fig,axs,_    = viz.init_orthomap(1,2,bboxplot,figsize=(16.5,7),)

for ax in axs:
    ax = viz.add_coast_grid(ax,bbox=bboxplot,fill_color="lightgray",fontsize=24)


    ax.plot(ds_gs2.lon.mean('mon'),ds_gs2.lat.mean('mon'),transform=proj,lw=2.5,c='red',ls='dashdot')

    # Plot Ice Edge
    ax.contour(icemask.lon,icemask.lat,mask_plot,colors="cyan",linewidths=2,
               transform=proj,levels=[0,1],zorder=-1)

    

ax      = axs[0]
ax.set_title(r"Log Ratio $\frac{\sigma^2(u_{geo})}{\sigma^2(u_{total})}$",fontsize=fsz_title)
plotvar = np.log( ugeo_var / uvel_var ).mean('ens') * mask
pcm     = ax.pcolormesh(plotvar.lon,plotvar.lat,plotvar,
                        transform=proj,cmap='cmo.balance',vmin=-4,vmax=4)

cb      = viz.hcbar(pcm,ax=ax,fraction=0.045)
cb.ax.tick_params(labelsize=fsz_tick)


ax      = axs[1]
ax.set_title(r"Log Ratio $\frac{\sigma^2(u_{ek})}{\sigma^2(u_{total})}$",fontsize=fsz_title)
plotvar = np.log( uek_var / uvel_var ).mean('ens') * mask
pcm     = ax.pcolormesh(plotvar.lon,plotvar.lat,plotvar,
                        transform=proj,cmap='cmo.balance',vmin=-4,vmax=4)

cb      = viz.hcbar(pcm,ax=ax,fraction=0.045)
cb.ax.tick_params(labelsize=fsz_tick)

plt.suptitle("Log Ratio of Total |U| Variance, Ens. Avg.",fontsize=fsz_title,y=1.075)

savename = "%sCESM1_HTR_EnsAvg_current_comparison_variance_log_ratio.png" % (figpath)
plt.savefig(savename,dpi=200,bbox_inches='tight')

#%% Try computing the coherence



#ugeo_pt = ugeo_mod.sel(lon=lonf,lat=latf,method='nearest') # [Ens x Time]
#uvel_pt = uvel_mod





#%%



#%%






#%%


# Load Mean SSH
ds_ssh          = dl.load_bsf(ensavg=False,ssh=True).mean('ens').SSH /100

# Load mean SST/SSS
ds_sss          = dl.load_monmean('SSS')
ds_sst          = dl.load_monmean('SST')

# # Load Centered Differences
# nc_gradT = "CESM1_HTR_FULL_Monthly_gradT2_SST.nc"
# nc_gradS = "CESM1_HTR_FULL_Monthly_gradT2_SSS.nc"
# ds_gradT = xr.open_dataset(rawpath + nc_gradT).load()
# ds_gradS = xr.open_dataset(rawpath + nc_gradS).load()





