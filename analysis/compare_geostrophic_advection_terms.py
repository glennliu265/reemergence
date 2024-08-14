#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Copied from compute_geostrophic_advection_term

Created on Wed Aug 14 13:23:31 2024

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

#%% Plotting Information

bboxplot                    = [-80,0,20,65]
mpl.rcParams['font.family'] = 'Avenir'
mons3                       = proc.get_monstr(nletters=3)
fsz_tick                    = 18
fsz_axis                    = 22
fsz_title                   = 28
proj                        = ccrs.PlateCarree()

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

# Load Mean SSH
ds_ssh          = dl.load_bsf(ensavg=False,ssh=True).mean('ens').SSH /100

# Load mean SST/SSS
ds_sss          = dl.load_monmean('SSS')
ds_sst          = dl.load_monmean('SST')

# Load Centered Differences
nc_gradT = "CESM1_HTR_FULL_Monthly_gradT2_SST.nc"
nc_gradS = "CESM1_HTR_FULL_Monthly_gradT2_SSS.nc"
ds_gradT = xr.open_dataset(rawpath + nc_gradT).load()
ds_gradS = xr.open_dataset(rawpath + nc_gradS).load()

# ================
#%% Load the files
# ================

# Load the geostrophic currents
st               = time.time()
nc_ugeo          = "CESM1LE_ugeo_NAtl_19200101_20050101_bilinear.nc"
path_ugeo        = rawpath
ds_ugeo          = xr.open_dataset(path_ugeo + nc_ugeo).load()
print("Loaded ugeo in %.2fs" % (time.time()-st))

# Compute monthly means
ugeo_monmean     = ds_ugeo.groupby('time.month').mean('time').mean('ens') # [lat x lon x month]
ugeo_mod_monmean = (ugeo_monmean.ug**2 + ugeo_monmean.vg**2)**0.5

#%% Compute the amplitude

ugeo_mod         = (ds_ugeo.ug ** 2 + ds_ugeo.vg ** 2)**0.5

#%% Make a plot, locate a point

im    = 1
qint  = 2
scale = 5
vlms  = [0,0.5]
cints = np.arange(-2.0,2.1,0.1) # SSH

for im in range(12):
    
    # Initialize figure
    fig,ax,_    = viz.init_orthomap(1,1,bboxplot,figsize=(24,14.5),)
    ax          = viz.add_coast_grid(ax,bbox=bboxplot,fill_color="lightgray",fontsize=24)
    
    # Plot the Vectors
    plotu   = ugeo_monmean.ug.isel(month=im) * mask
    plotv   = ugeo_monmean.vg.isel(month=im) * mask
    lon     = plotu.lon.data
    lat     = plotu.lat.data
    qv      = ax.quiver(lon[::qint],lat[::qint],
                        plotu.data[::qint,::qint],plotv.data[::qint,::qint],
                        transform=proj,scale=scale)
    
    # Plot the amplitude
    plotvar     = ugeo_mod_monmean.isel(month=im)
    pcm         = ax.pcolormesh(plotvar.lon,plotvar.lat,plotvar,
                                vmin=vlms[0],vmax=vlms[1],cmap='cmo.tempo',transform=proj,zorder=-1)
    cb          = viz.hcbar(pcm,ax=ax,fraction=0.035)
    cb.set_label("Current Speed (m/s)",fontsize=fsz_axis)
    cb.ax.tick_params(labelsize=fsz_tick)
    
    # plot SSH
    plotvar = ds_ssh.isel(mon=im)
    cl      = ax.contour(plotvar.lon,plotvar.lat,plotvar,levels=cints,
                         transform=proj,
                         linewidths=1.2,colors="navy",zorder=3)
    ax.clabel(cl,fontsize=fsz_tick)
    
    # # Plot Gulf Stream Position
    ax.plot(ds_gs2.lon.isel(mon=im),ds_gs2.lat.isel(mon=im),transform=proj,lw=2.5,c='red',ls='dashdot')
    
    # Plot Ice Edge
    ax.contour(icemask.lon,icemask.lat,mask_plot,colors="cyan",linewidths=2,
               transform=proj,levels=[0,1],zorder=-1)
    
    ax.set_title("%s Mean Geostrophic Currents and SSH\nCESM1-LE 1920-2005, 42-Member Ens. Avg" % (mons3[im]),fontsize=fsz_title,y=1.05)
    #ax.set_title("CESM1 Historical Ens. Avg., Ann. Mean",fontsize=fsz_title)
    
    # Save the File
    savename = "%sCESM1_HTR_EnsAvg_Ugeo_MonMean_mon%02i.png" % (figpath,im+1)
    plt.savefig(savename,dpi=200,bbox_inches='tight')

# <0> <0>  <0> <0> <0> <0>  <0> <0> <0> <0>  <0> <0> <0> <0>  <0> <0> <0> <0>  <0> <0>
#%% Select a point and examine the power spectra
# <0> <0>  <0> <0> <0> <0>  <0> <0> <0> <0>  <0> <0> <0> <0>  <0> <0> <0> <0>  <0> <0>
lonf            = -30
latf            = 50
ds_ugeo_pt      = ds_ugeo.sel(lon=lonf,lat=latf,method='nearest')
ugeo_mod_pt     = ugeo_mod.sel(lon=lonf,lat=latf,method='nearest')

ugeo_mod_pt_ds  = proc.xrdeseason(ugeo_mod_pt)
locfn,loctitle  = proc.make_locstring(lonf,latf)

#%% Compute the spectra

nsmooth  = 2
pct      = 0.10
loopdim  = 0 # Ens
dtmon    = 3600*24*30

in_ts    = ugeo_mod_pt_ds.data#_ds.data
specdict = scm.quick_spectrum(in_ts,nsmooth,pct,dt=dtmon,dim=loopdim,return_dict=True,make_arr=True)


#%% Visualize the spectra

xlims       = [1/(86*12*dtmon),1/(2*dtmon)]
xtks_per    = np.array([50,25,10,5,2,1,0.5]) # in Years
xtks_freq   = 1/(xtks_per * 12 * dtmon)

fig,ax      = plt.subplots(1,1,constrained_layout=True,figsize=(12,4.5))

# -----------------------------------------------------------------------------
# Plot each Ensemble
for e in range(42):
    plotspec = specdict['specs'][e,:]
    plotfreq = specdict['freqs'][e,:]
    ax.loglog(plotfreq,plotspec,alpha=0.25,label="")
 
# PLot Ens Avg
plotspec = specdict['specs'].mean(0)
plotfreq = specdict['freqs'].mean(0)
ax.loglog(plotfreq,plotspec,alpha=1,label="CESM1 Ens. Avg",color="navy")

# Plot AR(1) Null Hypothesis
plotcc0 = specdict['CCs'][:,:,0].mean(0)
plotcc1 = specdict['CCs'][:,:,1].mean(0)
ax.loglog(plotfreq,plotcc1,alpha=1,label="95% Conf.",color="k",lw=0.75,ls='dashed')
ax.loglog(plotfreq,plotcc0,alpha=1,label="Null Hypothesis",color="k",lw=0.75)
#ax.loglog(plotfreq,plotspec,alpha=1,label="CESM1 Ens. Avg",color="navy")
#ax.axvline([1/(2*dtmon)],label="NQ")

# Draw some lines
ax.axvline([1/(6*dtmon)],label="Semiannual",color='lightgray',ls='dotted')
ax.axvline([1/(12*dtmon)],label="Annual",color='gray',ls='dashed')
ax.axvline([1/(10*12*dtmon)],label="Decadal",color='dimgray',ls='dashdot')
ax.axvline([1/(50*12*dtmon)],label="50-yr",color='k',ls='solid')
ax.legend()

# Label Frequency Axis
ax.set_xlabel("Frequency (Cycles/Sec)")
ax.set_xlim(xlims)

# Label y-axis
ax.set_ylabel("Power ([m/s]$^2$/cps)")

# # Plot reference Line

ax.loglog(plotfreq,1/plotfreq,c='red',lw=0.5,ls='dotted')

ax2 = ax.twiny()
ax2.set_xlim(xlims)
ax2.set_xscale('log')
ax2.set_xticks(xtks_freq,labels=xtks_per)
ax2.set_xlabel("Period (Years)")
ax.set_title("Power Spectra for Geostrophic Currents @ %s" % loctitle)

savename = "%sCESM1_HTR_EnsAvg_Ugeo_spectra_%s_nsmooth%04i.png" % (figpath,locfn,nsmooth)
plt.savefig(savename,dpi=200,bbox_inches='tight')

# <0> <0>  <0> <0> <0> <0>  <0> <0> <0> <0>  <0> <0> <0> <0>  <0> <0> <0> <0>  <0> <0>
#%% Additional section (check deseasoning across timescales)
# <0> <0>  <0> <0> <0> <0>  <0> <0> <0> <0>  <0> <0> <0> <0>  <0> <0> <0> <0>  <0> <0>
e       = 0
winlen  = 30
ntime   = 1032
nyr     = int(ntime/12)


ts_test       = ugeo_mod_pt.isel(ens=e).data

ts_test_monyr = ts_test.reshape(nyr,12)


scycles = []
ychunk   = []
niter   = np.arange(0,nyr-winlen)
for ni in range(len(niter)):
    
    print("Take from %i to %i" % (ni,ni+winlen))
    ychunk.append("%i to %i" % (ni,ni+winlen))
    
    ts_seg  = ts_test_monyr[(ni):(ni+winlen),:].mean(0)
    scycles.append(ts_seg)
scycles = np.array(scycles)

#%% Plot everything

fig,ax  = plt.subplots(1,1)

alphas  = 0.05 + np.linspace(0.05,1,scycles.shape[0])
for ii in range(scycles.shape[0]):
    ax.plot(mons3,scycles[ii,:],label="",alpha=alphas[ii])
    
#%% Plot a subset

yids        = [0,10,20,30,40,50,55]
fig,ax      = viz.init_monplot(1,1)

basealpha   = 0.15
alphas      = basealpha + np.linspace(basealpha,1-basealpha,len(yids))

for ii in range(len(yids)):
    yid = yids[ii]
    ax.plot(mons3,scycles[yid,:],label="Year " + ychunk[yid],lw=2.5,alpha=alphas[ii])
ax.legend(bbox_to_anchor=(0.45,1.1, 0.5, 0.5),ncol=3)
ax.set_title("Seasonal Cycle in $|u_{geo}'|$ by Time Period (Years)\n@%s" % (loctitle))

ax.set_ylabel("[m/s]")
ax.set_xlabel("Month")

savename = "%sSeasonal_Cycle_by_Time_Period_%s.png" % (figpath,locfn)
plt.savefig(savename,dpi=200,bbox_inches='tight')

# <0> <0>  <0> <0> <0> <0>  <0> <0> <0> <0>  <0> <0> <0> <0>  <0> <0> <0> <0>  <0> <0>
#%% Remove Seasonal Cycle By Period
# <0> <0>  <0> <0> <0> <0>  <0> <0> <0> <0>  <0> <0> <0> <0>  <0> <0> <0> <0>  <0> <0>

#ds      = ugeo_mod_pt.isel(ens=0)
periods = (
    ['1920-01-01','1949-12-31'], # 00 to 30
    ['1950-01-01','1979-12-31'], # 30 to 60
    ['1980-01-01','2005-12-31'], # 55 to 86 >> Remove last full seasonal cycle  
    )

def deseason_byperiod(ds,periods,scycles=None,return_scycle=False):
    """
    Parameters
    ----------
    ds : TYPE
        Input DS containing dimension "time"
    periods : List of STR
        List containing pairs of [starttime,endtime] used for slicing corresponding to periods
        WARNING: Make sure there are no overlaps or gap...
    scycles : List of scycle with dim 'month', optional
        If provided, remove the seasonal cycle for each period. Otherwise comute seasonal cycle on the spot

    Returns
    -------
    ds_deseasoned : TYPE
        Deseasoned variable

    """
    
    ds_chunks   = []
    npers       = len(periods)
    scycle_out  = []
    for ip in range(npers): # Loop for each period
        trange          = periods[ip]
        ds_trange       = ds.sel(time=slice(trange[0],trange[1]))
        if scycles is None:
            
            scycle = ds_trange.groupby('time.month').mean('time')
        else:
            
            scycle = scycles[ip]
        
        ds_trange_ds    = ds_trange.groupby('time.month') - scycle #proc.xrdeseason(ds_trange,scycle)
        ds_chunks.append(ds_trange_ds)
        scycle_out.append(scycle)
    ds_deseasoned = xr.concat(ds_chunks,dim='time')
    
    if return_scycle:
        return ds_deseasoned,scycle_out
    return ds_deseasoned

# Apply Deseason and Compute Spectra
ugeo_mod_pt_ds_byper,scycle_remove  = deseason_byperiod(ugeo_mod_pt,periods,return_scycle=True)
specdict_byper                      = scm.quick_spectrum(ugeo_mod_pt_ds_byper.data,nsmooth,pct,dt=dtmon,dim=loopdim,return_dict=True,make_arr=True)

#%% Check effect of detrending

specdict = specdict_byper

fig,ax      = plt.subplots(1,1,constrained_layout=True,figsize=(12,4.5))

# -----------------------------------------------------------------------------
# Plot each Ensemble
for e in range(42):
    plotspec = specdict['specs'][e,:]
    plotfreq = specdict['freqs'][e,:]
    ax.loglog(plotfreq,plotspec,alpha=0.25,label="")
 
# PLot Ens Avg
plotspec = specdict['specs'].mean(0)
plotfreq = specdict['freqs'].mean(0)
ax.loglog(plotfreq,plotspec,alpha=1,label="CESM1 Ens. Avg",color="navy")

# Plot AR(1) Null Hypothesis
plotcc0 = specdict['CCs'][:,:,0].mean(0)
plotcc1 = specdict['CCs'][:,:,1].mean(0)
ax.loglog(plotfreq,plotcc1,alpha=1,label="95% Conf.",color="k",lw=0.75,ls='dashed')
ax.loglog(plotfreq,plotcc0,alpha=1,label="Null Hypothesis",color="k",lw=0.75)
#ax.loglog(plotfreq,plotspec,alpha=1,label="CESM1 Ens. Avg",color="navy")
#ax.axvline([1/(2*dtmon)],label="NQ")

# Draw some lines
ax.axvline([1/(6*dtmon)],label="Semiannual",color='lightgray',ls='dotted')
ax.axvline([1/(12*dtmon)],label="Annual",color='gray',ls='dashed')
ax.axvline([1/(10*12*dtmon)],label="Decadal",color='dimgray',ls='dashdot')
ax.axvline([1/(50*12*dtmon)],label="50-yr",color='k',ls='solid')
ax.legend()

# Label Frequency Axis
ax.set_xlabel("Frequency (Cycles/Sec)")
ax.set_xlim(xlims)

# Label y-axis
ax.set_ylabel("Power ([m/s]$^2$/cps)")

# # Plot reference Line

ax.loglog(plotfreq,1/plotfreq,c='red',lw=0.5,ls='dotted')

ax2 = ax.twiny()
ax2.set_xlim(xlims)
ax2.set_xscale('log')
ax2.set_xticks(xtks_freq,labels=xtks_per)
ax2.set_xlabel("Period (Years)")
ax.set_title("Power Spectra for Geostrophic Currents @ %s" % loctitle)

savename = "%sCESM1_HTR_EnsAvg_Ugeo_spectra_%s_nsmooth%04i_byperiod.png" % (figpath,locfn,nsmooth,)
plt.savefig(savename,dpi=200,bbox_inches='tight',transparent=True)


# ==================================================
#%% Try to Compute the geostrophic advection Terms
# ==================================================

# Compute anomalous geostrophic advection
st        = time.time()
ugeoprime = proc.xrdeseason(ds_ugeo)
print("Computed anomalous advection in %.2fs" % (time.time()-st))

#%% Read out the files

# Get Anomalous Geostrophic Advection
ug                  = ugeoprime.ug#.data # (lat, lon, ens, time)
vg                  = ugeoprime.vg#.data # (lat, lon, ens, time)

# Get Mean Gradients
def preproc_grads(ds):
    if 'ensemble' in list(ds.dims):
        ds = ds.rename(dict(ensemble='ens'))
    ds = ds.transpose('lat','lon','ens','month')
    return ds
ds_gradT = preproc_grads(ds_gradT)
ds_gradS = preproc_grads(ds_gradS)

# Apply monthly gradients to each month
st          = time.time()
ugprime_dTx = ug.groupby('time.month') * ds_gradT.dTdx2
vgprime_dTy = vg.groupby('time.month') * ds_gradT.dTdy2

ugprime_dSx = ug.groupby('time.month') * ds_gradS.dTdx2
vgprime_dSy = vg.groupby('time.month') * ds_gradS.dTdy2
print("Computed in %.2f" % (time.time()-st))

#%% Plot An Instant

e               = 22
t               = 105
dtmon           = 3600*24*30
qint            = 1
scale           = 5
cints_sst       = np.arange(250,310,2)
cints_sss       = np.arange(33,39,.3)
plotmode        = None # u, v, or none

# Initialize figure
fig,ax,_        = viz.init_orthomap(1,1,bboxplot,figsize=(24,14.5),)
ax              = viz.add_coast_grid(ax,bbox=bboxplot,fill_color="lightgray",fontsize=24)

# Plot the forcing
plotvar = (ugprime_dTx + vgprime_dTy).isel(time=t,ens=e) * dtmon * -1
pcm     = ax.pcolormesh(plotvar.lon,plotvar.lat,plotvar,transform=proj,
                        cmap='cmo.balance',vmin=-2.5,vmax=2.5)
cb      = viz.hcbar(pcm,ax=ax)
cb.set_label(r"Forcing ($u_{geo} \nabla \overline{T}$) [$\degree$C  per month]",fontsize=fsz_axis)

# Plot the contours
timestep = plotvar.time
im       = plotvar.time.month.item() # Get the Month
cints_grad = np.arange(-30,33,3)
if plotmode == "u":
    plotvar = ds_gradT.dTdx2.isel(ens=e,month=im) * dtmon
    cl      = ax.contour(plotvar.lon,plotvar.lat,plotvar * mask,transform=proj,colors='firebrick',levels=cints_grad)
    ax.clabel(cl,fontsize=fsz_tick)
elif plotmode == "v":
    plotvar = ds_gradT.dTdy2.isel(ens=e,month=im) * dtmon
    cl      = ax.contour(plotvar.lon,plotvar.lat,plotvar * mask,transform=proj,colors='firebrick',levels=cints_grad)
    ax.clabel(cl,fontsize=fsz_tick)
else:
    
    plotvar = ds_sst.SST.isel(ens=e,mon=im) # Mean Gradient
    cl      = ax.contour(plotvar.lon,plotvar.lat,plotvar * mask,transform=proj,colors='firebrick',levels=cints_sst)
    ax.clabel(cl,fontsize=fsz_tick)

# Plot the anomalous Geo Advection
if plotmode == "v":
    plotu   = xr.ones_like(ugeoprime.ug.isel(time=t,ens=e)) * mask
else:
    plotu   = ugeoprime.ug.isel(time=t,ens=e) * mask
if plotmode == 'u':
    plotv = xr.ones_like(ugeoprime.ug.isel(time=t,ens=e)) * mask
else:
    plotv   = ugeoprime.vg.isel(time=t,ens=e) * mask
lon     = plotu.lon.data
lat     = plotu.lat.data
qv      = ax.quiver(lon[::qint],lat[::qint],
                    plotu.data[::qint,::qint],plotv.data[::qint,::qint],
                    transform=proj,scale=scale,color='blue')

ax.set_title("Anomalous Geostrophic Advection @ %s, Ens. Member %02i" % (proc.noleap_tostr(timestep),e+1),fontsize=fsz_title)


# # Plot mean Ekman Advection
# plotu   = ds_ugeo.ug.isel(time=t,ens=e) * mask
# plotv   = ds_ugeo.vg.isel(time=t,ens=e) * mask
# lon     = plotu.lon.data
# lat     = plotu.lat.data
# qv      = ax.quiver(lon[::qint],lat[::qint],
#                     plotu.data[::qint,::qint],plotv.data[::qint,::qint],
#                     transform=proj,scale=scale,color='k')

# #%% Plot Mean Advection for both temperature and salinity
# scale            = .001

# fig,axs,_        = viz.init_orthomap(1,2,bboxplot,figsize=(24,14.5),)

# for ax in axs:
#     ax = viz.add_coast_grid(ax,bbox=bboxplot,fill_color="lightgray",fontsize=24)
    
#     ax.plot(ds_gs2.lon.mean('mon'),ds_gs2.lat.mean('mon'),transform=proj,lw=2.5,c='red',ls='dashdot')

#     # Plot Ice Edge
#     ax.contour(icemask.lon,icemask.lat,mask_plot,colors="cyan",linewidths=2,
#                transform=proj,levels=[0,1],zorder=-1)
    
#     # Plot the Geostrophic Currents
    
#     # Plot the anomalous Ekman Advection
#     plotu   = ugeoprime.ug.groupby('time').mean('time').mean('ens') * mask
#     plotv   = ugeoprime.vg.mean('time').mean('ens')* mask
#     lon     = plotu.lon.data
#     lat     = plotu.lat.data
#     qv      = ax.quiver(lon[::qint],lat[::qint],
#                         plotu.data[::qint,::qint],plotv.data[::qint,::qint],
#                         transform=proj,scale=scale,color='blue')
    

# Plot the Geostrophic Currents

#%% Do a sanity check

lonf            = -30
latf            = 50
t               = 44
e               = 22

ugpt            = ug.sel(lon=lonf,lat=latf,method='nearest').isel(time=t,ens=e)
im              = ugpt.time.month.data.item() - 1
dTxpt           = ds_gradT.dTdx2.sel(lon=lonf,lat=latf,method='nearest').isel(month=im,ens=e)

ugprime_dTx_pt  = ugprime_dTx.sel(lon=lonf,lat=latf,method='nearest').isel(time=t,ens=e)
print("%.4e * %.4e = %.4e == %.4e" % (ugpt.data.item(),
                              dTxpt.data.item(),
                              ugpt.data.item() * dTxpt.data.item(),
                              ugprime_dTx_pt.data.item(),
                             ))

# ++++++++++++++++++++++++++++++++++++++++++++++++
#%% Compute (and save) terms, and monthly variance
# ++++++++++++++++++++++++++++++++++++++++++++++++

ugeoprime_gradTbar        = -1 * (ugprime_dTx + vgprime_dTy)
ugeoprime_gradSbar        = -1 * (ugprime_dSx + vgprime_dSy)

ugeoprime_gradTbar_monvar = ugeoprime_gradTbar.groupby('time.month').var('time').rename("SST")
ugeoprime_gradSbar_monvar = ugeoprime_gradSbar.groupby('time.month').var('time').rename("SSS")

geoterm_T_savg            = proc.calc_savg(ugeoprime_gradTbar_monvar.rename(dict(month='mon')),ds=True)
geoterm_S_savg            = proc.calc_savg(ugeoprime_gradSbar_monvar.rename(dict(month='mon')),ds=True)

ugeo_grad_monvar          = xr.merge([ugeoprime_gradTbar_monvar,ugeoprime_gradSbar_monvar,mask.rename('mask')])


savename = "%sugeoprime_gradT_gradS_NATL_Monvar.nc" % rawpath
edict    = proc.make_encoding_dict(ugeo_grad_monvar)
ugeo_grad_monvar.to_netcdf(savename,encoding=edict)


#%% Examine the area average contribution of this term and plot monhtly mean

nens            = 42
bbox_sel        = [-40,-30,40,50] # NAC
ugeo_term_reg   = proc.sel_region_xr(ugeo_grad_monvar,bbox_sel).mean('lon').mean('lat')
fig,axs         = viz.init_monplot(1,2,figsize=(12.5,4.5))#plt.subplots(1,1,constrained_layout=True,figsize=(12,4.5))

for ii in range(2):
    ax = axs[ii]
    if ii == 0:
        plotvar = ugeo_term_reg.SST
        vname   = "SST"
        vunit   = "$\degree C$"
    else:
        plotvar = ugeo_term_reg.SSS
        vname   = "SSS"
        vunit   = "psu"
        
        
    for e in range(nens):
        ax.plot(mons3,plotvar.isel(ens=e) * dtmon**2,alpha=0.5)
    ax.plot(mons3,plotvar.mean('ens') * dtmon**2,color="k")
    ax.set_title(r"$u'_{geo} \nabla \overline{%s} $ " % vname)
    ax.set_ylabel("Interannual Variability (%s per mon)" % vunit)

#%% Plot Mean Variance of the geostrophic forcing term

vnames          = ["SST","SSS"]
vunits          = ["\degree C","psu"]
vmaxes_var      = [0.5,0.01]
vcmaps          = ["cmo.thermal",'cmo.haline']

qint            = 2

fig,axs,_        = viz.init_orthomap(1,2,bboxplot,figsize=(24,14.5),)

for ax in axs:
    ax = viz.add_coast_grid(ax,bbox=bboxplot,fill_color="lightgray",fontsize=24)
    
    ax.plot(ds_gs2.lon.mean('mon'),ds_gs2.lat.mean('mon'),transform=proj,lw=2.5,c='red',ls='dashdot')

    # Plot Ice Edge
    ax.contour(icemask.lon,icemask.lat,mask_plot,colors="cyan",linewidths=2,
               transform=proj,levels=[0,1],zorder=-1)
    
    # Plot the mean Geostrophic Currents
    # Plot the anomalous Ekman Advection
    plotu   = ds_ugeo.ug.mean('time').mean('ens') * mask
    plotv   = ds_ugeo.vg.mean('time').mean('ens') * mask
    lon     = plotu.lon.data
    lat     = plotu.lat.data
    qv      = ax.quiver(lon[::qint],lat[::qint],
                        plotu.data[::qint,::qint],plotv.data[::qint,::qint],
                        transform=proj,scale=scale,color='cyan')
    
for vv in range(2):
    
    ax      = axs[vv]
    vname   = vnames[vv]
    vunit   = vunits[vv]
    plotvar = (ugeo_grad_monvar[vname] * dtmon**2).mean('month').mean('ens') * mask
    
    pcm     = ax.pcolormesh(plotvar.lon,plotvar.lat,plotvar,transform=proj,vmax=vmaxes_var[vv],zorder=-2,cmap=vcmaps[vv])
    
    # Plot Contours
    if vname == "SST":
        plotvar = ds_sst.SST.isel(ens=e).mean('mon') # Mean Gradient
        cl      = ax.contour(plotvar.lon,plotvar.lat,plotvar * mask,transform=proj,colors='red',levels=cints_sst)
        ax.clabel(cl,fontsize=fsz_tick)
    else:
        plotvar = ds_sss.SSS.isel(ens=e).mean('mon') # Mean Gradient
        cl      = ax.contour(plotvar.lon,plotvar.lat,plotvar * mask,transform=proj,colors='violet',levels=cints_sss)
        ax.clabel(cl,fontsize=fsz_tick)
        
    
    
    cb      = viz.hcbar(pcm,ax=ax)
    ax.set_title(r"$u_{geo}' \cdot \nabla \overline{%s}$" % (vname),fontsize=fsz_title)
    cb.set_label(r"Forcing ($u_{geo}' \cdot \nabla \overline{%s}$) [$%s^2$  per month]" % (vname,vunit),fontsize=fsz_axis)
    cb.ax.tick_params(labelsize=fsz_tick)

    
savename = "%sCESM1_HTR_EnsAvg_Ugeoprime_ForcingTermVariance.png" % (figpath,)
plt.savefig(savename,dpi=200,bbox_inches='tight')



#%%
#%% Plot Variance




invar   = ugeo_grad_monvar.SST * dtmon**2


plot_sid = []

fig,axs,_        = viz.init_orthomap(1,4,bboxplot,figsize=(24,14.5),)

for ax in axs.flatten():
    
    ax = viz.add_coast_grid(ax,bbox=bboxplot,fill_color="lightgray",fontsize=24)
    ax.plot(ds_gs2.lon.mean('mon'),ds_gs2.lat.mean('mon'),transform=proj,lw=2.5,c='red',ls='dashdot')
    
    # Plot Ice Edge
    ax.contour(icemask.lon,icemask.lat,mask_plot,colors="cyan",linewidths=2,
               transform=proj,levels=[0,1],zorder=-1)
    
for sid in range(4):
    
    
    ax      = axs.flatten()[sid]
    plotvar = invar.isel(ens=0,month=0)#geoterm_T_savg.isel(season=sid).mean('ens')
    pcm     = ax.pcolormesh(plotvar.lon,plotvar.lat,plotvar,transform=proj,vmax=1,zorder=-2)
    
    
    cb      = viz.hcbar(pcm,ax=ax)
    cb.set_label(r"Forcing ($u_{geo} \nabla \overline{T}$) [$\degree$C  per month]",fontsize=fsz_axis)
    cb.ax.tick_params(labelsize=fsz_tick)
    
    

#%% Test Removal of Ugeo Term from Salinty and Temperature

#%% Load Salinity and temperature

# (Copied from compute coherence SST/SSS)
st     = time.time()
nc_ssti = "CESM1LE_SST_NAtl_19200101_20050101_bilinear.nc"
nc_sssi = "CESM1LE_SSS_NAtl_19200101_20050101_bilinear.nc"
ds_ssti = xr.open_dataset(path_ugeo + nc_ssti).SST.load()
ds_sssi = xr.open_dataset(path_ugeo + nc_sssi).SSS.load()
print("Loaded SST and SSS in %.2fs" % (time.time()-st))

def preproc_var(ds):
    ds  = ds.rename(dict(ensemble='ens'))
    dsa = proc.xrdeseason(ds)
    dsa = dsa - dsa.mean('ens')
    return dsa

invars      = [ds_ssti,ds_sssi]
invars_anom = [preproc_var(ds) for ds in invars]

sst,sss     = invars_anom

#%% Remove the term from salinity and temperature

sst_nogeo = sst - (ugeoprime_gradTbar * dtmon) # Flip sign for negative advection
sss_nogeo = sss - (ugeoprime_gradSbar * dtmon) # ibid 


#%%

# Select data for a pt
lonf      = -30
latf      = 48
ds_all_in = [sst,sst_nogeo,sss,sss_nogeo]
dspt      = [proc.selpt_ds(ds,lonf,latf).data for ds in ds_all_in]

expnames  = ["SST","SST (no $u_{geo}$)","SSS","SSS (no $u_{geo}$)"]
expcolors = ["firebrick","hotpink","navy","cornflowerblue"]

for ii in range(4):
    
    if np.any(np.isnan(dspt[ii])):
        idens,idtime = np.where(np.isnan(dspt[ii].data))
        #idcomb       = np.where(np.isnan(arr_pt_flatten[ii]))[0][0]
        print("NaN Detected in arr %02i (ens=%02i, t=%s)" % (ii,idens+1,idtime))
        dspt[ii][idens[0],idtime[0]] = 0.

#%% Compute some metrics

tsms = []
for ii in range(4):
    
    ints = dspt[ii]
    ints = [ints[ee,:] for ee in range(42)]
    tsm  = scm.compute_sm_metrics(ints)
    tsms.append(tsm)

#%% Check Persistence

kmonth  = 1
lags    = np.arange(37)
xtks    = lags[::3]

fig,ax  = plt.subplots(1,1,constrained_layout=True,figsize=(12,4.5))
ax,_    = viz.init_acplot(kmonth,xtks,lags,ax=ax,title="")

for ii in range(4):
    
    acfs = np.array(tsms[ii]['acfs'][kmonth])
    
    for e in range(42):
        ax.plot(lags,acfs[e,:],alpha=0.05,c=expcolors[ii])
    
    ax.plot(lags,acfs.mean(0),alpha=1,c=expcolors[ii],label=expnames[ii])
        
ax.legend()

#%% Examine the change in variance

fig,axs = viz.init_monplot(1,2,figsize=(12,4.5))

for ii in range(4):
    
    if ii < 2:
        ax = axs[0]
    else:
        ax = axs[1]
    
    # Get the Monthly Variance
    monvar = np.array(tsms[ii]['monvars']) # [Ens x Mon]
    
    # Plot it
    mu = monvar.mean(0)
    ax.plot(mons3,mu,label=expnames[ii],c=expcolors[ii])
    
for ax in axs:
    ax.legend()
    


# +++++++++++++++++++++++++++
#%% Compute and Check Spectra
# +++++++++++++++++++++++++++

nsmooth   = 2
pct       = 0.10
dtmon     = 3600*24*30

specdicts = []
for ii in range(4):
    in_ts    = dspt[ii]
    specdict = scm.quick_spectrum(in_ts,nsmooth,pct,dt=dtmon,return_dict=True,dim=0,make_arr=True)
    specdicts.append(specdict)

    

#%% Visualize the spectra


xlims       = [1/(86*12*dtmon),1/(2*dtmon)]
xtks_per    = np.array([50,25,10,5,2,1,0.5]) # in Years
xtks_freq   = 1/(xtks_per * 12 * dtmon)

fig,axs      = plt.subplots(2,1,constrained_layout=True,figsize=(12,10))


for ii in range(4):
    
    if ii < 2:
        ax = axs[0]
    else:
        ax = axs[1]
    
    specdict = specdicts[ii]
    
    # -----------------------------------------------------------------------------
    # # Plot each Ensemble
    # nens = specdict['specs'].shape[0]
    # for e in range(nens):
    #     plotspec = specdict['specs'][e,:]
    #     plotfreq = specdict['freqs'][e,:]
    #     ax.loglog(plotfreq,plotspec,alpha=0.25,label="",color=ecols[ii],ls=els[ii])
     
    # PLot Ens Avg
    plotspec = specdict['specs'].mean(0)
    plotfreq = specdict['freqs'].mean(0)
    ax.loglog(plotfreq,plotspec,alpha=1,label=expnames[ii] + ", [smooth=%i adj. bands]" % (nsmooth),color=expcolors[ii])
    
    # Plot AR(1) Null Hypothesis
    plotcc0 = specdict['CCs'][:,:,0].mean(0)
    plotcc1 = specdict['CCs'][:,:,1].mean(0)
    #ax.loglog(plotfreq,plotcc1,alpha=1,label="",color=ecols[ii],lw=0.75,ls='dashed')
    #ax.loglog(plotfreq,plotcc0,alpha=1,label="",color=ecols[ii],lw=0.75)

    
# # Draw some line

for ax in axs:
    ax.axvline([1/(6*dtmon)],label="Semiannual",color='lightgray',ls='dotted')
    ax.axvline([1/(12*dtmon)],label="Annual",color='gray',ls='dashed')
    ax.axvline([1/(10*12*dtmon)],label="Decadal",color='dimgray',ls='dashdot')
    ax.axvline([1/(50*12*dtmon)],label="50-yr",color='k',ls='solid')
    ax.legend(ncol=3)
    
    # Label Frequency Axis
    ax.set_xlabel("Frequency (Cycles/Sec)",fontsize=fsz_axis)
    ax.set_xlim(xlims)
    
    # Label y-axis
    ax.set_ylabel("Power ([m/s]$^2$/cps)",fontsize=fsz_axis)
    ax.tick_params(labelsize=fsz_tick)
    # # Plot reference Line
    
    #ax.loglog(plotfreq,1/plotfreq,c='red',lw=0.5,ls='dotted')
    
    ax2 = ax.twiny()
    ax2.set_xlim(xlims)
    ax2.set_xscale('log')
    ax2.set_xticks(xtks_freq,labels=xtks_per,fontsize=fsz_tick)
    ax2.set_xlabel("Period (Years)",fontsize=fsz_axis)
    #ax.set_title("Power Spectra for Geostrophic Currents @ %s" % loctitle)





# ---------------------------------------
#%% Case Study, since things seem funky...
# ---------------------------------------

"""
Terms Needed

 >>> Variables
- sst
- sss

 >>> Geostrophic Advection Term
- ugeoprime_gradTbar
- ugeoprime_gradSbar

 >>> Currents
 
 ug
 vg
 
 >>> Gradients
 
 ds_gradT.dTdx2
 ds_gradT.dTdy2
 
 ds_gradS.dTdx2
 ds_gradS.dTdy2

"""

e               = 0
lonf            = -30
latf            = 48

locfn,loctitle = proc.make_locstring(lonf,latf)

#%% Choose a particular Timepoint

zoom_time = True
itime     = 548 # Select A specific Instant

fig,axs = plt.subplots(2,1,constrained_layout=True,figsize=(12.5,8),sharex=True)

sstpt = proc.selpt_ds(sst.isel(ens=e),lonf,latf)
ssspt = proc.selpt_ds(sss.isel(ens=e),lonf,latf)
times = [proc.noleap_tostr(ssspt.time[ii]) for ii in range(1032)]
plott = np.arange(1032)
xtks  = plott[::(12*10)]

for ax in axs:
    ax.set_xticks(xtks,labels=np.array(times)[xtks],fontsize=fsz_tick)
    if zoom_time:
        ax.set_xlim([itime-120,itime+120])
    else:
        ax.set_xlim([xtks[0],plott[-1]])
    
    ax.grid(True,ls='dashed')
    ax.tick_params(labelsize=fsz_tick)

ax = axs[0]
ax.plot(plott,sstpt,c='magenta')
#sstpt.plot(ax=ax)
ax.axvline([itime],c='k')
ax.set_ylabel("SST ($\degree C$)",fontsize=fsz_axis)

ax = axs[1]
ax.plot(plott,ssspt,c='firebrick')
#ssspt.plot(ax=ax)

ax.axvline([itime],c='k')
ax.set_ylabel("SSS (psu)",fontsize=fsz_axis)

plt.suptitle("Anomaly Timeseries @ %s, Ens %02i" % (loctitle,e+1),fontsize=fsz_title)

savename = "%sCESM1_HTR_EnsAvg_Ugeoprime_Event_Timeseries_Ens%02i_t%3i_zoom%i.png" % (figpath,e+1,itime,zoom_time)
plt.savefig(savename,dpi=200,bbox_inches='tight')
#ax.set_xlim([350,390])

#%%


t               = itime
dtmon           = 3600*24*30
qint            = 1
scale           = 2.5
cints_sst       = np.arange(250,310,2)
cints_sss       = np.arange(33,39,.3)
zoom            = None # [-60,-20,40,55]
# Initialize figure

if zoom is not None:
    bbin = zoom
    zoomflag=True
else:
    bbin = bboxplot
    zoomflag=False

fig,axs,_        = viz.init_orthomap(1,2,bbin,figsize=(24,10),)
    
for ax in axs:
    ax              = viz.add_coast_grid(ax,bbox=bbin,fill_color="lightgray",fontsize=24)
    
    # Plot Anomalous Advection
    plotu   = ugeoprime.ug.isel(time=t,ens=e) * mask
    plotv   = ugeoprime.vg.isel(time=t,ens=e) * mask
    lon     = plotu.lon.data
    lat     = plotu.lat.data
    qv      = ax.quiver(lon[::qint],lat[::qint],
                        plotu.data[::qint,::qint],plotv.data[::qint,::qint],
                        transform=proj,scale=scale,color='blue')
    
    # Plot target point
    ax.plot(lonf,latf,transform=proj,marker="*",markersize=15,c='yellow',markeredgecolor='k')

for vv in range(2):
    ax = axs[vv]
    timestep = sstpt.time.isel(time=t)
    im       = sstpt.time.isel(time=t).month.item()
    
    
    if vv == 0: # SST
        
        # Mean Values
        vname       = "SST"
        vunit       = "$\degree C$"
        plotgrad    = ds_sst.SST.isel(ens=e,mon=im)
        cints_grad  = cints_sst
        gradcol     = 'magenta'
        
        # Anomaly
        plotanom    = sst.isel(ens=e,time=t)
        vlims_anom  = [-3,3]
        cmap_diff   = 'cmo.balance'
        
        # Forcing
        plotvar     = ugeoprime_gradTbar.isel(ens=e,time=t) * dtmon * mask
        cints_frc   = np.arange(-2,2.2,0.2)
        
        
    elif vv == 1: # SSS
        
        vname       = "SSS"
        vunit       = "$psu$"
        plotgrad    = ds_sss.SSS.isel(ens=e,mon=im)
        cints_grad  = cints_sss
        gradcol     = 'firebrick'
        
        plotanom    = sss.isel(ens=e,time=t)
        vlims_anom  = [-.5,0.5]
        cmap_diff   = 'cmo.delta'
        
        plotvar     = ugeoprime_gradSbar.isel(ens=e,time=t) * dtmon * mask
        cints_frc   = np.arange(-.5,.55,0.05)
    
        
    # Plot the Mean Contours
    cl = ax.contour(plotgrad.lon,plotgrad.lat,plotgrad * mask,transform=proj,colors=gradcol,levels=cints_grad)
    ax.clabel(cl,fontsize=fsz_tick)
    
    # Plot the Anomaly
    pcm = ax.pcolormesh(plotanom.lon,plotanom.lat,plotanom*mask,transform=proj,vmin=vlims_anom[0],vmax=vlims_anom[1],
                        cmap=cmap_diff,zorder=-2)
    
    
    # Add Colorbar
    cb = viz.hcbar(pcm,ax=ax)
    cb.set_label("%s Anomaly (%s)" % (vname,vunit),fontsize=fsz_axis)
    cb.ax.tick_params(labelsize=fsz_tick)
    
    # Add Forcing Contours
    clf = ax.contour(plotvar.lon,plotvar.lat,plotvar,transform=proj,colors="cyan",
                     levels=cints_frc,linewidths=.75)
    ax.clabel(clf,fontsize=fsz_tick)
    
    ax.set_title(r"$u_{geo}' \cdot \nabla \overline{%s}$ [%s per month]" % (vname[0],vunit),fontsize=fsz_axis)
    
    
    # pcm = ax.pcolormesh(plotvar.lon,plotvar.lat,plotvar,transform=proj,
    #                  )
    # cb = viz.hcbar(pcm,ax=ax)
    # cb.set_label("%s Geostrophic Advection Forcing (%s)" % (vname,vunit),fontsize=fsz_axis)
    # cb.ax.tick_params(labelsize=fsz_tick)
    
plt.suptitle("Anomalous Geostrophic Advection @ %s, Ens. Member %02i" % (proc.noleap_tostr(timestep),e+1),fontsize=fsz_title,y=1.05)
savename = "%sCESM1_HTR_EnsAvg_Ugeoprime_Event_Ens%02i_t%3i_zoom%i.png" % (figpath,e+1,t,zoomflag)
plt.savefig(savename,dpi=200,bbox_inches='tight')

# # Plot the forcing
# plotvar = (ugprime_dTx + vgprime_dTy).isel(time=t,ens=e) * dtmon * -1
# pcm     = ax.pcolormesh(plotvar.lon,plotvar.lat,plotvar,transform=proj,
#                         cmap='cmo.balance',vmin=-2.5,vmax=2.5)
# cb      = viz.hcbar(pcm,ax=ax)
# cb.set_label(r"Forcing ($u_{geo} \nabla \overline{T}$) [$\degree$C  per month]",fontsize=fsz_axis)

# # Plot the contours
# timestep = plotvar.time
# im       = plotvar.time.month.item() # Get the Month
# cints_grad = np.arange(-30,33,3)
# if plotmode == "u":
#     plotvar = ds_gradT.dTdx2.isel(ens=e,month=im) * dtmon
#     cl      = ax.contour(plotvar.lon,plotvar.lat,plotvar * mask,transform=proj,colors='firebrick',levels=cints_grad)
#     ax.clabel(cl,fontsize=fsz_tick)
# elif plotmode == "v":
#     plotvar = ds_gradT.dTdy2.isel(ens=e,month=im) * dtmon
#     cl      = ax.contour(plotvar.lon,plotvar.lat,plotvar * mask,transform=proj,colors='firebrick',levels=cints_grad)
#     ax.clabel(cl,fontsize=fsz_tick)
# else:
    
#     plotvar = ds_sst.SST.isel(ens=e,mon=im) # Mean Gradient
#     cl      = ax.contour(plotvar.lon,plotvar.lat,plotvar * mask,transform=proj,colors='firebrick',levels=cints_sst)
#     ax.clabel(cl,fontsize=fsz_tick)






#%%





#%%


# Get Gradients
dTdx = ds_gradT.dTdx2 # (lat, lon, ens, month)
dTdy = ds_gradT.dTdy2 # (lat, lon, ens, month)




#ugeoprime = ugeoprime





ugeoprime_gradTbar = 




#%% ---------------------------------------------------------------------------

# #%% Load the Ekman currents

# st          = time.time()
# nc_uek      = "CESM1_HTR_FULL_Uek_NAO_nomasklag1_nroll0_NAtl.nc"
# path_uek    = rawpath
# ds_uek      = xr.open_dataset(path_uek + nc_uek)#.load()
# print("Loaded uek in %.2fs" % (time.time()-st))

