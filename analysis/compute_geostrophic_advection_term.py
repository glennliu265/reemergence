#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Compare currents computed by the followingscripts

    - calc_ekman_advection_htr.py
    - calc_geostrophic_current.py
    - 
    
Calculate Anomalous Geostrophic Advection of Anomalies


Created on Tue Jul 30 10:03:00 2024

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
st          = time.time()
nc_ugeo     = "CESM1LE_ugeo_NAtl_19200101_20050101_bilinear.nc"
path_ugeo   = rawpath
ds_ugeo     = xr.open_dataset(path_ugeo + nc_ugeo).load()
print("Loaded ugeo in %.2fs" % (time.time()-st))

# Compute monthly means
ugeo_monmean     = ds_ugeo.groupby('time.month').mean('time').mean('ens') # [lat x lon x month]
ugeo_mod_monmean = (ugeo_monmean.ug**2 + ugeo_monmean.vg**2)**0.5

#%% Compute the amplitude
ugeo_mod = (ds_ugeo.ug ** 2 + ds_ugeo.vg ** 2)**0.5

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

#%% Select a point and examine the power spectra

lonf            = -30
latf            = 60
ds_ugeo_pt      = ds_ugeo.sel(lon=lonf,lat=latf,method='nearest')
ugeo_mod_pt     = ugeo_mod.sel(lon=lonf,lat=latf,method='nearest')

ugeo_mod_pt_ds  = proc.xrdeseason(ugeo_mod_pt)
locfn,loctitle  = proc.make_locstring(lonf,latf)

#%% Compute the spectra

nsmooth  = 2
pct      = 0.10
loopdim  = 0 # Ens
dtmon    = 3600*24*30

in_ts    = ugeo_mod_pt_ds.data
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

ax2 = ax.twiny()
ax2.set_xlim(xlims)
ax2.set_xscale('log')
ax2.set_xticks(xtks_freq,labels=xtks_per)
ax2.set_xlabel("Period (Years)")
ax.set_title("Power Spectra for Geostrophic Currents @ %s" % loctitle)

savename = "%sCESM1_HTR_EnsAvg_Ugeo_spectra_%s_nsmooth%04i.png" % (figpath,locfn,nsmooth)
plt.savefig(savename,dpi=200,bbox_inches='tight')

# ==================================================
#%% Try to Compute the geostrophic advection Terms
# ==================================================

# Compute anomalous geostrophic advection
st = time.time()
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
t               = 102
dtmon           = 3600*24*30
qint            = 1
scale           = 5
cints_sst       = np.arange(250,310,2)
plotmode        = None # u, v, or none


# Initialize figure
fig,ax,_    = viz.init_orthomap(1,1,bboxplot,figsize=(24,14.5),)
ax          = viz.add_coast_grid(ax,bbox=bboxplot,fill_color="lightgray",fontsize=24)

# Plot the forcing
plotvar = (ugprime_dTx + vgprime_dTy).isel(time=t,ens=e) * dtmon
pcm     = ax.pcolormesh(plotvar.lon,plotvar.lat,plotvar,transform=proj,
                        cmap='cmo.balance',vmin=-2.5,vmax=2.5)
cb      = viz.hcbar(pcm,ax=ax)
cb.set_label(r"Forcing ($u_{geo} \nabla \overline{T}$) [$\degree$C  per month]",fontsize=fsz_axis)

# Plot the contours
timestep = plotvar.time
im      = plotvar.time.month.item() # Get the Month
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

# Plot the anomalous Ekman Advection
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




#%% Compute (and save) monthly variance


ugeoprime_gradTbar = ugprime_dTx + vgprime_dTy
ugeoprime_gradSbar = ugprime_dSx + vgprime_dSy


ugeoprime_gradTbar_monvar = ugeoprime_gradTbar.groupby('time.month').var('time').rename("SST")
ugeoprime_gradSbar_monvar = ugeoprime_gradSbar.groupby('time.month').var('time').rename("SSS")




ugeo_grad_monvar = xr.merge([ugeoprime_gradTbar_monvar,ugeoprime_gradSbar_monvar,mask.rename('mask')])


savename = "%sugeoprime_gradT_gradS_NATL_Monvar.nc"
edict    = proc.make_encoding_dict(ugeo_grad_monvar)
ugeo_grad_monvar.to_netcdf(savename,encoding=edict)


#%% Examine the area average contribution of this term and plot monhtly mean

nens=42

bbox_sel = [-40,-30,40,50] # NAC
ugeo_term_reg = proc.sel_region_xr(ugeo_grad_monvar,bbox_sel).mean('lon').mean('lat')

fig,axs = viz.init_monplot(1,2,figsize=(12.5,4.5))#plt.subplots(1,1,constrained_layout=True,figsize=(12,4.5))

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

