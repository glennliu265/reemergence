#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Compute the total ekman advection
Based on full ugeo term computation in compare_geostrophic_advection_terms

Created on Wed Oct 16 15:15:40 2024

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


dtmon = 3600*24*30
#%% Load Necessary Information


vnames = ["SST","SSS"]

# First, Load the Full Gradients (as computed from compare_geostrophic_advection_terms)
# Takes ~48s
st        = time.time()
gradfulls = []
for vv in range(2):
    vname  = vnames[vv]
    ncname = "%sCESM1_HTR_FULL_Monthly_gradT_FULL_%s.nc" % (rawpath,vname)
    ds     = xr.open_dataset(ncname).load()
    gradfulls.append(ds)
print("Loaded full gradients in %.2fs" % (time.time()-st))


# Next, Load the Ekman Advection (takes ~16s)
st     = time.time()
ncname = "%sCESM1LE_uek_NAtl_19200101_20050101_bilinear.nc" % (rawpath)
ds_uek = xr.open_dataset(ncname).load()
print("Loaded Ekman Currents in %.2fs" % (time.time()-st))

#%% Preprocess to make sure things are the same

gradfulls = [proc.format_ds_dims(ds) for ds in gradfulls]
ds_uek    = proc.format_ds_dims(ds_uek)

#%% Compute Total Ekman transport

uek_transports = []
for vv in range(2):
    
    # Get the gradient
    grad_in = gradfulls[vv]
    
    # Compute the terms
    uek_transport = ds_uek.u_ek * grad_in.dx + ds_uek.v_ek * grad_in.dy
    
    uek_transports.append(uek_transport.copy())

#%% Save the total value

for vv in range(2):
    vname  = vnames[vv]
    ncname = "%sCESM1_HTR_FULL_Uek_%s_Transport_Full.nc" % (rawpath,vname)
    ds_out = uek_transports[vv].rename(vname)
    edict  = proc.make_encoding_dict(ds_out)
    ds_out.to_netcdf(ncname,encoding=edict)

#%% Check roughly to see if they make sense

itime = 0
iens  = 0
vv    = 0

if vv == 0:
    vmax = 1e-2
else:
    vmax = 2.5e-3

# Single Map (Orthomap)

# Font Sizes
fsz_tick  = 14
fsz_axis  = 22
fsz_title = 28

qint       = 2
plot_point = True
pmesh      = False

# Initialize Plot and Map
fig,ax,_    = viz.init_orthomap(1,1,bboxplot,figsize=(24,6.5))
ax          = viz.add_coast_grid(ax,bbox=bboxplot,fill_color="lightgray",fontsize=fsz_tick)


# Plot Currents
plotu = ds_uek.u_ek.isel(ens=iens,time=itime).data
plotv = ds_uek.v_ek.isel(ens=iens,time=itime).data
xx,yy = np.meshgrid(ds_uek.lon.data,ds_uek.lat.data)
ax.quiver(xx[::qint,::qint],yy[::qint,::qint],plotu[::qint,::qint],plotv[::qint,::qint],
          color='darkslateblue',transform=proj,alpha=0.75)

# Plot The Transport
plotvar = uek_transports[vv].isel(ens=iens,time=itime) * dtmon * -1
pcm     = ax.pcolormesh(plotvar.lon,plotvar.lat,plotvar,
                        vmin=-vmax,vmax=vmax,
                        transform=proj,zorder=-1)
cb      = viz.hcbar(pcm,ax=ax)


# # Plot Mean SST (Colors)
# plotvar = ds_sst.SST.mean('ens').mean('mon').transpose('lat','lon') #* mask_apply
# if pmesh:
#     pcm     = ax.pcolormesh(plotvar.lon,plotvar.lat,plotvar,transform=proj,zorder=-1,
#                 linewidths=1.5,cmap="RdYlBu_r",vmin=280,vmax=300)
# else:
#     cints_sstmean = np.arange(280,301,1)
#     pcm     = ax.contourf(plotvar.lon,plotvar.lat,plotvar,transform=proj,zorder=-1,
#                 cmap="RdYlBu_r",levels=cints_sstmean)
# cb = viz.hcbar(pcm,ax=ax,fraction=0.045)
# cb.set_label("SST ($\degree C$)",fontsize=fsz_axis)
# cb.ax.tick_params(labelsize=fsz_tick)

# # Plot Mean SSS (Contours)
# plotvar = ds_sss.SSS.mean('ens').mean('mon').transpose('lat','lon') #* mask_reg
# cl = ax.contour(plotvar.lon,plotvar.lat,plotvar,transform=proj,
#             linewidths=1.5,colors="darkviolet",levels=cints_sssmean,linestyles='dashed')
# ax.clabel(cl,fontsize=fsz_tick)

# # Plot Gulf Stream Positionfig,axs,_       = viz.init_orthomap(1,2,bboxplot,figsize=(20,10))
# ax.plot(ds_gs2.lon.mean('mon'),ds_gs2.lat.mean('mon'),transform=proj,lw=1.75,c='k',ls='dashdot')

# # Plot Ice Edge
# ax.contour(icemask.lon,icemask.lat,mask_plot,colors="cyan",linewidths=2.5,
#            transform=proj,levels=[0,1],zorder=-1)

#figname = "%sCESM1_Locator_MeanState.png" % (figpath,)
#plt.savefig(figname,dpi=200,bbox_inches='tight')

#%% Check values

uek_mag = np.sqrt(ds_uek.u_ek**2 + ds_uek.v_ek**2)
uek_mag.isel(ens=0).mean('time').plot(vmin=-.01,vmax=.01)















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

ds_grad_bar = [ds_gradT,ds_gradS]