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

# <0> <0>
#%% Check interannual variability of the components

lonf = -35  
latf = 53


def selpt_monvar(ds,lonf,latf):
    dspt = proc.selpt_ds(ds,lonf,latf)
    dspt_monvar = dspt.groupby('time.month').var('time')
    return dspt_monvar


in_ds = [uek_transports[0],uek_transports[1],
         ds_uek.u_ek,ds_uek.v_ek,
         gradfulls[0].dx,gradfulls[0].dy,
         gradfulls[1].dx,gradfulls[1].dy,
         ]
    
in_monvars = [selpt_monvar(ds) for ds in in_ds]
innames    = ["Transport (SST)", "Transport (SSS)", "Uek", "Vek", "dTdx", "dTdy",
              "dSdx","dSdy"]




#%% Plot interannual variability in these terms


fig,axs = viz.init_monplot(1,5,constrained_layout=True,figsize=(22,3.5))

#vv = 0
for vv in [0,1]:
    ax = axs[vv]
    ax.plot(mons3,in_monvars[vv].mean('ens'),label=innames[vv])
    ax.set_title(innames[vv])
    ax.legend()

ax = axs[2] # Plot The Ekman Velocities
for vv in [2,3]:
    ax.plot(mons3,in_monvars[vv].mean('ens'),label=innames[vv])
    ax.set_title("Ekman Velocities")
ax.legend()


ax = axs[3] # Plot SST Gradients
for vv in [4,5]:
    ax.plot(mons3,in_monvars[vv].mean('ens'),label=innames[vv])
    ax.set_title("SST Gradients")
ax.legend()

ax = axs[4] # Plot SSS Gradients
for vv in [6,7]:
    ax.plot(mons3,in_monvars[vv].mean('ens'),label=innames[vv])
    ax.set_title("SSS Gradients")
ax.legend()

# --------------------------------------------------
#%% Load and check more components (wind stress etc)
# --------------------------------------------------
omega     = 7.2921e-5 # rad/sec
rho       = 1026      # kg/m3
mldpath   = input_path + "mld/"
mldnc     = "CESM1_HTR_FULL_HMXL_NAtl.nc"
output_path_uek =  rawpath
tauxnc    = "CESM1LE_TAUX_NAtl_19200101_20050101_bilinear.nc"
tauync    = "CESM1LE_TAUY_NAtl_19200101_20050101_bilinear.nc"
anomalize = False


# Load mixed layer depth climatological cycle, already converted to meters
#mldpath   = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/proc/model_input/mld/" # Take from model input file, processed by prep_SSS_inputs
hclim     = xr.open_dataset(mldpath + mldnc).h.load() # [mon x ens x lat x lon]
hclim     = proc.format_ds_dims(hclim) # ('ens', 'mon', 'lat', 'lon')

#hclim     = hclim.rename({'ens':'ensemble','mon': 'month'})

# First, let's deal with the coriolis parameters
llcoords  = {'lat':hclim.lat.values,'lon':hclim.lon.values,}
xx,yy     = np.meshgrid(hclim.lon.values,hclim.lat.values) 
f         = 2*omega*np.sin(np.radians(yy))
dividef   = 1/f 
dividef[np.abs(yy)<=6] = np.nan # Remove large values around equator
da_dividef = xr.DataArray(dividef,coords=llcoords,dims=llcoords)
# da_dividef.plot()


st          = time.time()
taux        = xr.open_dataset(output_path_uek + tauxnc).load() # (ensemble: 42, time: 1032, lat: 96, lon: 89)
tauy        = xr.open_dataset(output_path_uek + tauync).load()
print("Loaded variables in %.2fs" % (time.time()-st))

# Convert stress from stress on OCN on ATM --> ATM on OCN
taux_flip   = taux.TAUX * -1
tauy_flip   = tauy.TAUY * -1

# Compute Anomalies
if anomalize:
    taux_anom   = proc.xrdeseason(taux_flip)
    tauy_anom   = proc.xrdeseason(tauy_flip)
    
    # Rename Dimension
    taux_anom   = proc.format_ds_dims(taux_anom)
    tauy_anom   = proc.format_ds_dims(tauy_anom)
    
    # Remove Ens. Avg for detrending
    taux_anom   = taux_anom - taux_anom.mean('ens')
    tauy_anom   = tauy_anom - tauy_anom.mean('ens')
else:
    print("Data will not be anomalized")
    taux_anom   = proc.format_ds_dims(taux_flip)
    tauy_anom   = proc.format_ds_dims(tauy_flip)
    
    
#taux_anom

# <0> <0> <0> <0> <0> <0> <0> <0> <0> <0> <0> 
#%% Plot the variation of tau and h
# <0> <0> <0> <0> <0> <0> <0> <0> <0> <0> <0> 

in_ds2      = [taux_anom,tauy_anom,]
in_monvars2 = [selpt_monvar(ds,lonf,latf) for ds in in_ds2]
innames2    = ["tau_x","tau_y","h_bar"]
hpt         = proc.selpt_ds(hclim,lonf,latf)

fig,axs = viz.init_monplot(1,2,figsize=(8,3.5))

ax = axs[0]
for vv in range(2):
    ax.plot(mons3,in_monvars2[vv].mean('ens'),label=innames[2])
ax.legend()
ax.set_title("Wind Stress")

ax = axs[1]
ax.set_title("1/h")
ax.plot(mons3,1/hpt.mean('ens'),label="")
ax.legend()

#%% Lets recompute, but using ekman depths

# Take smaller of 100 or hclim
d_ek = xr.where(hclim > 100, 100, hclim)
d_ek = d_ek.rename(dict(mon='month'))

uek_d = da_dividef / (rho*d_ek) * tauy_anom.groupby('time.month')
vek_d = da_dividef / (rho*d_ek) * taux_anom.groupby('time.month')

# <0> <0> <0> <0> <0> <0> <0> <0> <0> <0> <0> 
#%% Now compare the u_ek for ekman depth effect
# <0> <0> <0> <0> <0> <0> <0> <0> <0> <0> <0> 

in_ds3       = [ds_uek.u_ek*100,ds_uek.v_ek*100,uek_d,vek_d]
in_monvars3  = [selpt_monvar(ds,lonf,latf) for ds in in_ds3]
innames3     = ["uek","vek","uek (d_ek)","vek (d_ek)"]

inc3  = ['b','firebrick','cornflowerblue','hotpink']
inls3 = ['solid','solid','dotted','dotted']

fig,ax = viz.init_monplot(1,1,figsize=(8,4.5))
for vv in range(4):
    ax.plot(mons3,in_monvars3[vv].mean('ens'),label=innames3[vv],lw=4.5,c=inc3[vv],ls=inls3[vv])
ax.legend()


#%% Now compute the ekman transport (total) to see net effect on the Ekman Term

ekman_transport_dek = []
for vv in range(2):
    ektrans = uek_d * gradfulls[vv].dx +  vek_d * gradfulls[vv].dy
    ekman_transport_dek.append(ektrans)
    
#%% Compare the two ekman transports
in_ds4      = [uek_transports[0]*100,ekman_transport_dek[0],uek_transports[1]*100,ekman_transport_dek[1]]
in_monvars4 = [selpt_monvar(ds,lonf,latf) for ds in in_ds4]
innames4     = ["SST Transport","SST Transport (d_ek)","SSS Transport","SSS Transport (d_ek)"]

fig,axs     = viz.init_monplot(1,2,figsize=(8,3.5))


# for ii in range(4):
#     print(ii)
#     ds = in_ds4[ii]
#     selpt_monvar(ds,lonf,latf)


ax = axs[0]
for ii in [0,1]:
    ax.plot(mons3,in_monvars4[ii].mean('ens'),label=innames4[ii])
ax.legend()
ax.set_title("SST Ekman Transport")

ax = axs[1]
for ii in [2,3]:
    ax.plot(mons3,in_monvars4[ii].mean('ens'),label=innames4[ii])
ax.legend()
ax.set_title("SSS Ekman Transport")

    
#%%





#%%
#%%

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