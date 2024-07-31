#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Calculate the Geostrophic Current

Using expression from Doglioni et al. 2023

Created on Thu Jul 25 15:34:35 2024

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

#%% Custom Modules

# amvpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/03_Scripts/" # amv module
# scmpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/03_Scripts/stochmod/model/"

amvpath = "/home/glliu/00_Scripts/01_Projects/00_Commons/" # amv module
scmpath = "/home/glliu/00_Scripts/01_Projects/01_AMV/02_stochmod/stochmod/model/" # scm module

sys.path.append(amvpath)
sys.path.append(scmpath)

from amv import proc,viz
import scm
import amv.loaders as dl
#import cvd_utils as cvd

#%% User Edits

# Load SSH
#ncpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/proc/CESM1/NATL_proc/"
#ncname = "CESM1_HTR_SSH_bilinear_regridded_AllEns.nc"

ncpath      = "/stormtrack/data3/glliu/01_Data/02_AMV_Project/03_reemergence/proc/CESM1/NATL_proc/"
ncname      = "CESM1LE_SSH_NAtl_19200101_20050101_bilinear.nc"
outpath     = ncpath


outname     = outpath + "CESM1LE_ugeo_NAtl_19200101_20050101_bilinear.nc"

# Set Constant
RE     = 6.3781e6  # Earth's Radius [m]
g      = 9.81      # Gravitational Accel [m/s2]
omega  = 7.2921e-5 # Earth's rotation rate [rad/s]


def calc_f(phi,omega=7.2921e-5):
    return 2 * omega * np.sin(phi)


def calc_grad_centered(ds): # Copied from structure in calc_ekman_advection_htr
    dx,dy = proc.calc_dx_dy(ds.lon.values,ds.lat.values,centered=True)
    
    # Convert to DataArray
    daconv   = [dx,dy]
    llcoords = {'lat':ds.lat.values,'lon':ds.lon.values,}
    da_out   = [xr.DataArray(ingrad,coords=llcoords,dims=llcoords) for ingrad in daconv]
    dx,dy = da_out
    
    # Roll and Compute Gradients (centered difference)
    ddx = (ds.roll(lon=-1) - ds.roll(lon=1)) / dx
    ddy = (ds.roll(lat=-1) - ds.roll(lat=1)) / dy
    ddy.loc[dict(lat=ddy.lat.values[-1])] = 0 # Set top latitude to zero (since latitude is not periodic)
    
    return ddx,ddy

#%% Load SSH
ds_ssh = xr.open_dataset(ncpath + ncname).load()

#%% Make Meshgrid and compute Coriolis Parameter

# Make Lat/Lon Meshgrid
xx,yy  = np.meshgrid(ds_ssh.lon.data,ds_ssh.lat.data,)
coords = dict(lat=ds_ssh.lat,lon=ds_ssh.lon)
xx     = xr.DataArray(xx,coords=coords,dims=coords,name='lonrad')
yy     = xr.DataArray(yy,coords=coords,dims=coords,name='latrad')
mgrid  = xr.merge([xx,yy])
mgrid_radians = np.radians(mgrid)

# Calculate Coriolis Parameter
f      = calc_f(mgrid_radians.latrad,omega=omega) # [Lat]

# Avoid points within 10 degrees of equator
f_fix  = xr.where(np.abs(f.lat) < 10,np.nan,f)

#%% Try to do distance way (using same approach as calc_ekman_advection)

# Calculate Distances
ds_in      = ds_ssh.SSH / 100

# Compute the gradients
dh_dx,dh_dy = calc_grad_centered(ds_in)

#%% Compute Geostrophic Velocities

"""
          g   dh
u_g =  - --- ----
          f   dy
         
          g   dh
v_g =  - --- ----
          f   dx         
    

"""

ug   = -g/f_fix * dh_dy
vg   = g/f_fix * dh_dx

mask = xr.ones_like(f_fix)
mask = xr.where(np.abs(mask.lat) > 70,np.nan,mask)
    

# Compute the Speed
umod = np.sqrt(ug ** 2 + vg **2)

ds_proc = [ug,vg,umod]
ds_proc = [ds.rename(dict(ensemble='ens')) for ds in ds_proc]
ug,vg,umod = ds_proc



#%% Save the Output

ug    = ug.rename("ug")
vg    = vg.rename("vg")
ugeo  = xr.merge([ug,vg])
edict = proc.make_encoding_dict(ugeo)
ugeo.to_netcdf(outname,encoding=edict)



#%% Debuggg  ==================================================================



#%% Try Plotting the Currents (Timestep)

mons3       = proc.get_monstr()

bboxplot    = [-80,0,20,70]
qint        = 3
t           = 7
e           = 3
scale       = 1
cints       = np.arange(-2.0,2.1,0.1)

fig,ax  = plt.subplots(1,1,
                      subplot_kw={'projection':ccrs.PlateCarree()},constrained_layout=True,)

ax      = viz.add_coast_grid(ax,fill_color="lightgray",bbox=bboxplot)

# Plot the speed
plotvar = umod.isel(ens=e,time=t) * mask
pcm     = ax.pcolormesh(plotvar.lon,plotvar.lat,plotvar,vmin=0,vmax=0.5,
                        cmap='cmo.tempo')
cb      = fig.colorbar(pcm,ax=ax,fraction=0.025)
cb.set_label("Speed (m/s)")


# Plot the Vectors
lon     = ug.lon.data
lat     = ug.lat.data
plotu   = (ug.isel(ens=e,time=t) * mask).data
plotv   = (vg.isel(ens=e,time=t) * mask).data
qv      = ax.quiver(lon[::qint],lat[::qint],plotu[::qint,::qint],plotv[::qint,::qint],
                    scale=scale,)

plotvar = ds_ssh.isel(ensemble=e,time=t).SSH/100
cl      = ax.contour(plotvar.lon,plotvar.lat,plotvar,levels=cints,
                     linewidths=1.2,colors="dimgray")
ax.clabel(cl)
ax.set_title("Geostrophic Currents, Ens %02i, t=%s" % (e+1,ug.time.values[t]))

plt.show()

#%% Scrap =====================================================================

def dplot(ds):
    ds.isel(ens=e,mon=im).plot()
    return None

#%% Differentiate (for degree approach from Doglioni et al 2023)
nlons = np.arange(0,ds_ssh.lon.shape[0])
nlats = np.arange(0,ds_ssh.lat.shape[0])

mgrid_radians['lon'] = nlons
mgrid_radians['lat'] = nlats
dtheta = mgrid_radians.lonrad.differentiate('lon')
dphi   = mgrid_radians.latrad.differentiate('lat')

# Differentiate Eta
ds_ssh_in = ds_ssh.copy()
ds_ssh_in['lon'] = nlons
ds_ssh_in['lat'] = nlats

dssh_dx   = (ds_ssh_in.SSH / 100).differentiate('lon')
dssh_dy   = (ds_ssh_in.SSH / 100).differentiate('lat')


# Replace Lat lon] --------
dssh_dx['lon'] = ds_ssh.lon
dssh_dx['lat'] = ds_ssh.lat

dssh_dy['lon'] = ds_ssh.lon
dssh_dy['lat'] = ds_ssh.lat

dtheta['lon'] = ds_ssh.lon
dtheta['lat'] = ds_ssh.lat

dphi['lon'] = ds_ssh.lon
dphi['lat'] = ds_ssh.lat

mgrid_radians['lon'] = ds_ssh.lon
mgrid_radians['lat'] = ds_ssh.lat


#%% Dogiolini Formula

ug = - g / (RE * f_fix)  * (dssh_dx/dtheta)
vg =   g / (RE * f_fix * np.cos(mgrid_radians.lonrad))  * (dssh_dy/dphi)


dx  = mgrid.lonrad.differentiate('lon')
dy  = mgrid.latrad.differentiate('lat')



#%% Try Plotting the Currents (Monthly)

mons3       = proc.get_monstr()

bboxplot    = [-80,0,20,70]
qint        = 3
im          = 7
e           = 3
scale       = 1
cints       = np.arange(-2.0,2.1,0.1)

fig,ax      = plt.subplots(1,1,
                      subplot_kw={'projection':ccrs.PlateCarree()},constrained_layout=True,)

ax          = viz.add_coast_grid(ax,fill_color="lightgray",bbox=bboxplot)

# Plot the speed
plotvar     = umod.isel(ens=e,mon=im) * mask
pcm         = ax.pcolormesh(plotvar.lon,plotvar.lat,plotvar,vmin=0,vmax=0.5,
                        cmap='cmo.tempo')
cb          = fig.colorbar(pcm,ax=ax,fraction=0.025)
cb.set_label("Speed (m/s)")

# Plot the Vectors
lon     = ug.lon.data
lat     = ug.lat.data
plotu   = (ug.isel(ens=e,mon=im) * mask).data
plotv   = (vg.isel(ens=e,mon=im) * mask).data
qv      = ax.quiver(lon[::qint],lat[::qint],plotu[::qint,::qint],plotv[::qint,::qint],
                    scale=scale,)

# Plot SSH Contours
plotvar = ds_ssh.isel(ens=e,mon=im).SSH/100
cl      = ax.contour(plotvar.lon,plotvar.lat,plotvar,levels=cints,
                     linewidths=1.2,colors="dimgray")



ax.clabel(cl)
ax.set_title("Geostrophic Currents, Ens %02i, Month %s" % (e+1,mons3[im]))


#%% 

