#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Calculate Heat Penetration Depth (HPD) given a density profile.

Loads output from [crop_NATL_byens.py]

Pointwise loop using

Based on Hosoda et al. 2016

Created on Wed Apr 17 10:03:59 2024

@author: gliu
"""

from amv import proc, viz
import time
import scm

import numpy as np
import xarray as xr
import sys
import os
import tqdm
import matplotlib.pyplot as plt

# ----------------------------------
# %% Import custom modules and paths
# ----------------------------------

# Indicate the Machine!
machine = "Astraeus"

# First Load the Parameter File
cwd = os.getcwd()
sys.path.append(cwd+"/../")
import reemergence_params as rparams

# Paths and Load Modules
pathdict = rparams.machine_paths[machine]
sys.path.append(pathdict['amvpath'])
sys.path.append(pathdict['scmpath'])

# Set needed paths
figpath = pathdict['figpath']
proc.makedir(figpath)
input_path = pathdict['input_path']
output_path = pathdict['output_path']
procpath = pathdict['procpath']

# %%


# datpath = "/Users/gliu/Globus_File_Transfer/CESM1_LE/Historical/PD/"
# ncname  = "b.e11.B20TRC5CNBDRD.f09_g16.002.pop.h.PD.192001-200512.nc"
datpath = pathdict['raw_path'] + "ocn_var_3d/"

ncname = "CESM1_HTR_FULL_PD_NAtl_Crop.nc"

outpath = ""


"""
Which surface sigma value to use?
 "max"     - maximum surface density of mean seasonal cycle (Hosoda et al. 2016)
 "lastmon" - surface density of the previous month
 "currmon" - surface density of the current month

"""
sigmathres = "lastmon" # 


# %% For Debugging, let's try at a single point first =========================

lonf = -30 + 360
latf = 50
outpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/ptdata/lon330_lat50/"

ds   = xr.open_dataset(datpath+ncname).load()
ds   = proc.fix_febstart(ds)

# %% Get data for a point
dspt = proc.find_tlatlon(ds, lonf, latf, verbose=True)

rho  = dspt.PD  # [time x depth]


# %% Here's the process


rho_scycle = rho.isel(z_t=0).groupby('time.month').mean('time')  # [12]
sigma_max = rho_scycle.max().values.item()


rho_in = rho.values
z_t = rho.z_t.values
ntime, nz = rho_in.shape
nyrs = int(ntime/12)
scycle = rho_in[:, 0].reshape(nyrs, 12).mean(0)
sigma_max = np.nanmax(rho_in[:, 0].reshape(nyrs, 12).mean(0))


# Repeat for 3 methods
hpd_methods    = []
scycle_methods = []
for ii in range(3):
    
    if ii == 0:
        sigmathres = "max"
    elif ii == 1:
        sigmathres = "lastmon"
    elif ii == 2:
        sigmathres = "currmon"
    
    
    count = 0
    hpd_pt = []
    for t in range(ntime):
        
        # Select the threshold
        if sigmathres == "max":
            rho_thres = sigma_max
            t_index   = t
        elif sigmathres == "lastmon":
            rho_thres = rho_in[t-1,0]
            t_index   = t - 1
        elif sigmathres == "currmon":
            rho_thres = rho_in[t,0]
            t_index   = t
        
        
        # Take difference and identify point of first crossing
        drho = rho_in[t, :] - rho_thres
        icross = np.where(drho > 0)[0][0]  # Identify point of first crossing
        if drho[icross+1] < 0:
            print("Warning, point after icross at t=%i is still negative")
            count += 1
        
        # Get the depth
        depths = [z_t[icross-1], z_t[icross]]
        rhos = [rho_in[t, icross-1], rho_in[t, icross]]
        zcross = np.interp(rho_thres, rhos, depths,)
        
        
        hpd_pt.append(zcross)
    
    hpd_pt = np.array(hpd_pt)
    scycle = proc.calc_clim(hpd_pt,0)
    
    hpd_methods.append(hpd_pt)
    scycle_methods.append(scycle)
    
hpd_pt = hpd_methods[0]
# fig,ax = plt.subplots(1,1)
# ax.plot(np.array(rhos)-1.025,np.array(depths)/100)
# ax.plot(sigma_max-1.025,zcross/100,marker="x")
#%% Plot to compare methods
hpd_labs = ["Max","Last Month","Current Month"]
hpd_cols = ["orange","violet","limegreen"]
mons3   = proc.get_monstr()
fig, ax = viz.init_monplot(1, 1)

for ii in range(3):
    ax.plot(scycle_methods[ii]/100,label=hpd_labs[ii])
ax.legend()


#%% Check Algorithm and visualize things above
# Note: Need to run section below to get mld variable (dsh)

trange = np.arange(0,36,1)
#times  = rho.time.isel(time=trange)
sigth  = (rho[trange,:]-1)*1000

mons33 = np.tile(mons3,3)

fig,ax = plt.subplots(1,1,constrained_layout=True,figsize=(12,4.5))

# Plot the potential density
cf = ax.contourf(trange,z_t/100,sigth.T,cmap='cmo.dense')

# Plot Threshold
thresplot = (sigma_max-1)*1000
cl        = ax.contour(trange,z_t/100,sigth.T,colors='k',levels=[thresplot,])
ax.clabel(cl)

# Plot the mixed layer depth
hplot = dsh.sel(lon=lonf-360,lat=latf,method='nearest').isel(ens=0).h.values
hplot = np.tile(hplot,3)
ax.plot(trange,hplot,label="HMXL",c='navy')

# Plot the HPD
for ii in range(2):
    hpdplot = hpd_methods[ii][trange]/100
    ax.plot(trange,hpdplot,label="HPD (%s)" % (hpd_labs[ii]),ls='dashed',c=hpd_cols[ii])
    
    
    scplot  = np.tile(scycle_methods[ii]/100,3)
    ax.plot(trange,scplot,c=hpd_cols[ii],ls='solid',lw=1,label="HPD (%s, scycle)" % (hpd_labs[ii]))


ax.legend(ncol=2)

# Labels
ax.set_xticks(trange,labels=mons33)
ax.set_ylim([0,200])
plt.gca().invert_yaxis()
ax.set_ylabel("Depth (m)")
ax.set_xlabel("Time Index (month)")
cb = fig.colorbar(cf,ax=ax,fraction=0.025,pad=0.01,orientation='vertical')
cb.set_label(r"Potential Density $\sigma_{ \theta }$ $(kg/m^3)$")
ax.grid(True,ls='dotted',color="k",alpha=0.4)

ax.set_title("Potential Density and HPD estimates from t= %02i to %02i" % (trange[0],trange[1]))

savename = "%sDensity_Vertical_Profile_HPD_trange%02ito%02i.png" % (figpath,trange[0],trange[1])
plt.savefig(savename,dpi=150)
# %% Make the above into a unfunc

def calc_hpd(rho_in, z_t, debug=True,sigmathres='max'):
    ntime, nz = rho_in.shape
    nyrs = int(ntime/12)
    # Get maximum z
    scycle = rho_in[:, 0].reshape(nyrs, 12).mean(0)
    sigma_max = np.nanmax(rho_in[:, 0].reshape(nyrs, 12).mean(0))

    count = 0
    hpd_pt = []
    for t in range(ntime):  # Loop for each timestep
        
        # Select the threshold
        if sigmathres == "max":
            rho_thres = sigma_max
            t_index   = t
        elif sigmathres == "lastmon":
            rho_thres = rho_in[t-1,0]
            t_index   = t - 1
        elif sigmathres == "currmon":
            rho_thres = rho_in[t,0]
            t_index   = t
                

        # Difference between rho and rho_thres
        drho = rho_in[t, :] - rho_thres
        
        # Identify point of first crossing
        icross = np.where(drho > 0)[0]
        if len(icross) == 0:
            hpd_pt.append(np.nan)
            continue
        else:
            icross = icross[0]  # Take first point
        if drho[icross+1] < 0:
            if debug:
                print("Warning, point after icross at t=%i is still negative" % t)
            count += 1

        # Get the depth
        depths = [z_t[icross-1], z_t[icross]]
        rhos = [rho_in[t, icross-1], rho_in[t, icross]]
        zcross = np.interp(sigma_max, rhos, depths,)
        hpd_pt.append(zcross)
    hpd_pt = np.array(hpd_pt)
    return hpd_pt


hpd_func = lambda a,b: calc_hpd(a,b,sigmathres='lastmon')

st = time.time()
hpd_all = xr.apply_ufunc(
    hpd_func,  # Pass the function
    ds.PD,  # The inputs in order that is expected
    rho.z_t,
    # Which dimensions to operate over for each argument...
    input_core_dims=[['time', 'z_t'], ['z_t']],
    output_core_dims=[['time'],],  # Output Dimension
    # exclude_dims=set(("",)),
    vectorize=True,  # True to loop over non-core dims
)
print("Ran Function in %.2fs" % (time.time()-st))

# Reassign TLAT and TLON
hpd_all['PD'] = hpd_all / 100  # Divide to convert to meters
hpd_all['TLONG'] = ds.TLONG
hpd_all['TLAT'] = ds.TLAT


edict = dict(PD=dict(zlib=True))
savename = "%sCESM1_HTR_FULL_HPD_NAtl_Crop_sigmathres%s.nc" % (datpath,thresname)
hpd_all.to_netcdf(savename, encoding=edict)

# %% Compute Seasonal Cycle and regrid


hpd_scycle = hpd_all.groupby('time.month').mean('time')
hpd_scycle = hpd_scycle.rename(dict(month='mon'))
hpd_scycle = hpd_scycle.rename("PD")
hpd_scycle = hpd_scycle/100

savename = "%sCESM1_HTR_FULL_HPD_NAtl_Crop_Scycle.nc" % datpath
hpd_scycle.to_netcdf(savename, encoding=edict)


# %% Regrid (copied from regrid_detrainment damping)

# Retrieve MLD to grab corresponding region
bbox = [-80, 0, 20, 65]
mldpath = input_path + "mld/"
mldnc = "CESM1_HTR_FULL_HMXL_NAtl.nc"
dsh = xr.open_dataset(mldpath+mldnc)
dshreg = dsh.sel(lon=slice(bbox[0], bbox[1]), lat=slice(bbox[2], bbox[3]))
outlat = dshreg.lat.values
outlon = dshreg.lon.values
nlon, nlat = len(outlon), len(outlat)

# Copied section below (but use hpd_scycle)
ds_in = hpd_scycle.copy()
ds_in = ds_in.rename(dict(TLONG='lon', TLAT='lat'))
dsl = ds_in


# Loop and remap by selecting nearest point
var_out = np.zeros((12, nlat, nlon)) * np.nan
# Took (1h 11 min if you don't load, 2 sec if you load, T-T)
for o in tqdm.tqdm(range(nlon)):
    lonf = outlon[o]

    if lonf < 0:
        lonf += 360

    for a in range(nlat):
        latf = outlat[a]

        # Get the nearest point
        outids = proc.get_pt_nearest(
            dsl, lonf, latf, tlon_name="lon", tlat_name="lat", returnid=True, debug=False)
        dspt = dsl.isel(nlat=outids[0], nlon=outids[1])

        # Reassign variables
        var_out[:, a, o] = dspt.values


coords = dict(mon=np.arange(1, 13, 1), lat=outlat, lon=outlon)
da_hpd = xr.DataArray(var_out, coords=coords, dims=coords, name="h")

savename = "%smld/CESM1_HTR_FULL_HPD_NAtl_Ens01.nc" % (input_path,)
edict = dict(h=dict(zlib=True))
da_hpd.to_netcdf(savename, encoding=edict)


# %% Check to make sure it is the same

# hpd_ufunc = proc.find_tlatlon(hpd_all,lonf,latf)

# #fig,ax = plt.subplots(1,1,figsize=(8,3.4))
# #ax.plot(hpd_ufunc/100)
# #ax.plot(hpd_pt/100,color="orange")
# plt.plot((hpd_ufunc-hpd_pt)/100)

# #%% Debug Plots


# %% Compare Mixed Layer Depths

plotvars = [da_hpd, dsh.isel(ens=0).h]
plotnames = ["HPD", "HMXL"]
lonf = -30
latf = 50
locfn,loctitle=proc.make_locstring(lonf,latf)

mons3 = proc.get_monstr()
fig, ax = viz.init_monplot(1, 1)
for ii in range(2):
    plotvar = proc.selpt_ds(plotvars[ii], lonf, latf)

    ax.plot(mons3, plotvar, label=plotnames[ii],lw=2.5)
ax.legend()
plt.gca().invert_yaxis()
ax.set_ylabel("Depth (meters)")
ax.set_title("Seasonal MLD Cycle @ %s" % loctitle)

savename= "%sHPD_v_HMXL_%s.png"% (figpath,locfn)
plt.savefig(savename,dpi=150,bbox_inches='tight')

#%% Visualize Monthly values of HPD
import cartopy.crs as ccrs

ii = 1

proj  = ccrs.PlateCarree()

vlms  = [0,1200]
invar = plotvars[ii]
lon   = invar.lon
lat   = invar.lat

fig,axs,_ = viz.init_orthomap(4,3,bbox,figsize=(12,14))

for aa in range(12):
    ax = axs.flatten()[aa]
    im = np.roll(np.arange(12),1)[aa]
    ax.set_title(mons3[im],fontsize=20)
    
    ax = viz.add_coast_grid(ax,bbox=bbox,fill_color="k")
    
    if vlms is None:
        pcm = ax.pcolormesh(lon,lat,invar.isel(mon=im),transform=proj,
                            cmap='cmo.dense',)
        fig.colorbar(pcm,ax=ax,pad=0.01,fraction=0.025,orientation='horizontal')
    else:
        pcm = ax.pcolormesh(lon,lat,invar.isel(mon=im),transform=proj,
                            cmap='cmo.dense',vmin=vlms[0],vmax=vlms[1])

if vlms is not None:
    cb = viz.hcbar(pcm,ax=axs.flatten(),fraction=0.025)
    cb.set_label("%s (meters)" % plotnames[ii])
        



savename= "%s%s_Spatial_Pattern.png"% (figpath,plotnames[ii])
plt.savefig(savename,dpi=150,bbox_inches='tight')

#%% Same as above but do differences

import cartopy.crs as ccrs

proj  = ccrs.PlateCarree()

vlms  = [-500,500]
invar = plotvars[1] - plotvars[0]
lon   = invar.lon
lat   = invar.lat

fig,axs,_ = viz.init_orthomap(4,3,bbox,figsize=(12,14))

for aa in range(12):
    ax = axs.flatten()[aa]
    im = np.roll(np.arange(12),1)[aa]
    ax.set_title(mons3[im],fontsize=20)
    
    ax = viz.add_coast_grid(ax,bbox=bbox,fill_color="k")
    
    if vlms is None:
        pcm = ax.pcolormesh(lon,lat,invar.isel(mon=im),transform=proj,
                            cmap='cmo.balance',)
        fig.colorbar(pcm,ax=ax,pad=0.01,fraction=0.025,orientation='horizontal')
    else:
        pcm = ax.pcolormesh(lon,lat,invar.isel(mon=im),transform=proj,
                            cmap='cmo.balance',vmin=vlms[0],vmax=vlms[1])

if vlms is not None:
    cb = viz.hcbar(pcm,ax=axs.flatten(),fraction=0.025)
    cb.set_label("%s-%s (meters)" % (plotnames[1],plotnames[0]))
        



savename= "%sHPD_v_HMXL_diff_Spatial_Pattern.png"% (figpath,)
plt.savefig(savename,dpi=150,bbox_inches='tight')
# ds.sel(lat=latf,lon=lonf,method='nearest')
# %%
