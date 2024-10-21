#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calc_lbd_e

Compute the SST-Evaporation Feedback on SSS
As described in Eqn 25 of Frankignoul et al. 1998.

            lbd_a * Cp * Sbar
lbd_e  = ------------------------
                 L (1 + B)



Copied upper section from visualize_atmospheric_persistence

To Do:
    - Test out seasonal variation in Sbar, Bowen Ratio, etc

Created on Mon Apr 15 17:15:30 2024

@author: gliu

"""


import numpy as np
import xarray as xr
import sys
import time
import matplotlib.pyplot as plt
import matplotlib as mpl
import cartopy.crs as ccrs
import os
from tqdm import tqdm

# ----------------------------------
# %% Import custom modules and paths
# ----------------------------------

# Indicate the Machine!
machine = "Astraeus"

# First Load the Parameter File
sys.path.append("../")

# Paths and Load Modules
import reemergence_params as rparams
pathdict = rparams.machine_paths[machine]

sys.path.append(pathdict['amvpath'])
sys.path.append(pathdict['scmpath'])

# Set needed paths
figpath = pathdict['figpath']

input_path = pathdict['input_path']
output_path = pathdict['output_path']
procpath = pathdict['procpath']
raw_path = pathdict['raw_path']


dpath = input_path + "damping/"
fpath = input_path + "forcing/"
mpath = input_path + "mld/"
maskpath = input_path + "masks/"

from amv import proc,viz

proc.makedir(figpath)
#%% Set Constants

dt         = 3600*24*30 # Timestep [s]
cp         = 3850       # 
rho        = 1026       # Density [kg/m3]
B          = 0.2        # Bowen Ratio, from Frankignoul et al 1998
L          = 2.5e6      # Specific Heat of Evaporation [J/kg], from SSS model document

#%% CESM1 Version  (Astraeus) -----------------------------------------

no_bowen   = True

# Load Sbar # ('mon', 'ens', 'lat', 'lon')
Sbar       = xr.open_dataset(fpath + "CESM1_HTR_FULL_Sbar_NAtl.nc").Sbar.load()

# Load mld # ('mon', 'ens', 'lat', 'lon')
h          = xr.open_dataset(mpath + "CESM1_HTR_FULL_HMXL_NAtl.nc").h.load()

# Load Bowen Ratio # ('ens', 'mon', 'lat', 'lon') --> this has been fixed
B          = xr.open_dataset(dpath + "CESM1LE_BowenRatio_NAtl_19200101_20050101_Bcorr3.nc").B.load()
B          = B.transpose('mon','ens','lat','lon')

# Load Land Ice Mask
#mask      = xr.open_dataset(maskpath + "cesm1_htr_5degbilinear_limask_0.3p_0.05p_year1920to2005_NAtl_enssum.nc").MASK.load()

#%% Indicate file to load

# Load lbd_a # ('mon', 'ens', 'lat', 'lon')
#dampname = "qnet_damping_nomasklag1"
#ncname   = "CESM1_HTR_FULL_qnet_damping_nomasklag1.nc"
dampname = "LHFLX_damping_nomasklag1"
ncname   = "CESM1_HTR_FULL_LHFLX_damping_nomasklag1.nc"
ds       = xr.open_dataset(dpath+ncname).load()
lbd_a    = ds.damping

coords   = h.coords
lbd_a    = lbd_a.assign_coords(coords) # Somethinga bout their latitude doesn't agree?

#%% Calculate lbd_e (see Frankignoul et al. 1998)

# Do Conversion
if no_bowen:
    B = 0
conversion_factor = (cp * Sbar) / (L * (1+B)) # [psu/K]

lbd_a_conv         = lbd_a / (rho*cp*h) # [1/sec]

lbd_e              = lbd_a_conv * conversion_factor # [psu/K/sec]

#%% Save the output (currently in [psu/K/sec])

lbd_e        = lbd_e.rename("lbd_e")
edict        = dict(lbd_e=dict(zlib=True))
savename     = "%sCESM1LE_HTR_FULL_lbde_Bcorr3_lbda_%s.nc" % (fpath,dampname)
if no_bowen:
    savename=proc.addstrtoext(savename,"_noBowen",adjust=-1)
lbd_e.to_netcdf(savename,encoding=edict)

lbd_e_ensavg = lbd_e.mean('ens')
savename     = "%sCESM1LE_HTR_FULL_lbde_Bcorr3_lbda_%s_EnsAvg.nc" % (fpath,dampname)
if no_bowen:
    savename=proc.addstrtoext(savename,"_noBowen",adjust=-1)
lbd_e_ensavg.to_netcdf(savename,encoding=edict)

# -<o>- -<o>- -<o>- -<o>- -<o>- -<o>- -<o>- -<o>- -<o>- -<o>- -<o>- -<o>- -<o>-
#%% Debug Visualation (based on Young-Oh's commment on Draft 03)
"""
Check if summertime maxima in lbd_e is due to evaporation, or due to shallow mixed layers, etc?

"""
bboxplot = [-80, 0, 20, 65]
# Get Variables and Compute Seasonal Average
invars = [lbd_e,lbd_a,Sbar,h]
savgs  = [proc.calc_savg(ds,ds=True) for ds in invars]

rowvar_names = ["$\lambda^e$","$\lambda^a$","$\overline{S}$","$h$"]
rowvar_vlims = [[0,1.5e-8],[-20,20],[34,37],[0,50]]
rowvar_cmaps = ['inferno','cmo.balance','cmo.haline','cmo.dense']
#%% Plot 1 (All Season)
# Components
# Row 1 (lbd_a Raw)
# Row 2 (Sbar)
# Row 3 (h)

fsz_title = 24
proj        = ccrs.PlateCarree()


fig,axs,_ = viz.init_orthomap(4,4,bboxplot=bboxplot,figsize=(18,12))



for ax in axs.flatten():
    ax = viz.add_coast_grid(ax,bboxplot,fill_color="lightgray",fontsize=20,
                            fix_lon=np.arange(-80,10,10),fix_lat=np.arange(0,70,10),grid_color="k")


for vv in range(4):
    
    for ss in range(4):
        ax = axs[vv,ss]
        
        plotvar = savgs[vv].isel(season=ss).mean('ens')
        season  = plotvar.season.item()
        
        vlims = rowvar_vlims[vv]
        cmap  = rowvar_cmaps[vv]
        
        if vv == 0:
            ax.set_title(season,fontsize=fsz_title)
        if ss == 0:
            viz.add_ylabel(rowvar_names[vv],ax=ax,fontsize=fsz_title)
            
        pcm = ax.pcolormesh(plotvar.lon,plotvar.lat,plotvar,
                            vmin=vlims[0],vmax=vlims[1],cmap=cmap,
                            transform=proj)
        cb  = viz.hcbar(pcm,ax=ax)
#%% Plot 2 (Gulf Stream and Summertime Focus)


rowvar_vlims_summer = [[0,1.5e-8],[-20,20],[35,37],[0,30]]
ss          = 2
bbox_gs     = [-80,0,25,50]
fig,axs,_   = viz.init_orthomap(4,1,bboxplot=bbox_gs,figsize=(5,16))

for ax in axs.flatten():
    ax = viz.add_coast_grid(ax,bboxplot,fill_color="lightgray",fontsize=20,
                            fix_lon=np.arange(-80,10,10),fix_lat=np.arange(0,70,10),grid_color="k")



for vv in range(4):
    ax = axs[vv]
    vlims = rowvar_vlims_summer[vv]
    cmap  = rowvar_cmaps[vv]
    ax.set_title(rowvar_names[vv],fontsize=fsz_title)
    plotvar = savgs[vv].isel(season=ss).mean('ens')
    pcm = ax.pcolormesh(plotvar.lon,plotvar.lat,plotvar,
                        vmin=vlims[0],vmax=vlims[1],cmap=cmap,
                        transform=proj)
    cb  = viz.hcbar(pcm,ax=ax)
    
    
# -<o>- -<o>- -<o>- -<o>- -<o>- -<o>- -<o>- -<o>- -<o>- -<o>- -<o>- -<o>- -<o>- 
#%% CESM1 Regrid Version  -----------------------------------------



# Load Sbar # ('mon', 'ens', 'lat', 'lon')
Sbar       = xr.open_dataset(fpath + "cesm1_htr_5degbilinear_Sbar_Global_1920to2005.nc").Sbar.load()

# Load mld # ('mon', 'ens', 'lat', 'lon')
h          = xr.open_dataset(mpath + "cesm1_htr_5degbilinear_HMXL_NAtl_1920to2005.nc").h.load()

# Load Bowen Ratio # ('ens', 'mon', 'lat', 'lon') --> this has been fixed
B          = xr.open_dataset(dpath + "cesm1_htr_5degbilinear_BowenRatio_NAtl_19200101_20050101_Bcorr3.nc").B.load()
B          = B.transpose('mon','ens','lat','lon')

# Load Land Ice Mask
mask      = xr.open_dataset(maskpath + "cesm1_htr_5degbilinear_limask_0.3p_0.05p_year1920to2005_NAtl_enssum.nc").mask.load()

#%% Indicate file to load

# Load lbd_a # ('mon', 'ens', 'lat', 'lon')
dampname = "cesm1le5degqnetDamp"#"qnet_damping_nomasklag1"
ncname   = "cesm1_htr_5degbilinear_qnet_damping_damping_cesm1le5degqnetDamp_EnsAvg.nc"#"CESM1_HTR_FULL_qnet_damping_nomasklag1.nc"
ds       = xr.open_dataset(dpath+ncname).load()
lbd_a    = ds.damping

#coords   = h.coords
#lbd_a    = lbd_a.assign_coords(coords) # Somethinga bout their latitude doesn't agree?

#%% Calculate lbd_e (see Frankignoul et al. 1998)

# Hack Fix
in_ds = [Sbar,B,lbd_a,h]
in_ds = [ds.drop_duplicates('lon') for ds in in_ds]

in_ds = [ds.rename({'month':'mon'})]


Sbar,B,lbd_a,h = in_ds

Sbar = Sbar.rename({'month':'mon'})
# Do Conversion
conversion_factor = (cp * Sbar) / (L * (1+B)) # [psu/K]

lbd_a_conv         = lbd_a / (rho*cp*h) # [1/sec]

lbd_e              = lbd_a_conv * conversion_factor # [psu/K/sec]

#%% Save the output (currently in [psu/K/sec])

lbd_e        = lbd_e.rename("lbd_e")
edict        = dict(lbd_e=dict(zlib=True))
savename     = "%scesm1le_htr_5degbilinear_lbde_Bcorr3_lbda_%s.nc" % (fpath,dampname)
lbd_e.to_netcdf(savename,encoding=edict)

lbd_e_ensavg = lbd_e.mean('ens')
savename     = proc.addstrtoext(savename,"_EnsAvg",adjust=-1)#"%sCESM1LE_HTR_FULL_lbde_Bcorr3_lbda_%s_EnsAvg.nc" % (fpath,dampname)
lbd_e_ensavg.to_netcdf(savename,encoding=edict)




#%% Debugging Viz -------------------------------------------------------------

# Plotting Params
ds = h
mpl.rcParams['font.family'] = 'JetBrains Mono'
bboxplot = [-80, 0, 20, 65]
proj = ccrs.PlateCarree()
lon = ds.lon.values
lat = ds.lat.values
mons3 = proc.get_monstr()
fsz_axis = 16


#%% Visualize Conversion Factor
pv       = conversion_factor.mean('ens').mean('mon')
vlms     = [-.07,.07]
title    = "Conversion Factor (psu/K)"
savename = "%sConversion_factor.png" % figpath

fig,ax,_ = viz.init_orthomap(1,1,bboxplot,figsize=(10,8))
ax       = viz.add_coast_grid(ax,bbox=bboxplot,fill_color="k")
pcm      = ax.pcolormesh(lon,lat,pv,transform=proj,vmin=vlms[0],vmax=vlms[1])
cb       = viz.hcbar(pcm,ax=ax)
#cb = fig.colorbar(pcm,ax=ax,fraction=0.025,pad=0.01,orientation='horizontal')
cb.set_label(title,fontsize=fsz_axis)
plt.savefig(savename,dpi=150,bbox_inches='tight')

#%% Visualize lbd_a_conv

pv       = 1/(lbd_a_conv.mean('ens').mean('mon') * dt)
vlms     = [0,16]#[-.2,.2]#[-1e-7,1e-7]
title    = "Timescale (Month)"#"$\lambda^a (1/sec)$"
savename = "%sLbd_a_conv.png" % figpath

fig,ax,_ = viz.init_orthomap(1,1,bboxplot,figsize=(10,8))
ax       = viz.add_coast_grid(ax,bbox=bboxplot,fill_color="k")
pcm      = ax.pcolormesh(lon,lat,pv,transform=proj,vmin=vlms[0],vmax=vlms[1])
cb       = viz.hcbar(pcm,ax=ax)
#cb = fig.colorbar(pcm,ax=ax,fraction=0.025,pad=0.01,orientation='horizontal')
cb.set_label(title,fontsize=fsz_axis)
plt.savefig(savename,dpi=150,bbox_inches='tight')


#%% Visualize lbd_e

pv       = lbd_e.mean('ens').mean('mon') * dt
vlms     = [0,0.03]#[-.2,.2]#[-1e-7,1e-7]
title    = "$\lambda^e$ (psu/K/mon)"
savename = "%sLbd_e.png" % figpath

fig,ax,_ = viz.init_orthomap(1,1,bboxplot,figsize=(10,8))
ax       = viz.add_coast_grid(ax,bbox=bboxplot,fill_color="k")
if vlms is not None:
    pcm      = ax.pcolormesh(lon,lat,pv,transform=proj,vmin=vlms[0],vmax=vlms[1])
else:
    pcm      = ax.pcolormesh(lon,lat,pv,transform=proj)
cb       = viz.hcbar(pcm,ax=ax)
#cb = fig.colorbar(pcm,ax=ax,fraction=0.025,pad=0.01,orientation='horizontal')
cb.set_label(title,fontsize=fsz_axis)
plt.savefig(savename,dpi=150,bbox_inches='tight')


#%% Visualize Bowen ratio

#pv       = np.abs(B.isel(ens=0,mon=2))#.mean('ens').mean('mon') 
pv       = Bcorr.mean('ens').mean('mon') * mask.squeeze()
vlms     = [-1,1]#[-10,10]#None#[0,1]#[-.2,.2]#[-1e-7,1e-7]
title    = "Bowen Ratio"
savename = "%sBowen_Ratio.png" % figpath

fig,ax,_ = viz.init_orthomap(1,1,bboxplot,figsize=(10,8))
ax       = viz.add_coast_grid(ax,bbox=bboxplot,fill_color="k")
if vlms is not None:
    pcm      = ax.pcolormesh(lon,lat,pv,transform=proj,vmin=vlms[0],vmax=vlms[1])
else:
    pcm      = ax.pcolormesh(lon,lat,pv,transform=proj)
cb       = viz.hcbar(pcm,ax=ax)
#cb = fig.colorbar(pcm,ax=ax,fraction=0.025,pad=0.01,orientation='horizontal')
cb.set_label(title,fontsize=fsz_axis)
plt.savefig(savename,dpi=150,bbox_inches='tight')


#%% Check what is going on with Bowen Ratio


shflx = xr.open_dataset(raw_path+"CESM1LE_SHFLX_NAtl_Hist_SAvg.nc").load().SHFLX
lhflx = xr.open_dataset(raw_path+"CESM1LE_LHFLX_NAtl_Hist_SAvg.nc").load().LHFLX

invars   = [shflx,lhflx]
flxnames = ['SHFLX',"LHFLX"]
#%% Plot SHFLX, LHFLX
e  = 1
im = 2

for im in range(12):
    
    vlms      = [-5,5]
    pvs       = [iv.isel(ensemble=e,mon=im) for iv in invars]
    fig,axs,_ = viz.init_orthomap(1,2,bboxplot,figsize=(12,8))
    
    for ii in range(2):
        
        ax = axs[ii]
        ax       = viz.add_coast_grid(ax,bbox=bboxplot,fill_color="k")
        ax.set_title(flxnames[ii])
        if vlms is not None:
            pcm      = ax.pcolormesh(lon,lat,pvs[ii],transform=proj,cmap='cmo.balance',
                                     vmin=vlms[0],vmax=vlms[1])
        else:
            pcm      = ax.pcolormesh(lon,lat,pvs[ii],transform=proj)
        cb       = viz.hcbar(pcm,ax=ax)
        
    plt.suptitle("%s Monthly Mean Flux (Upwards Positive, W/m2, Ens=%02i)" % (mons3[im],e+1),fontsize=20,y=0.69)


    savename = "%sFlxCompare_ClimCycle_Ens%02i_mon%02i.png" % (figpath,e+1,im+1)
    plt.savefig(savename,dpi=150,bbox_inches='tight')
    
#%% Compute a B replacement factor (Note, I've moved this to calc_bowen_ratio)
# Delete this.

def interp_scycle(ts,thres=3,):
    
    ts       = np.abs(ts)
    idexceed = np.where(ts > thres)[0]
    for ii in idexceed:
        if ii + 1 > 11:
            iip1 = 0
        else:
            iip1 = 11
        ts[ii] = np.interp(1,[0,2],[ts[ii-1],ts[iip1]])
    return ts

Bcorr = xr.apply_ufunc(
    
    interp_scycle,  # Pass the function
    B,  # The inputs in order that is expected
    # Which dimensions to operate over for each argument...
    input_core_dims =[['mon'],],
    output_core_dims=[['mon'],],  # Output Dimension
    #exclude_dims=set(("",)),
    vectorize=True,  # True to loop over non-core dims
)


        
        

Bmasked = B * mask.squeeze()






