#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Using continuity (from Buckley et al. 2014 and 2015 papers), compute the vertical Ekman velocities

Created on Thu Mar 20 13:49:07 2025

@author: gliu
"""


import numpy as np
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import sys
import cmocean
import xarray as xr
import time
from   tqdm import tqdm
import matplotlib as mpl

#%% Import modules

stormtrack    = 0
if stormtrack == 0:
    projpath   = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/"
    datpath     = projpath + '01_Data/model_output/'
    outpathdat  = datpath + '/proc/'
    figpath     = projpath + "02_Figures/20240207"
   
    lipath  = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/01_Data/landicemask_enssum.npy"
    rawpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/proc/CESM1/NATL_proc/"
    
    sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/03_Scripts/stochmod/model/")
    sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/03_Scripts/")
    
    # Path of model input
    outpath     = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/proc/model_input/forcing/"
    outpathdat  = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/proc/CESM1/NATL_proc/"
    
elif stormtrack == 1:
    #datpath     = "/stormtrack/data3/glliu/01_Data/02_AMV_Project/02_stochmod/Model_Data/model_output/"
    #rawpath     = "/stormtrack/data3/glliu/01_Data/02_AMV_Project/02_stochmod/Model_Data/model_input/"
    rawpath     = "/stormtrack/data3/glliu/01_Data/02_AMV_Project/03_reemergence/proc/CESM1/NATL_proc/"
    datpath     = rawpath
    outpathdat  = datpath + '/proc/'
    
    sys.path.append("/home/glliu/00_Scripts/01_Projects/00_Commons/")
    sys.path.append("/home/glliu/00_Scripts/01_Projects/01_AMV/02_stochmod/stochmod/model/")
    figpath     = "/home/glliu/02_Figures/00_Scrap/"
    
    outpath     = "/stormtrack/data3/glliu/01_Data/02_AMV_Project/03_reemergence/proc/model_input/forcing/"

from amv import proc,viz
import scm
import tbx

proc.makedir(figpath)

#%% User Edits

# Load Ekman Advection Files
output_path_uek     = outpathdat 
savename_uek        = "%sCESM1LE_uek_NAtl_19200101_20050101_bilinear.nc" % (output_path_uek)
ds                  = xr.open_dataset(savename_uek).load()

vek                 = ds.v_ek
uek                 = ds.u_ek

# Wind Stress Information (for reference)
tauxnc              = "CESM1LE_TAUX_NAtl_19200101_20050101_bilinear.nc"
tauync              = "CESM1LE_TAUY_NAtl_19200101_20050101_bilinear.nc"
dstaux                = xr.open_dataset(output_path_uek + tauxnc).load() # (ensemble: 42, time: 1032, lat: 96, lon: 89)
dstauy                = xr.open_dataset(output_path_uek + tauync).load()

# Convert stress from stress on OCN on ATM --> ATM on OCN
taux                = dstaux.TAUX * -1
tauy                = dstauy.TAUY * -1

# Load Mixed-Layer gradients
vnames              = ["TEMP","SALT"]
path3d              = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/proc/CESM1/NATL_proc/ocn_var_3d/"
#salt_mld_nc         =  "CESM1_HTR_SALT_MLD_Gradient.nc"
mld_dz_nc           = path3d + "CESM1_HTR_%s_MLD_Gradient.nc"
ds_3d_all           = [xr.open_dataset(mld_dz_nc % vnames[vv]).load() for vv in range(2)]#xr.open_dataset(path3d + salt_mld_nc)


# Convert to 


# Get Time Mean Values
uek_mean    = uek.mean('time')
vek_mean    = vek.mean('time')
taux_mean   = taux.mean('time')
tauy_mean   = tauy.mean('time')


#%% Compute Vertical Pumping

wek         = uek + vek

wek_mean    = wek.mean('time')

#%% Plot Settings

bboxplot                    = [-80,0,20,65]
mpl.rcParams['font.family'] = 'Avenir'
mons3                       = proc.get_monstr(nletters=3)
fsz_tick                    = 18
fsz_axis                    = 22
fsz_title                   = 28

proj                        = ccrs.PlateCarree()



#%% Do a quick visualization just for reference

qint_ek  = 1
qint_tau = 2
iens     = 0
tauscale = 2
ekscale  = .005

bboxplot = [-80,0,10,70]
# Initialize Plot and Map
fig,ax = viz.init_regplot(bboxin=bboxplot)


plotvar = wek_mean.isel(ens=iens) * -1
pcm = ax.pcolormesh(plotvar.lon,plotvar.lat,plotvar,
                    cmap='cmo.balance',vmin=-0.0002,vmax=0.0002,
                    transform=proj)
cb  = viz.hcbar(pcm,ax=ax,fraction=0.045)

# First, plot the Vek and Uek
qint    = qint_ek
plotu   = uek_mean.isel(ens=iens) * -1
plotv   = vek_mean.isel(ens=iens) * -1
umod    = np.sqrt(plotu**2 + plotv **2)
lon     = plotu.lon.data
lat     = plotu.lat.data
# qv      = ax.quiver(lon[::qint],lat[::qint],
#                     plotu.data[::qint,::qint],plotv.data[::qint,::qint],
#                     scale=ekscale,transform=proj,color='darkblue')
qv      = ax.streamplot(lon[::qint],lat[::qint],
                    plotu.data[::qint,::qint],plotv.data[::qint,::qint],
                    color=umod.data[::qint,::qint],
                    density=2,
                    transform=proj)



# Colors are wek


#qk = ax.quiverkey(qv,.0,1,0.1,r"0.1 $\frac{m}{s}$",fontproperties=dict(size=10))


# # Plot Tau
qint    = qint_tau
plotu   = taux_mean.isel(ensemble=iens)
plotv   = tauy_mean.isel(ensemble=iens)
lon     = plotu.lon.data
lat     = plotu.lat.data
qv      = ax.quiver(lon[::qint],lat[::qint],
                    plotu.data[::qint,::qint],plotv.data[::qint,::qint],
                    scale=tauscale,transform=proj,color='gray')

#qk = ax.quiverkey(qv,.0,1,0.1,r"0.1 $\frac{m}{s}$",fontproperties=dict(size=10))

#fig,ax,_    = viz.init_orthomap(1,1,bboxplot,figsize=(24,6.5))
#x          = viz.add_coast_grid(ax,bbox=bboxplot,fill_color="lightgray",fontsize=fsz_tick)


#%% Load in Detrain gradients (from compute_dz_ens)

vnames = ['TEMP','SALT']
path3d  = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/proc/CESM1/NATL_proc/ocn_var_3d/"


ds3d = []
for vv in range(2):
    vname   = vnames[vv]
    nc3d    = "%sCESM1_HTR_%s_Detrain_Gradients.nc" % (path3d,vname)
    ds3d.append(xr.open_dataset(nc3d).load())
    









