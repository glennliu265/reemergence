#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Make a plot of the mean conditions from SST/SSS in observations

Created on Tue Oct  8 11:31:39 2024

@author: gliu
"""

import numpy as np
import xarray as xr
import sys
import time
import matplotlib.pyplot as plt
import os
import cartopy.crs as ccrs
from tqdm import tqdm

import matplotlib as mpl

# ----------------------------------
#%% Import custom modules and paths
# ----------------------------------

# Indicate the Machine!
machine = "Astraeus"

# First Load the Parameter File
cwd = os.getcwd()
sys.path.append(cwd+ "/..")
sys.path.append("../")
import reemergence_params as rparams

# Paths and Load Modules
pathdict   = rparams.machine_paths[machine]

sys.path.append(pathdict['amvpath'])
sys.path.append(pathdict['scmpath'])
from amv import proc,viz
import scm
import amv.loaders as dl
import yo_box as ybx

# Set needed paths
figpath     = pathdict['figpath']
proc.makedir(figpath)
input_path  = pathdict['input_path']
output_path = pathdict['output_path']
procpath    = pathdict['procpath']


#%% Load some Datasets

# Load EN4 Salinity
en4path    = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/01_Data/EN4/proc/"
en4nc      = "EN4_concatenate_1900to2021_lon-80to00_lat00to65.nc"
ds_en4     = xr.open_dataset(en4path+en4nc).load()
sbar_en4   = ds_en4.salinity.mean('time')
tbar_en4   = ds_en4.temperature.mean('time')

# Load GLORYS v12
gloryspath  = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/01_Data/glorys12v1/"
glorysnc    = "glorys12v1_so_NAtl_1993_2019_merge.nc"
ds_glorys   = xr.open_dataset(gloryspath + glorysnc).load()
sbar_glorys = ds_glorys.so.mean('time')

# Load HadISST 

#%% Visualize Mean temperature and salinity from EN4
bboxplot    = [-80,0,0,65]
mpl.rcParams['font.family'] = 'Avenir'
proj        = ccrs.PlateCarree()

fsz_tick                    = 18
fsz_axis                    = 22
fsz_title                   = 28


#%%
pmesh         = False

cints_sssmean = np.arange(31,37.6,0.2)
cints_sstmean = np.arange(273,305,1)

# Initialize Plot
fig,ax,_    = viz.init_orthomap(1,1,bboxplot,figsize=(26,12))
ax          = viz.add_coast_grid(ax,bbox=bboxplot,fill_color="lightgray",fontsize=fsz_tick)

# Plot Mean SST (Colors)
plotvar = tbar_en4 #ds_sst.SST.mean('ens').mean('mon').transpose('lat','lon') * mask_apply
if pmesh:
    pcm     = ax.pcolormesh(plotvar.lon,plotvar.lat,plotvar,transform=proj,zorder=-1,
                linewidths=1.5,cmap="RdYlBu_r",vmin=280,vmax=300)
else:
    
    pcm     = ax.contourf(plotvar.lon,plotvar.lat,plotvar,transform=proj,zorder=-1,
                cmap="RdYlBu_r",levels=cints_sstmean)
cb = viz.hcbar(pcm,ax=ax,fraction=0.045)
cb.set_label("SST ($\degree C$)",fontsize=fsz_axis)
cb.ax.tick_params(labelsize=fsz_tick)

# Plot Mean SSS (Contours)
plotvar = sbar_en4 #ds_sss.SSS.mean('ens').mean('mon').transpose('lat','lon') * mask_reg
cl = ax.contour(plotvar.lon,plotvar.lat,plotvar,transform=proj,
            linewidths=0.75,colors="darkviolet",levels=cints_sssmean,linestyles='dashed')
ax.clabel(cl,fontsize=fsz_tick)

ax.set_title("Average Temperature and Salinity in EN4\n 1900 to 2021",
             fontsize=fsz_title)

savename = "%sEN4_Mean_SST_SSS_1900ti2021.png" % (figpath)
plt.savefig(savename,dpi=150,bbox_inches='tight')
#ax.set_title("CESM1 Historical Ens. Avg., Ann. Mean",fontsize=fsz_title)








#%%



