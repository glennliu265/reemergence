#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Visualize Ekman Forcing Output from calc_ekman_advetion_htr

Created on Mon Feb 12 19:59:52 2024

@author: gliu
"""

import numpy as np
import xarray as xr
import sys
import time
import matplotlib.pyplot as plt

#%% Import Custom Modules
amvpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/03_Scripts/" # amv module
scmpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/03_Scripts/stochmod/model/"

sys.path.append(amvpath)
sys.path.append(scmpath)

from amv import proc,viz
import scm
import amv.loaders as dl
import yo_box as ybx



#%% Load some files

figpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/02_Figures/20240217"

# Load Ekman Forcing
fp1  = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/proc/model_input/forcing/"
ncek = "CESM1_HTR_FULL_Qek_SSS_NAO_nomasklag1_nroll0_NAtl_EnsAvg.nc"


dsek = xr.open_dataset(fp1+ncek)



#%%

im = 0
n  = 0
scale = 99
xint = 2
yint = 2
mons3 = proc.get_monstr()
bbox = [-80,0,20,65]
#fig,ax,mdict=viz.init_orthomap(1,1,bbox)

fig,ax=viz.geosubplots(1,1,)
#proj = mdict['noProj']
ax = viz.add_coast_grid(ax,bbox=bbox,fill_color='lightgray')


# Select what to plot
x = dsek.lon.values[::xint]
y = dsek.lat.values[::yint]

U = dsek.Uek.isel(mon=im,mode=n).values[::yint,::xint]
V = dsek.Vek.isel(mon=im,mode=n).values[::yint,::xint]

Q = dsek.Qek.isel(mon=im,mode=n)

ax.quiver(x,y,U,V,scale=0.2)
pcm = ax.pcolormesh(dsek.lon,dsek.lat,Q,zorder=-1,cmap='cmo.balance',vmin=-15,vmax=15)
cb=fig.colorbar(pcm,ax=ax,fraction=0.025,pad=0.01)
cb.set_label("$Q_{ek}$ ($W/m^2$)")

ax.set_title("Ekman Forcing (colors) and Advection (vectors)\n %s | EOF Mode %i | CESM1 Historical Ens Avg." % (mons3[im],n+1))
savename = "%sQek_Advection_Pattern_CESM1_EnsAvg_Mon%02i_Mode%03i.png" % (figpath,im+1,n+1)
plt.savefig(savename,dpi=150,bbox_inches='tight')
