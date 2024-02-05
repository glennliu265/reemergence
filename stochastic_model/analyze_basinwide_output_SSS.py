#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Analyze output produced by run_SSS_basinwide.py
Copied upper section of that script on Feb 1 Thu

Created on Thu Feb  1 22:48:44 2024

@author: gliu
"""



import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import sys
from tqdm import tqdm
import copy
import glob

#%% Import Custom Modules
amvpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/03_Scripts/" # amv module
scmpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/03_Scripts/stochmod/model/"

sys.path.append(amvpath)
sys.path.append(scmpath)

from amv import proc,viz
import scm
import amv.loaders as dl
import yo_box as ybx


#%% General Variables/User Edits, Set Experiment Name

expnames_long = ["No Shift","Half Shift","Shift Forcing and MLD"]
expnames      = ["Test_Td0.1_SPG_noroll","Test_Td0.1_SPG_froll1-mroll1","Test_Td0.1_SPG_allroll1_halfmode",]

# expparams   = {
#     'bbox_sim'      : [-65,0,45,65],
#     'nyrs'          : 1000,
#     'runids'        : ["test%02i" % i for i in np.arange(0,11,1)],
#     'PRECTOT'       : "CESM1_HTR_FULL_PRECTOT_NAtl_EnsAvg.nc",
#     'LHFLX'         : "CESM1_HTR_FULL_Eprime_nroll0_NAtl_EnsAvg.nc",
#     'h'             : "CESM1_HTR_FULL_HMXL_NAtl_EnsAvg.nc",
#     'lbd_d'         : 0.10,
#     'Sbar'          : "CESM1_HTR_FULL_Sbar_NAtl_EnsAvg.nc",
#     'beta'          : None, # If None, just compute entrainment damping
#     'kprev'         : "CESM1_HTR_FULL_kprev_NAtl_EnsAvg.nc",
#     'lbd_a'         : None, # NEEDS TO BE ALREADY CONVERTED TO 1/Mon !!!
#     'froll'         : 0,
#     'mroll'         : 0,
#     'droll'         : 0,
#     'halfmode'      : False,
#     }

# expparams   = {
#     'bbox_sim'      : [-65,0,45,65],
#     'nyrs'          : 1000,
#     'runids'        : ["test%02i" % i for i in np.arange(0,11,1)],
#     'PRECTOT'       : "CESM1_HTR_FULL_PRECTOT_NAtl_EnsAvg.nc",
#     'LHFLX'         : "CESM1_HTR_FULL_Eprime_nroll0_NAtl_EnsAvg.nc",
#     'h'             : "CESM1_HTR_FULL_HMXL_NAtl_EnsAvg.nc",
#     'lbd_d'         : 0.10,
#     'Sbar'          : "CESM1_HTR_FULL_Sbar_NAtl_EnsAvg.nc",
#     'beta'          : None, # If None, just compute entrainment damping
#     'kprev'         : "CESM1_HTR_FULL_kprev_NAtl_EnsAvg.nc",
#     'lbd_a'         : None, # NEEDS TO BE ALREADY CONVERTED TO 1/Mon !!!
#     'froll'         : 0,
#     'mroll'         : 0,
#     'droll'         : 0,
#     'halfmode'      : False,
#     }
# Parameters needed from expparams
bbox_sim = [-65,0,45,65]
bbox_plot = [-65,0,45,65]
# Constants
dt  = 3600*24*30 # Timestep [s]
cp  = 3850       # 
rho = 1026       # Density [kg/m3]
B   = 0.2        # Bowen Ratio, from Frankignoul et al 1998
L   = 2.5e6      # Specific Heat of Evaporation [J/kg], from SSS model document

debug = False

#%% Load output

output_path = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/sm_experiments/"
figpath     = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/01_hfdamping/02_Figures/20240202/"
proc.makedir(figpath)

nexps  = len(expnames)
ds_out = [] 
for ex in range(nexps):
    expname = expnames[ex]
    
    expdir       = output_path + expname + "/Output/"
    nclist       = glob.glob(expdir +"*.nc")
    nclist.sort()
    print(nclist)
    
    ds_all = xr.open_mfdataset(nclist,concat_dim="run",combine='nested').load()
    ds_out.append(ds_all)
    
    

#%% Load CESM1 Output for SSS

ncpath  = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/01_Data/CESM_proc/"
ncname  = "SSS_FULL_HTR_lon-80to0_lat0to65_DTEnsAvg.nc"
ds_cesm = xr.open_dataset(ncpath+ncname).squeeze()

ds_cesm = proc.sel_region_xr(ds_cesm,bbox_sim)

ds_cesm = proc.fix_febstart(ds_cesm)
ds_cesm = ds_cesm.sel(time=slice('1920-01-01','2005-12-31'))
#%% Load some dims for plotting
ds   = ds_out[0]
lon  = ds.lon.values
lat  = ds.lat.values
tsim = ds.time.values


latf=50
lonf=-30
klon,klat = proc.find_latlon(lonf,latf,lon,lat)
locfn,loctitle=proc.make_locstring(lonf,latf)

#%% Compare Variance Across Simulations

import matplotlib as mpl
mpl.rcParams['mathtext.fontset'] = 'stix'#'custom' 
mpl.rcParams['font.family'] = 'STIXGeneral'#'Courier'#'STIXGeneral'

plotdiff = False

vlms = [0,0.25]
fig,axs = viz.geosubplots(2,2,figsize=(10,5),)

for a,ax in enumerate(axs.flatten()):
    
    blabels=[0,0,0,0]
    if a%2 == 0:
        blabels[0] = 1
    if a>1:
        blabels[-1] =1 
    # Formatting
    ax=viz.add_coast_grid(ax,bbox=bbox_plot,fill_color="darkgray",blabels=blabels)
    
    # PLotting
    if a > 0:
        
        ds_in   = ds_out[a-1]
        if plotdiff:
            plotvar_sm = ds_in.std('time').mean('run')
            plotvar_ce = ds_cesm.std('time').mean('ensemble')
            plotvar = plotvar_sm - plotvar_ce
            vlms = [-.1,.1]
            
            cmap = 'RdBu_r'
        else:
            plotvar = ds_in.std('time').mean('run')
        title   = "Stochastic Model (%s) - CESM1" % expnames_long[a-1] 
    else:
        ds_in = ds_cesm
        plotvar = ds_in.std('time').mean('ensemble')
        title   = "CESM1 Historical"
        cmap = 'cmo.haline'
    pcm = ax.pcolormesh(lon,lat,plotvar.SSS,vmin=vlms[0],vmax=vlms[-1],cmap=cmap)
    fig.colorbar(pcm,ax=ax,orientation='horizontal',fraction=0.05,pad=0.01)
    ax.set_title(title)
    
plt.suptitle(r"SSS Std. Dev. (psu)")

savename = "%sSSS_Variance_ShiftComparison_diff%0i.png" % (figpath,plotdiff)
plt.savefig(savename,dpi=150)
plt.show()


#%% Merge DS for comparison

nrun,ntime,nlat,nlon  = ds_out[0].SSS.shape
nens,ntimec,nlat,nlon = ds_cesm.SSS.shape

smflatten   = [ds.SSS.values.reshape(nrun*ntime,nlat,nlon) for ds in ds_out]
cesmflatten = ds_cesm.SSS.values.reshape(nens*ntimec,nlat,nlon)
cesmflatten[np.isnan(cesmflatten)] = 0

dsloop     = smflatten + [cesmflatten,]
nameloops  = ["Stochastic Model (%s)" % a for a in expnames_long] + ["CESM1 Historical",]
loopcolors = ["blue","orange","violet","k"]

dspt = [ds[:,klat,klon] for ds in dsloop]
tsm  = scm.compute_sm_metrics(dspt)

#%% Compute T2 (winter)





#%% Plot ACF

lags   = np.arange(37)
xtks   = np.arange(0,37,3)
kmonth = 1
fig,ax = viz.init_acplot(kmonth,xtks,lags)

for ii in range(4):
    ax.plot(lags,tsm['acfs'][kmonth][ii],label=nameloops[ii],lw=3.5,c=loopcolors[ii])
    
ax.legend()
plt.show()

#%% Plot Monvar
mons3 = proc.get_monstr(nletters=3)

fig,ax = viz.init_monplot(1,1)

for ii in range(4):
    ax.plot(mons3,tsm['monvars'][ii],label=nameloops[ii],lw=3.5,c=loopcolors[ii])
    
ax.legend()
plt.show()


#%%
#%%
# Below is still scrap

#%%
#%%

#%% Set Paths

# Path to Experiment Data
output_path = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/sm_experiments/"
expdir       = output_path + expname + "/"

# Named Paths
outpath_sm = expdir + "Output/" # path to stochastic model output
figpath    = expdir + "Figures/" # path to figures

proc.makedir(expdir + "Input")
proc.makedir(outpath_sm)
proc.makedir(expdir + "Metrics")
proc.makedir(figpath)




#%%



#%%

# Paths and Experiment
input_path = "/stormtrack/data3/glliu/01_Data/02_AMV_Project/03_reemergence/proc/model_input/"
output_path= "/stormtrack/data3/glliu/01_Data/02_AMV_Project/03_reemergence/sm_experiments/"

expname    = "Test_Td0.1_SPG"

expparams   = {
    'bbox_sim'      : [-65,0,45,65],
    'nyrs'          : 1000,
    'runids'        : ["test%02i" % i for i in np.arange(1,11,1)],
    'PRECTOT'       : "CESM1_HTR_FULL_PRECTOT_NAtl_EnsAvg.nc",
    'LHFLX'         : "CESM1_HTR_FULL_Eprime_nroll0_NAtl_EnsAvg.nc",
    'h'             : "CESM1_HTR_FULL_HMXL_NAtl_EnsAvg.nc",
    'lbd_d'         : 0.10,
    'Sbar'          : "CESM1_HTR_FULL_Sbar_NAtl_EnsAvg.nc",
    'beta'          : None, # If None, just compute entrainment damping
    'kprev'         : "CESM1_HTR_FULL_kprev_NAtl_EnsAvg.nc",
    'lbd_a'         : None, # NEEDS TO BE ALREADY CONVERTED TO 1/Mon !!!
    }

# Constants
dt  = 3600*24*30 # Timestep [s]
cp  = 3850       # 
rho = 1026       # Density [kg/m3]
B   = 0.2        # Bowen Ratio, from Frankignoul et al 1998
L   = 2.5e6      # Specific Heat of Evaporation [J/kg], from SSS model document

debug = False


