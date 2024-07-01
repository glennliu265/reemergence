#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Examine the Stochastic Model vs. CESM1 Results by Cluster

Copied upper section of compare_reemergence_maps

Created on Mon Jul  1 11:59:14 2024

@author: gliu
"""


import xarray as xr
import numpy as np
import matplotlib as mpl
import cartopy.crs as ccrs

import matplotlib.pyplot as plt
import sys
import glob
import os

import tqdm
import time

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

#%% Indicate the experiments/ load the files 

# Variable Information
vnames      = ["SST","SSS"]
vcolors     = ["hotpink","navy"]
vunits      = ["$\degree C$","$psu$"]
vmarkers    = ['o','x']

# CESM NetCDFs
ncs_cesm    = ["CESM1_1920to2005_SSTACF_lag00to60_ALL_ensALL.nc",
            "CESM1_1920to2005_SSSACF_lag00to60_ALL_ensALL.nc"]

# Note I might need to rerun all of this...
ncs_sm      = ["SM_SST_EOF_LbddCorr_Rerun_SST_autocorrelation_thresALL_lag00to60.nc",
            "SM_SSS_EOF_LbddCorr_Rerun_lbdE_neg_SSS_autocorrelation_thresALL_lag00to60.nc"]

compare_name = "CESMvSM_PaperOutline"

# Load Pointwise ACFs for each runs
st          = time.time()
ds_cesm     = []
ds_sm       = []
for vv in range(2):
    
    # Load CESM (ens, lon, lat, mons, thres, lags)
    ds = xr.open_dataset(procpath + ncs_cesm[vv]).acf.load()
    ds_cesm.append(ds.copy())
    
    # Load SM
    ds = xr.open_dataset(procpath + ncs_sm[vv])[vnames[vv]].load()
    ds_sm.append(ds.copy())
print("Loaded output in %.2fs" % (time.time()-st))


acf_lw      = 2.5

ds_all      = ds_cesm + ds_sm
lags        = ds_all[0].lags.data
expnames    = ["CESM SST","CESM SSS","SM SST","SM SSS"]
expcols     = ["firebrick","navy","lightsalmon","cornflowerblue"]
expls       = ["solid","solid","dashed","dashed"]
expmarkers  = ["o","d","+","x"]


#%% Load the cluster_output

clustpath = procpath + "clustering/"
clustnc   = "CESM1_Cluster_Labels_AllMons_nclust06.nc"
dsclust   = xr.open_dataset(clustpath + clustnc).load()


# Cluster Options
nclusts    = len(dsclust.cluster)
cmap_in    = mpl.cm.get_cmap("PuOr",nclusts)



#%% Plotting options

bboxplot                    = [-80,0,20,65]
bboxplot_clust              = [-80,0,0,65]
mpl.rcParams['font.family'] = 'Avenir'
mons3                       = proc.get_monstr(nletters=3)
fsz_tick                    = 18
fsz_axis                    = 18
fsz_title                   = 24

proj                        = ccrs.PlateCarree()


#%% First Plot a reference map for the clusters

fig,axs,_  = viz.init_orthomap(1,2,bboxplot_clust,figsize=(16,8))
for vv in range(2):
    ax     = axs[vv]
    pv     = dsclust.isel(varname=vv,).cluster_maps
    ax     = viz.add_coast_grid(ax,bbox=bboxplot_clust,fill_color="lightgray")
    
    pcm    = ax.pcolormesh(pv.lon,pv.lat,pv+1,transform=proj,cmap=cmap_in)
    cb     = viz.hcbar(pcm,ax=ax)
    cb.set_label("Cluster Number",fontsize=fsz_axis)
    ax.set_title(vnames[vv],fontsize=fsz_title)
    
#%% Plot Reference ACFs

im      = 1
fig,axs = plt.subplots(2,1)

for vv in range(2):
    ax  = axs[vv]
    
    clustacf  = dsclust.isel(varname=vv,mons=im).cluster_centers
    
    for nn in range(nclusts):
        pv = clustacf.isel(cluster=nn)
        ax.plot(pv.lag,pv,label="Cluster " + str(pv.cluster.data))
    ax.legend(ncol=3)
    ax.set_title(vnames[vv])

#%% Select regions. Start with SST


iclust       = 3# Cluster Index
vv           = 0 # Variable Index
im           = 1 # Month Index



# Indicate the ds
if vv == 0:
    
    cesm_in = ds_all[0]
    sm_in   = ds_all[2]
    
    
elif vv == 1:
    cesm_in = ds_all[1]
    sm_in   = ds_all[3]
    

# Get Cluster Maps and Resize to CESM/SM Output
cluster_maps   = dsclust.cluster_maps #.data
cluster_maps,cesm_in,sm_in = proc.resize_ds([cluster_maps,cesm_in,sm_in])

# Make the mask based on the index
clust_mask     = cluster_maps[vv,:,:] == iclust
clust_mask     = xr.where(clust_mask,clust_mask,np.nan)

# Apply the mask

# cesm in shape: (42, 65, 48, 12, 1, 61) --> # [ens x lat x lon x lag]
cesm_sel = cesm_in.isel(mons=im).squeeze() * clust_mask#.data[None,:,:,None]

# sm shape: (65, 48, 12, 1, 61) --> # ('lon', 'lat', 'lags')
sm_sel   = sm_in.isel(mons=im).squeeze()  * clust_mask

# Take cluster area average (not area - weighted... 0)
cesm_cavg = cesm_sel.mean('lat').mean('lon')#.mean('ens') # :ags
sm_cavg   = sm_sel.mean('lat').mean('lon')


# Plot the difference
xtks        = lags[::3]
fig,ax      = plt.subplots(1,1,figsize=(10.5,4.5),constrained_layout=True)
ax,_        = viz.init_acplot(im,xtks,lags,ax=ax,title="")


for e in range(42):
    ax.plot(lags,cesm_cavg.isel(ens=e),label="",alpha=0.1,c='gray')
mu = cesm_cavg.mean('ens')
ax.plot(lags,mu,c='k',label="CESM Ens. Avg")
ax.plot(lags,sm_cavg,c="cornflowerblue",label="Stochastic Model")
ax.legend()

ax.set_title("%s %s ACF Stochastic Model v. CESM for Cluster %i" % (mons3[im],
                                                                    vnames[vv],
                                                                    iclust+1),fontsize=20)

outname = "%sCESM1vSM_%s_clustavg%0i_mon%02i.png" % (figpath,vnames[vv],iclust+1,im+1,)
plt.savefig(outname,dpi=150,bbox_inches='tight')






    
    
    
    
    
    



