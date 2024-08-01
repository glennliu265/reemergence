#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Visualize Persistence Timescale (pointwise) for the stochastic model compared to CESM

Copied upper section from [viz_SST_SSS_coupling.py]

Created on Tue Apr 30 17:48:21 2024

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

#%% 

# Indicate files containing ACFs
sst_expname = "SM_SST_EOF_LbddCorr_Rerun_SST_autocorrelation_thresALL_lag00to60.nc"
sss_expname = "SM_SSS_EOF_LbddCorr_Rerun_lbdE_SSS_autocorrelation_thresALL_lag00to60.nc"
cesm_name   = "CESM1_1920to2005_%sACF_lag00to60_ALL_ensALL.nc"

# Plotting Parameters



#%% Load the files (computed by pointwise_autocorrelation_smoutput) # Took 30.12s

st       = time.time()
sm_sss   = xr.open_dataset(procpath+sss_expname).SSS.load()        # (lon: 65, lat: 48, mons: 12, thres: 1, lags: 61)
sm_sst   = xr.open_dataset(procpath+sst_expname).SST.load()

cesm_sst = xr.open_dataset(procpath+cesm_name % "SST").acf.load()  #  (lon: 89, lat: 96, mons: 12, thres: 1, ens: 42, lags: 61)
cesm_sss = xr.open_dataset(procpath+cesm_name % "SSS").acf.load()
print("Loaded all data in %.2fs" % (time.time()-st))

#%% Do some Preprocessing (slice to same size, etc)
acfs_in = [cesm_sst,sm_sst,cesm_sss,sm_sss]

acfs_in_rsz = proc.resize_ds(acfs_in)

explabs = ["SST (CESM)","SST (Stochastic Model)","SSS (CESM)","SSS (Stochastic Model)"]

#%% Load BSF and Ice Mask (copied from compare_detrainment_damping)

bsf      = dl.load_bsf()

# Load Land Ice Mask
icemask  = xr.open_dataset(input_path + "masks/CESM1LE_HTR_limask_pacificmask_enssum_lon-90to20_lat0to90.nc")

# Resize
bsf,icemask,_    = proc.resize_ds([bsf,icemask,acfs_in_rsz[0]])
bsf_savg = proc.calc_savg_mon(bsf)

#
mask = icemask.MASK.squeeze()
mask_plot = xr.where(np.isnan(mask),0,mask)#mask.copy()

mask_apply = icemask.MASK.squeeze().values
#mask_plot[np.isnan(mask)] = 0


#%% Compute T2


t2_all  = []
for ex in range(4):
    t2 = proc.calc_T2(acfs_in_rsz[ex],axis=-1,ds=True)
    t2_all.append(t2)
    
    
# cesm size : (42, 65, 48, 12, 1)
# sm size   : (65, 48, 12, 1)
    



#%% Plotting Params
mpl.rcParams['font.family'] = 'JetBrains Mono'
bboxplot                    = [-80,0,20,65]
proj                        = ccrs.PlateCarree()
lon                         = sm_sst.lon.values
lat                         = sm_sst.lat.values
mons3                       = proc.get_monstr()



#%% Visualize T2 (wintertime, ens avg)

imons           = [11,0,1]
vlms            = [0,30]
use_contour     = True


expvlms = ([0,18],[0,18],[0,30],[0,50])

t2_plot = [t2_all[0][:,:,:,imons,:].mean(0).squeeze().mean(-1),
           t2_all[1][:,:,imons,:].squeeze().mean(-1),
           t2_all[2][:,:,:,imons,:].mean(0).squeeze().mean(-1),
           t2_all[3][:,:,imons,:].squeeze().mean(-1),
           ]


fig,axs,_ = viz.init_orthomap(2,2,bboxplot,figsize=(12,10))

for a in range(4):
    
    ax      = axs.flatten()[a]
    ax      = viz.add_coast_grid(ax,bbox=bboxplot,fill_color="k")
    plotvar = t2_plot[a].T
    
    vlms = expvlms[a]
    
    if vlms is not None:
        if use_contour:
            cints = np.linspace(vlms[0],vlms[1],9)
            pcm   = ax.contourf(lon,lat,plotvar,transform=proj,levels=cints,extend='both')
            cl    = ax.contour(lon,lat,plotvar,transform=proj,levels=cints,colors="k",linewidths=0.75)
            ax.clabel(cl)
        else:
            
            pcm = ax.pcolormesh(lon,lat,plotvar,transform=proj,vmin=vlms[0],vmax=vlms[1])
    else:
        pcm = ax.pcolormesh(lon,lat,plotvar,transform=proj)
        
    ax.contour(lon,lat,mask_plot,colors="w",linewidths=0.75,transform=proj,levels=[0,1])
    
    fig.colorbar(pcm,ax=ax,orientation='horizontal',pad=0.01,fraction=0.045)
    ax.set_title(explabs[a])
    
    

plt.suptitle("Wintertime Mean Persistence Timescale ($T^2$, months)")
savename = "%sPointwise_WinterTime_T2_CESM1_v_SM_Paper_Outline_Draft.png" % (figpath)
plt.savefig(savename)

#%% Check pattern Correlation

t2_calc = [t2 * mask_apply.T for t2 in t2_plot]

patcorsst = proc.patterncorr(t2_calc[0],t2_calc[1]) 
patcorsss = proc.patterncorr(t2_calc[2],t2_calc[3]) 
print("SST Pattern Correlation is %.3f" % patcorsst)
print("SSS Pattern Correlation is %.3f" % patcorsss)

# Idea for Null Hypothesis >> Scramble Non NaN points, and see distribution

def scramble_patcorr(map1,map2,mciter=10000):
    # Clone of patterncorr, but repeat after shuffling data
    # From Taylor 2001,Eqn. 1, Ignore Area Weights
    # Calculate pattern correation between two 2d variables (lat x lon)
    
    # Get Non NaN values, Flatten, Array Size
    map1ok = map1.copy()
    map1ok = map1ok[~np.isnan(map1ok)].flatten()
    map2ok = map2.copy()
    map2ok = map2ok[~np.isnan(map2ok)].flatten()
    N      = len(map1ok)
    
    
    # Anomalize (remove spatial mean and calc spatial stdev)
    map1a  = map1ok - map1ok.mean()
    map2a  = map2ok - map2ok.mean()
    std1   = np.std(map1ok)
    std2   = np.std(map2ok)
    
    r_sim = []
    for mc in tqdm.tqdm(range(mciter)): # Shuffle pixels and recompute pattern corr
        map1a_shuffle = np.random.choice(map1a,size=N,replace=False)
        map2a_shuffle = np.random.choice(map2a,size=N,replace=False)
        #std1   = np.std(map1ok)
        #std2   = np.std(map2ok)
        R             = 1/N*np.sum(map1a_shuffle*map2a_shuffle)/(std1*std2)
        r_sim.append(R)
    
    return r_sim

r_sim_sst = scramble_patcorr(t2_calc[0],t2_calc[1])

plt.scatter(t2_calc[0].flatten(),t2_calc[1].flatten())
#%%

# if vlms is not None:
#     fig.colorbar(pcm,ax=axs.flatten(),orientation='horizontal',pad=0.01,fraction=0.035)
    



