#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Check Whitening of Fprime

Created on Mon Jan 22 22:15:17 2024

@author: gliu
"""


import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import sys
from tqdm import tqdm
import copy

#%% Import Custom Modules
amvpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/03_Scripts/" # amv module
scmpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/03_Scripts/stochmod/model/"

sys.path.append(amvpath)
sys.path.append(scmpath)

from amv import proc,viz
import scm
import amv.loaders as dl
import yo_box as ybx


#%% General Variables/User Edits

# Path to Input Data
input_path = '/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/01_Data/model_input/'

figpath    = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/02_Figures/20240122/"
proc.makedir(figpath)


#%% Add in Fprime Std (takes awhile to load)
lonf =-30
latf = 50

# Load Fprime 
fname    = "Fprime_PIC_SLAB_rolln0.nc"
dsf      = xr.open_dataset(input_path+"../"+fname)#.Fprime.load()
dsf      = dsf.sel(lon=lonf+360,lat=latf,method='nearest')
#dsf      = proc.format_ds(dsf)

# Flip Longitude
#dsf = proc.format_ds(dsf)

# Compute Monthly variance
fprime_ori = dsf.Fprime.values
#dsmonvar = dsf.groupby('time.month').var('time')
#fprimestd = dsmonvar.values.transpose(2,1,0)

#%% Load SLAB SST at the point

ncts = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/01_Data/CESM_proc/TS_anom_PIC_SLAB.nc"
dsts = xr.open_dataset(ncts)
dspt = dsts.sel(lon=lonf+360,lat=latf,method='nearest')
sst_slab = dspt.TS.values

#tsmetrics_slab = scm.compute_sm_metrics([sst_slab,]) 

#%% Load Heat Flux Feedback

fnamehff = input_path+"damping/" + "PIC_SLAB_NHFLX_Damping_mode5_lag1_ensorem1.nc"
dshff = xr.open_dataset(fnamehff)
dshff = dshff.sel(lon=lonf,lat=latf,method='nearest')

lbd_a = dshff.lambda_qnet.values #[Month]

#%% Load Qunet

fnameqnet = input_path + "../CESM_proc/NHFLX_PIC_SLAB.nc"
dsq = xr.open_dataset(fnameqnet)
dsq      = dsq.sel(lon=lonf+360,lat=latf,method='nearest')

qnet = dsq.NHFLX.values # [year x mon]


#%% Compute Fprime in a few ways

nyr,nmon = qnet.shape

lbda_tile = np.tile(lbd_a,nyr)

# F'(t) = Qnet(t) + lbd_a(t) *T(t)
fprime_rolln0    = qnet.flatten() + lbda_tile * sst_slab

# F'(t) = Qnet(t) + lbd_a(t) *T(t-1)
fprime_rolln1    = qnet.flatten() + lbda_tile * np.roll(sst_slab,1)

# F'(t) = Qnet(t) + lbd_a(t-1) * T(t-1)
fprime_rolln1_dr = qnet.flatten() + np.roll(lbda_tile,1) * np.roll(sst_slab,1)


#%% Analysis Input:

finput = [qnet.flatten(),
          fprime_ori,
          fprime_rolln0,
          fprime_rolln1,
          fprime_rolln1_dr]

fnames = ["Qnet",
          "Fprime Original",
          "Fprime (No Roll)",
          "Fprime (Roll T)",
          "Fprime (Roll T and lbd)",
    ]

fmarkers = ["*","x","d","o","s"]
fcolors  = ["gray","k","red","goldenrod","blue"]
fls      = ["solid",'dotted','dashed','dashed','dashed'] 

nsmooth   = 50
finput_ds = [proc.deseason(ts).flatten() for ts in finput]


tsmetrics = scm.compute_sm_metrics(finput_ds,nsmooth=nsmooth,)

#%% Examine Monthly Variance of each forcing

nexps  = len(finput)
mons3 = proc.get_monstr(nletters=3)
fig,ax = viz.init_monplot(1,1,figsize=(6,3))

for ex in range(nexps):
    plotvar = np.sqrt(tsmetrics['monvars'][ex])
    ax.plot(mons3,plotvar,label=fnames[ex],ls=fls[ex],marker=fmarkers[ex],c=fcolors[ex],lw=2.5)
ax.legend()

#%% Check the spectra

dtplot = 3600*24*30
locfn,loctitle=proc.make_locstring(lonf,latf)
plotyrs = [100,50,25,10,5,2,1]
xtks    = 1/(np.array(plotyrs)*12)

# freqs   = obsmetrics['freqs']
# specs   = obsmetrics['specs']

fig,ax = plt.subplots(1,1,constrained_layout=1,figsize=(8,3))
for ex in range(nexps):
    
    lab      = "%s (var=%.2f)" % (fnames[ex],np.var(finput[ex]))
    plotfreq = tsmetrics['freqs'][ex] 
    plotspec = tsmetrics['specs'][ex]
    
    ax.plot(plotfreq*dtplot,plotspec/dtplot,label=lab,c=fcolors[ex])




ax.set_xticks(xtks,labels=plotyrs)
ax.set_xlim([xtks[0],xtks[-1]])
ax.legend(ncol=2)
ax.set_xlabel("Period (Years)")
ax.set_ylabel("$psu^2$/cpy")
ax.set_title("SSS Power Spectra @ %s" % (loctitle))
savename = "%sQnet_Fprime_Spectra_PIC_SLAB_CESM1.png" % (figpath)
plt.savefig(savename,dpi=150,bbox_inches='tight')
ax.set_ylim([0,10000])
#ax.set_ylim

#%% Save output
outpath  = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/ptdata/lon330_lat50/Fprime_rolltest/"

savename = "RollTest_Fprime.npz"
np.savez(savename,**{
    "forcings":finput,
    "names":fnames,
    "metris":tsmetrics,
    "markers":fmarkers,
    "colors":fcolors,
    "ls":fls,
    "nsmooth":nsmooth,
    },allow_pickle=True)










