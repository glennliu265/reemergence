#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Get Salinity data from observation based datasets

Created on Thu Jan 18 11:57:25 2024

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

#%% Use Edits

# Indicate which point you want to take
lonf = -30
latf = 50
locfn,loctitle=proc.make_locstring(lonf,latf)
locfn360,_ = proc.make_locstring(lonf+360,latf)

# Figure and Output Path
figpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/02_Figures/20240119/"
proc.makedir(figpath)
outpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/ptdata/%s/" % (locfn360)
proc.makedir(outpath)
# ---

# Load EN4 Dataset (5.022 depth?)
nc_en4      = "EN4_concatenate_1900to2021_lon-80to00_lat00to65.nc"
datpath_en4 = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/01_Data/EN4/proc/"
ds_en4     = xr.open_dataset(datpath_en4+nc_en4)

# Select Point and variable
time_en4 = ds_en4.time.values
sss_en4  = ds_en4.salinity.sel(lon=lonf,lat=latf,method='nearest').load()


# ---

#GlorysV12 dataset (.494 depth?)
nc_glorys      = "glorys12v1_so_NAtl_1993_2019_merge.nc"
datpath_glorys = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/01_Data/glorys12v1/"
ds_glorys  = xr.open_dataset(datpath_glorys+nc_glorys)

# Select Point and Variable
time_glorys = ds_glorys.time.values
sss_glorys = ds_glorys.so.sel(lon=lonf,lat=latf,method='nearest').load()

# ---

# Processing Inputs
sss_in         = [sss_en4.values,sss_glorys.values]
times          = [time_en4,time_glorys]
dataset_names  =  ["EN4","GLORYS12v1"]
dataset_colors = ["lightcoral","navy"]
ndata          = len(sss_in)


# Misc
mons3 = proc.get_monstr(nletters=3)


# Other Toggles
debug = True
#%% Preprocessing (Detrend, Remove Seasonal Cycle)


# Deseason
scycles     = []
vanoms      = []
for n in range(ndata):
    
    sss = sss_in[n]
    
    scycle,tsanom = proc.calc_clim(sss,0,returnts=1) # Calculate Climatology
    tsanom = tsanom - scycle[None,:] # Deseason
    
    scycles.append(scycle)
    vanoms.append(tsanom.flatten())

if debug:
    
    # Examine Seasonal Cycle 
    fig,ax = viz.init_monplot(1,1,)
    for n in range(ndata):
        ax.plot(mons3,scycles[n],label=dataset_names[n],c=dataset_colors[n],marker="o")
    ax.legend()
    ax.set_title("Seasonal Cycle in Salinity @ %s" % (loctitle))
    ax.set_ylabel("Salinity (psu)")
    
    savename = "%sSSS_Scycle_Obs_EN4_GLORYS.png" % figpath
    plt.savefig(savename,dpi=150,bbox_inches='tight')
    
    
#%% Convert Datetime

#timesdt = proc.convert_datenum(times[0])



#%% Check Existence of a trend
plot_both=True # True to plot other timeseries in background

fig,axs = plt.subplots(2,1,constrained_layout=True,figsize=(10,4))

for n in range(ndata):
    ax  = axs[n]
    sss = vanoms[n]
    
    if plot_both:
        if n == 1:
            ax.plot(times[0],vanoms[0],label=dataset_names[0],c=dataset_colors[0],alpha=1)
            ax.set_xlim(times[1][0],times[1][-1])
            
        elif n == 0:
            ax.plot(times[1],vanoms[1],label=dataset_names[1],c=dataset_colors[1],alpha=0.5)
            
            # Set Xlim
            ax.set_xlim([times[0][0],times[0][-1]])
            
            # Plot Range of GLORYS12V1
            ax.axvline([times[1][0]],color="k",ls='dashed',lw=0.75,label="")
            ax.axvline([times[1][-1]],color="k",ls='dashed',lw=0.75,label="")
            
    ax.plot(times[n],sss,label=dataset_names[n],c=dataset_colors[n])
    
    ax.set_ylim([-0.55,0.55])
    ax.set_ylabel("Salinity (psu)")
    ax.axhline([0],ls='dashed',c="k",alpha=0.2)
    #ax.grid(True)
    plt.suptitle("SSS Timeseries @ %s" % loctitle)
    
    ax.legend()
    savename = "%sSSS_Timeseries_Obs_EN4_GLORYS.png" % figpath
    plt.savefig(savename,dpi=150,bbox_inches='tight')

#%% Calculate Metrics (without Detrending)

nsmooth = 2
lags    = np.arange(37)
tsmetrics_nodetrend = scm.compute_sm_metrics(vanoms,lags=lags,nsmooth=nsmooth)

#%% Examine Autocorrelation, ETC

kmonth = 1
xtksl  = np.arange(0,37,3)

for kmonth in range(12):
    
    fig,ax = viz.init_acplot(kmonth,xtksl,lags,title="")
    
    for n in range(ndata):
        plotvar = tsmetrics_nodetrend['acfs'][kmonth][n]
        ax.plot(lags,plotvar,label=dataset_names[n],c=dataset_colors[n],marker="o")
    ax.legend()
    ax.set_title("%s ACF @ %s" % (mons3[kmonth],loctitle))
    
    savename = "%sSSS_ACF_mon%02i_Obs_EN4_GLORYS.png" % (figpath,kmonth+1)
    plt.savefig(savename,dpi=150,bbox_inches='tight')
    plt.close()



#%% Plot Monthly Variance


fig,ax=viz.init_monplot(1,1,)

for n in range(ndata):
    plotvar = tsmetrics_nodetrend['monvars'][n]
    lab = "%s (var=%f)" % (dataset_names[n],np.var(vanoms[n]))
    ax.plot(mons3,plotvar,label=lab,c=dataset_colors[n],marker="o")

ax.legend()
ax.set_ylabel("SSS Variance ($psu^2$)")
ax.set_title("SSS Monthly Variance @ %s" % loctitle)
savename = "%sSSS_MonVar_Obs_EN4_GLORYS.png" % (figpath)
plt.savefig(savename,dpi=150,bbox_inches='tight')

#%% Plot the spectra
dtplot  = 3600*24*30
plotyrs = [100,50,25,10,5,2]
xtks    = 1/(np.array(plotyrs)*12)

freqs   = tsmetrics_nodetrend['freqs']
specs   = tsmetrics_nodetrend['specs']

fig,ax = plt.subplots(1,1,constrained_layout=1,figsize=(8,3))
for n in range(ndata):
    lab = "%s (var=%f)" % (dataset_names[n],np.var(vanoms[n]))
    ax.plot(freqs[n]*dtplot,specs[n]/dtplot,label=lab,c=dataset_colors[n],marker="o")
ax.set_xticks(xtks,labels=plotyrs)
ax.set_xlim([xtks[0],xtks[-1]])
ax.legend()
ax.set_xlabel("Period (Years)")
ax.set_ylabel("$psu^2$/cpy")
ax.set_title("SSS Power Spectra (nsmooth=%i) @ %s" % (nsmooth,loctitle))
savename = "%sSSS_Power_Spectra_Obs_EN4_GLORYS.png" % (figpath)
plt.savefig(savename,dpi=150,bbox_inches='tight')


#%% Save the output

outdict = {
    'scycles'        : scycles, # Seasonal Cycles
    'sss'            : vanoms,  # SSS anomaly timeseries
    'tsmetrics'      : tsmetrics_nodetrend, # Metrics (ACF, Spectra, Monthly Variance)
    'times'          : times, # Corresponding np.datetime64
    'dataset_names'  : dataset_names,
    'dataset_colors' : dataset_colors,
    'lags'           : lags,
    'nsmooth'        : nsmooth,
    }

savename = "%sObs_SSS_Metrics.npz" % outpath
np.savez(savename,**outdict,allow_pickle=True)