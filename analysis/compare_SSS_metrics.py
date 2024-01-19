#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Compare SSS Metrics at a point
Copied sections from get_SSS_point_obs.py

Created on Thu Jan 18 17:43:16 2024

@author: gliu

"""

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import sys
from tqdm import tqdm
import copy

import datetime

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


# Set other things
mons3      = proc.get_monstr(nletters=3)
cnames     =["CESM1_PIC","CESM1_HTR"]
clongnames = ["CESM1 (PiControl)","CESM1 (Historical LE)"]
ccolors    = ["k","gray"]

#%% Load some data

# ---

# Load SSS from CESM1 PIC [get_point_data_CESM_PIC.py]
nc_pic  = "CESM1_FULL_PIC_SALT.nc"
ds_pic  = xr.open_dataset(outpath+nc_pic).isel(z_t=0) # From 500 cm
sss_pic  = ds_pic.SALT.values # [time]
time_pic = ds_pic.time.values

# ---

# Load SSS from CESM1 Historical [get_point_data_stormtrack.py]
nc_htr  = "CESM1_htr_SSS.nc"
ds_htr  = xr.open_dataset(outpath+nc_htr)
sss_htr = ds_htr.SSS.values.squeeze() # [ens x time]
time_htr = ds_htr.time.values
nens,_ = sss_htr.shape

# ---

# Load metrics calculated in [get_SSS_point_obs.py]
obsname = "%sObs_SSS_Metrics.npz" % outpath
obsdict = np.load(obsname,allow_pickle=True)

# Loadout some data
obsnames = obsdict['dataset_names']
obscols  = obsdict['dataset_colors']
ndata    = len(obsnames)

obstimes = obsdict['times']
obsmetrics = obsdict['tsmetrics'].item()


#%% Remove Seasonal Cycle

anom_pic,scyc_pic = proc.deseason(sss_pic,return_scycle=True)
anom_htr,scyc_htr = proc.deseason(sss_htr,dim=1,return_scycle=True) # [nens,nyrs,month]

# Squeeze so that it is just [12] or [ens x 12]
scyc_pic = scyc_pic.squeeze()
scyc_htr = scyc_htr.squeeze()

# From [year x mon] to [time]
anom_pic = anom_pic.flatten()
_,nyrs,_ = anom_htr.shape
anom_htr = anom_htr.reshape(nens,12*nyrs) # [nens, ntime]

# Get Month and Day
time_pic_str  = ["%s-%s" % (t.year,t.month) for t in time_pic]
time_htr_str  = ["%s-%s" % (t.year,t.month) for t in time_htr]
#%% Compare Seasonal Cycles

# Plot Obs
fig,ax = viz.init_monplot(1,1,)
for n in range(ndata):
    ax.plot(mons3,obsdict['scycles'][n],label=obsnames[n],c=obscols[n],marker="o")


# Plot CESM1 PIC
ax.plot(mons3,scyc_pic,label=clongnames[0],color=ccolors[0],)

# Plot CESM1 HTR
for e in range(nens):
    if e == 0:
        lab=clongnames[1] + ", Indv. Member"
    else:
        lab =""
    ax.plot(mons3,scyc_htr[e,:],label=lab,color=ccolors[1],alpha=0.15)
ax.plot(mons3,scyc_htr.mean(0),label=clongnames[1] + ", Ens. Avg",color=ccolors[1])


ax.set_ylim([35,35.6])
ax.legend(ncol=2,loc="upper left")

ax.set_title("Seasonal Cycle in Salinity @ %s" % (loctitle))
ax.set_ylabel("Salinity (psu)")

savename = "%sSSS_Scycle_Obs_CESM1.png" % figpath
plt.savefig(savename,dpi=150,bbox_inches='tight')

#%% Compare Timeseries

ts_target = obsdict['sss'][1]
ts_input  = anom_pic.squeeze()

def sliding_rmse(ts_target,ts_input,return_score=True):
    
    ntime_targ = len(ts_target)
    ntime      = len(ts_input)
    nstarts      = ntime-ntime_targ
    rms_score  = []
    for t in range(nstarts):
        ts_in  = ts_input[t:(t+ntime_targ)]
        rmserr = np.sqrt(np.nanmean(ts_target-ts_in)**2)
        rms_score.append(rmserr)
    idmin = np.nanargmin(rms_score)
    if return_score:
        return idmin,ts_input[idmin:(idmin+ntime_targ)],rms_score
    else:
        return idmin,ts_input[idmin:(idmin+ntime_targ)]
    
    

# Find Comparable Period in CESM PIC
pic_idmin,pic_match,pic_score = sliding_rmse(ts_target,ts_input,return_score=True)

# Find Comparable Period in CESM1 LENS
slideoutput = [sliding_rmse(ts_target,anom_htr[e,:]) for e in range(nens)]
scores_htr  = [np.argmin(s[2]) for s in slideoutput]
minens      = np.argmin(scores_htr)

htr_idmin   = slideoutput[minens][0]
htr_match   = slideoutput[minens][1]

# idmins_htr  = []
# matches_htr = []
# scores_htr  = []
# for e in range(nens):
#     ts_input  = anom_htr[e,:].squeeze()
#     idm,match_ts,rmscore=sliding_rmse(ts_target,ts_input,return_score=True)
#     idmins_htr.append(idm)
#     matches_htr,app

# Find same period for EN4
en4_idmin,en4_match,en4_score = sliding_rmse(ts_target,obsdict['sss'][0],return_score=True)

#%% Plot a similar time period
ntime_glorys = len(obstimes[1])
fig,ax       = plt.subplots(1,1,constrained_layout=True,figsize=(12,4))

# Plot PiC
idsel  = np.arange(pic_idmin,pic_idmin+ntime_glorys)
trange = np.arange(ntime_glorys)
lab    = "%s (%s to %s)" % (cnames[0],time_pic_str[idsel[0]],time_pic_str[idsel[-1]])
ax.plot(trange,
        anom_pic[idsel],
        label=lab,
        c=ccolors[0]
        )

# Plot CESM-Historical
idselh = np.arange(htr_idmin,htr_idmin+ntime_glorys)
lab    = "%s (%s to %s)" % (cnames[1],time_htr_str[idselh[0]],time_htr_str[idselh[-1]])
ax.plot(trange,
        anom_htr[minens,idselh],
        label=lab,
        c=ccolors[1]
        )

# Plot Glorys
lab = "%s (%s to %s)" % (obsnames[1],obstimes[1][0],obstimes[1][-1])
ax.plot(trange,obsdict['sss'][1],label=lab,color=obscols[1])

# Plot en4
idselen4 = np.arange(en4_idmin,en4_idmin+ntime_glorys)
lab = "%s (%s to %s)" % (obsnames[0],obstimes[0][idselen4[0]],obstimes[0][idselen4[-1]])
ax.plot(trange,
        en4_match,
        label=lab,
        c=obscols[0]
        )

ax.legend(ncol=2)
ax.set_xlim([trange[0],trange[-1]])

ax.set_ylabel("psu")
ax.set_xlabel("Time (months)")

savename = "%sSSS_Timeseries_Obs_CESM1.png" % figpath
plt.savefig(savename,dpi=150,bbox_inches='tight')

#%% Just Plot the timeseries

#fig,axs = plt.subplots(4,1)




#%% Compute some metrics

htr_list = [anom_htr[e,:] for e in range(nens)]
nanid = [np.any(np.isnan(s)) for s in htr_list]
htr_list = [anom_htr[e,:] for e in range(nens) if np.all(~np.isnan(anom_htr[e,:]))]
tsmetrics_pic = scm.compute_sm_metrics([anom_pic,],nsmooth=25)
tsmetrics_htr = scm.compute_sm_metrics(htr_list,nsmooth=5) # [ens][metric]

#%% Plot Autocorrelation

kmonth = 1
xtksl  = np.arange(0,37,3)
lags   = np.arange(37)


for kmonth in range(12):

    fig,ax = plt.subplots(1,1,figsize=(10,3.5),constrained_layout=True)
    ax,_ = viz.init_acplot(kmonth,xtksl,lags,title="",ax=ax)
    
    
    for n in range(ndata):
        plotvar = obsmetrics['acfs'][kmonth][n]
        ax.plot(lags,plotvar,label=obsnames[n],c=obscols[n],marker="o")
    
    ax.plot(lags,tsmetrics_pic['acfs'][kmonth][0],label=cnames[0],c=ccolors[0],marker="d")
    
    
    acf_eavg = np.zeros(37)
    for e in range(len(htr_list)):
        plotvar = tsmetrics_htr['acfs'][kmonth][e]
        if e == 0:
            lab = "%s (Indv. Member)" % cnames[1]
        else:
            lab = ""
            
        ax.plot(lags,plotvar,label=lab,c=ccolors[1],marker="d",alpha=0.05)
        acf_eavg += plotvar
    acf_eavg = acf_eavg/len(htr_list)
    lab = "%s (Ens. Avg.)" % cnames[1]
    ax.plot(lags,acf_eavg,label=lab,c=ccolors[1],marker="d")
    
    ax.legend(ncol=3,fontsize=8)
    ax.set_title("%s ACF @ %s" % (mons3[kmonth],loctitle))
    savename = "%sSSS_ACF_mon%02i_Obs_CESM1.png" % (figpath,kmonth+1)
    plt.savefig(savename,dpi=150,bbox_inches='tight')
    #plt.show()



#%% plot monthly variance

fig,ax = viz.init_monplot(1,1)

for n in range(ndata):
    plotvar = obsmetrics['monvars'][n]
    ax.plot(mons3,plotvar,label=obsnames[n],c=obscols[n],marker="o")


ax.plot(mons3,tsmetrics_pic['monvars'][0],label=cnames[0],c=ccolors[0],marker="d")

monvar_eavg = np.zeros(12)
for e in range(len(htr_list)):
    plotvar = tsmetrics_htr['monvars'][e]
    if e == 0:
        lab = "%s (Indv. Member)" % cnames[1]
    else:
        lab = ""
        
    ax.plot(mons3,plotvar,label=lab,c=ccolors[1],marker="d",alpha=0.05)
    monvar_eavg += plotvar
monvar_eavg = monvar_eavg/len(htr_list)
ax.plot(mons3,monvar_eavg,label=lab,c=ccolors[1],marker="d")
ax.legend()
#ax.plot(lags,tsmetrics_pic['acfs'][kmonth][0],label=cnames[0],c=ccolors[0],marker="d")
ax.set_title("SSS Monthly Variance @ %s" % loctitle)
savename = "%sSSS_MonVar_Obs_CESM1.png" % (figpath)
plt.savefig(savename,dpi=150,bbox_inches='tight')

#%% Plot Spectra
#nsmooth

dtplot  = 3600*24*30
plotyrs = [100,50,25,10,5,2]
xtks    = 1/(np.array(plotyrs)*12)

freqs   = obsmetrics['freqs']
specs   = obsmetrics['specs']

fig,ax = plt.subplots(1,1,constrained_layout=1,figsize=(8,3))
for n in range(ndata):
    lab = "%s (var=%f)" % (obsnames[n],np.var(obsdict['sss'][n]))
    
    ax.plot(freqs[n]*dtplot,specs[n]/dtplot,label=lab,c=obscols[n],marker="o")


lab = "%s (var=%f)" % (cnames[0],np.var(anom_pic))
ax.plot(tsmetrics_pic['freqs'][0]*dtplot,tsmetrics_pic['specs'][0]/dtplot,label=lab,c=ccolors[0],marker=".")


for e in range(len(htr_list)):
    if e == 0:
        spec_eavg = tsmetrics_htr['specs'][e]
    else:
        spec_eavg += tsmetrics_htr['specs'][e]
spec_eavg = spec_eavg/len(htr_list)

lab = "%s (var=%f)" % (cnames[1],np.nanmean(np.var(anom_htr,1)))
ax.plot(tsmetrics_htr['freqs'][0]*dtplot,spec_eavg/dtplot,label=lab,c=ccolors[1],marker=".")




ax.set_xticks(xtks,labels=plotyrs)
ax.set_xlim([xtks[0],xtks[-1]])
ax.legend()
ax.set_xlabel("Period (Years)")
ax.set_ylabel("$psu^2$/cpy")
ax.set_title("SSS Power Spectra @ %s" % (loctitle))
savename = "%sSSS_Power_Spectra_Obs_CESM1.png" % (figpath)
plt.savefig(savename,dpi=150,bbox_inches='tight')


#%% Save Output

outdict = {
    'scycles'        : [scyc_pic,scyc_htr], # Seasonal Cycles
    'sss'            : [anom_pic,anom_htr],  # SSS anomaly timeseries
    'tsmetrics'      : [tsmetrics_pic,tsmetrics_htr], # Metrics (ACF, Spectra, Monthly Variance). For Htr, its [metric][kmonth][ens]
    'times'          : [time_pic_str,time_htr_str], # Corresponding np.datetime64
    'dataset_names'  : cnames,
    'dataset_colors' : ccolors,
    'lags'           : lags,
    'nsmooth'        : [25,5], # Manually Coded because :(
    }

savename = "%sCESM1_SSS_Metrics.npz" % outpath
np.savez(savename,**outdict,allow_pickle=True)