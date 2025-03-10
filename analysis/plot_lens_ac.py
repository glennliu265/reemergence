#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  2 18:31:47 2024

Visualize ACF for SST and/or SSS at a single point.
Can also examine the mean ACF over a region.

Copied upper section from stochmod_point.

@author: gliu
"""

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import sys
from tqdm import tqdm
import copy
import matplotlib as mpl

#%% Import Custom Modules
amvpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/03_Scripts/" # amv module
scmpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/03_Scripts/stochmod/model/"

sys.path.append(amvpath)
sys.path.append(scmpath)

from amv import proc,viz
import scm
import amv.loaders as dl
import yo_box as ybx

#%% User Edits

# Location
lonf           = 330
latf           = 50
locfn,loctitle = proc.make_locstring(lonf,latf)

datpath        = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/ptdata/lon%s_lat%s/" % (lonf,latf)
figpath        = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/02_Figures/20241201/"
proc.makedir(figpath)

flxs           = ["LHFLX","SHFLX","FLNS","FSNS","qnet"]
prcps          = ["PRECC","PRECL","PRECSC","PRECSL",]
varnames_ac    = ["SST","SSS"]

# Forcing from 
Fori  = np.array([53.36403275, 50.47200521, 43.19549306, 32.95324516, 26.30336189,
           22.53761546, 22.93124771, 26.54155223, 32.79647001, 39.71981049,
           45.65141678, 50.43875758])
Ftest = np.ones(12)*53.36403275

# Other plotting information
mons3          = proc.get_monstr(nletters=3)

#%% Set mode

darkmode = True
if darkmode:
    dfcol = "w"
    bgcol = np.array([15,15,15])/256
    transparent = True
    plt.style.use('dark_background')
    mpl.rcParams['font.family']     = 'Avenir'
else:
    dfcol = "k"
    bgcol = "w"
    transparent = False
    plt.style.use('default')

#%% Retrieve the autocorrelation for SSS and SST from CESM1
datpath_ac  = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/proc/"
varnames_ac = ["SST","SSS"]
ac_colors   = ["darkorange","lightseagreen"]

# Read in datasets
ds_pt = []
ac_pt = []
for v in range(2):
    ds = xr.open_dataset("%sHTR-FULL_%s_autocorrelation_thres0.nc" % (datpath_ac,varnames_ac[v]))
    
    ds  = ds.sel(lon=lonf-360,lat=latf,method='nearest').sel(thres="ALL").load()# [ens lag mon]
    ds_pt.append(ds)
    ac_pt.append(ds[varnames_ac[v]].values) # [variable][ens x lag x month]

#%% Load Everything

# Read in datasets
ds_all = []
for v in range(2):
    ds = xr.open_dataset("%sHTR-FULL_%s_autocorrelation_thres0.nc" % (datpath_ac,varnames_ac[v]))
    
    ds  = ds.sel(thres="ALL").load()# [ens lag mon]
    ds_all.append(ds)
   # ac_pt.append(ds[varnames_ac[v]].values) # [variable][ens x lag x month]


#%% Make Locator

fsztk    = 8


bboxplot = [-65,0,35,70]
bboxsel  = [-40,-25,50,60] # SPG

# bboxplot = [-80,0,20,70]
# bboxsel  = [-70,-55,30,40] # SPG

import cartopy.crs as ccrs
#plt.style.use("default")
fig,ax = plt.subplots(1,1,figsize=(2,2),
                      subplot_kw={'projection':ccrs.PlateCarree()},constrained_layout=True)
# ax=viz.add_coast_grid(ax,bbox=bboxplot,line_color=dfcol,grid_color=dfcol,
#                       fill_color=dfcol,fix_lon=[-40,-25],fix_lat=[50,60],fontsize=fsztk)

ax=viz.add_coast_grid(ax,bbox=bboxplot,line_color=dfcol,grid_color=dfcol,
                      fill_color=dfcol,fix_lon=[bboxsel[0],bboxsel[1]],fix_lat=[bboxsel[2],bboxsel[3]],fontsize=fsztk)

viz.plot_box(bboxsel,linewidth=2.5,color="cornflowerblue")

savename ="%sSST_BBOX.png" % (figpath)
plt.savefig(savename,transparent=True,dpi=300,bbox_inches='tight')

#%% Sect Region and average autocorrelation
acreg = []
for v in range(2):
    dsreg =proc.sel_region_xr(ds_all[v],bboxsel)
    acsreg = dsreg[varnames_ac[v]].values
    acreg.append(np.nanmean(acsreg,(3,4)))


#%% Make a plot

acin = acreg
fsztk   = 14
fsza    = 14
fszl    = 16
alph    = 0.15
plotvs  = [0,1]
vnames  = ["SST","SSS"]
vcolors = ['w', np.array([2,183,189])/256]

ytks = np.arange(-.0,1.1,0.2)
kmonth = 1
xtks  = np.arange(0,37,2)
mons3 = proc.get_monstr(nletters=3)
xtks_mon = (xtks + kmonth)%12
xtks_mon = [mons3[xx] for xx in xtks_mon]


xlm   = [0,18]
#monlb = ["(Feb)","(Aug)"]
lags = np.arange(0,37,1)
xlbls = ["%s\n(%s)" % (xtks[l],xtks_mon[l]) for l in range(len(xtks))] 
ylbls = ["%.1f" % i for i in ytks]

for plotii in range(2):
    #plt.style.use("default")
    fig,ax= plt.subplots(1,1,constrained_layout=True,figsize=(10,3.5))
    
    ax.spines[['right', 'top']].set_visible(False)
    
    ax.set_xticks(xtks,labels=xlbls,fontsize=fsztk)
    ax.set_yticks(ytks,labels=ylbls,fontsize=fsztk)
    
    ax.set_xlim(xlm)
    ax.set_ylim([ytks[0],ytks[-1]])
    
    ax.set_xlabel("Lag (Months)",fontsize=fsza)
    ax.set_ylabel("Correlation",fontsize=fsza)
    
    nens,_,_ = acin[0].shape
    pcs = ["gray",'k']
    
    
    for v in range(len(plotvs)):
        if plotii == 0 and v == 1:
            continue
        
        acs = acin[v][:,:,kmonth] # ens x lag x mon
        
        pc=pcs[v]
        for e in range(nens):
            plotac=acs[e,:]
            
            if v == 0:
                col_in = dfcol
            else:
                col_in = vcolors[v]
            
            if e == 0 and plotii == 0:
                lbl="Ensemble Member (%s)"% vnames[v]
            else:
                lbl=""
            
            
            ax.plot(lags,plotac,color=col_in,alpha=alph,label=lbl)   # color used to be pc
        ax.plot(lags,acs.mean(0),color=col_in,label="Ensemble Mean (%s)" % vnames[v],marker="o",markerfacecolor="None")
        
        
    ax.legend(fontsize=fszl,framealpha=0.2,ncol=2)
    savename ="%sSST_ACF_plotii%i.png" % (figpath,plotii)
    plt.savefig(savename,transparent=True,dpi=300,bbox_inches='tight')
    
    
    #viz.init_acplot(kmonth,xtks,lags,ax=ax)

#%% Do the same but with SSS as well

