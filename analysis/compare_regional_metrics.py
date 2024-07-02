#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Compare Regional Metrics

Works with the output from postprocess_smoutput_auto

Created on Wed Mar 13 13:06:45 2024

@author: gliu

"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import xarray as xr
import sys
from tqdm import tqdm
import copy
import glob
import time
import cartopy.crs as ccrs
import os

# ----------------------------------
#%% Import custom modules and paths
# ----------------------------------

# Indicate the Machine!
machine = "Astraeus"

# First Load the Parameter File
cwd = os.getcwd()
sys.path.append(cwd+"/../")
import reemergence_params as rparams

# Paths and Load Modules
pathdict   = rparams.machine_paths[machine]

sys.path.append(pathdict['amvpath'])
sys.path.append(pathdict['scmpath'])
sys.path.append(pathdict['scmpath'] + "../")
from amv import proc,viz
import scm
import amv.loaders as dl
import yo_box as ybx
import stochmod_params as sparams

# Set needed paths
figpath     = pathdict['figpath']
input_path  = pathdict['input_path']
output_path = pathdict['output_path']
procpath    = pathdict['procpath']
rawpath     = pathdict['raw_path']

# Make Needed Paths
proc.makedir(figpath)

#%% Indicate experiments to load

# Check updates after switching detrainment and correting Fprime (SST)
regionset       = "TCMPi24"
comparename     = "SST_AprilUpdate"
expnames        = ["SST_EOF_LbddEnsMean","SST_EOF_LbddCorr_Rerun","SST_CESM"]
expnames_long   = ["Stochastic Model (Exp Fit)","Stochastic Model (Corr.)","CESM1"]
expnames_short  = ["SM_old","SM_new","CESM"]
ecols           = ["forestgreen","goldenrod","k"]
els             = ["solid",'dashed','solid']
emarkers        = ["d","x","o"]

# Check updates after switching detrainment and correting Fprime (SSS)
regionset       = "TCMPi24"
comparename     = "SSS_AprilUpdate"
expnames        = ["SSS_EOF_Qek_LbddEnsMean","SSS_EOF_LbddCorr_Rerun","SSS_CESM"]
expnames_long   = ["Stochastic Model (Exp Fit)","Stochastic Model (Corr.)","CESM1"]
expnames_short  = ["SM_old","SM_new","CESM"]
ecols           = ["forestgreen","goldenrod","k"]
els             = ["solid",'dashed','solid']
emarkers        = ["d","x","o"]


# Compare impact of adding lbd_e
regionset       = "TCMPi24"
comparename     = "lbde_comparison"
expnames        = ["SSS_EOF_LbddCorr_Rerun_lbdE","SSS_EOF_LbddCorr_Rerun","SSS_CESM"]
expnames_long   = ["Stochastic Model (with $\lambda^e$)","Stochastic Model","CESM1"]
expnames_short  = ["SM_lbde","SM","CESM"]
ecols           = ["forestgreen","goldenrod","k"]
els             = ["solid",'dashed','solid']
emarkers        = ["d","x","o"]








# # CSU Comparisons (SSS)
# regionset       = "SSSCSU"
# comparename     = "lbde_comparison_CSU"
# expnames        = ["SSS_EOF_LbddCorr_Rerun","SSS_EOF_LbddCorr_Rerun_NoLbdd","SSS_EOF_LbddCorr_Rerun_lbdE","SSS_CESM",]
# expnames_long   = ["Stochastic Model","Stochastic Model (No $\lambda^d$)","Stochastic Model (with $\lambda^e$)","CESM1"]
# expnames_short  = ["SM","SM_NoLbdd","SM_lbde","CESM"]
# ecols           = ["forestgreen","goldenrod","magenta","k"]
# els             = ["solid",'dashed','dotted','solid']
# emarkers        = ["d","x",'+',"o"]

# CSU Comparisons (SST)
regionset       = "SSSCSU"
comparename     = "SST_comparison_CSU"
expnames        = ["SST_EOF_LbddCorr_Rerun","SST_EOF_LbddCorr_Rerun_NoLbdd","SST_CESM"]
expnames_long   = ["Stochastic Model","Stochastic Model (No $\lambda^d$)","CESM1"]
expnames_short  = ["SM","SM_NoLbdd","CESM"]
ecols           = ["forestgreen","goldenrod","k"]
els             = ["solid",'dashed','solid']
emarkers        = ["d","x","o"]



# # Compare SSS with and without detrainment damping
# regionset       = "TCMPi24"
# comparename     = "SSS_Lbdd"
# expnames        = ["SSS_EOF_Qek_LbddEnsMean","SSS_EOF_NoLbdd","SSS_CESM"]
# expnames_long   = ["Stochastic Model (with Detrainment Damping)","Stochastic Model","CESM1"]
# expnames_short  = ["SM ($\lambda ^d$)","SM","CESM"]
# ecols           = ["forestgreen","goldenrod","k"]
# els             = ["solid",'dashed','solid']
# emarkers        = ["d","x","o"]

#  Same as comparing lbd_e effect, but with Evaporation forcing corrections
regionset       = "SSSCSU"
comparename     = "lbde_comparison_signcorr"
expnames        = ["SSS_EOF_LbddCorr_Rerun_lbdE_neg","SSS_EOF_LbddCorr_Rerun_lbdE","SSS_EOF_LbddCorr_Rerun","SSS_CESM"]
expnames_long   = ["Stochastic Model (sign corrected + $\lambda^e$)","Stochastic Model (with $\lambda^e$)","Stochastic Model","CESM1"]
expnames_short  = ["SM_lbde_neg","SM_lbde","SM","CESM"]
ecols           = ["magenta","forestgreen","goldenrod","k"]
els             = ['dotted',"solid",'dashed','solid']
emarkers        = ['+',"d","x","o"]


#  Same as comparing lbd_e effect, but with Evaporation forcing corrections
regionset       = "SSSCSU"
comparename     = "lhflx_v_full"
expnames        = ["SSS_EOF_LHFLX_lbdE","SSS_EOF_LbddCorr_Rerun_lbdE_neg","SSS_EOF_LbddCorr_Rerun_lbdE","SSS_EOF_LbddCorr_Rerun","SSS_CESM"]
expnames_long   = ["Stochastic Model (LHFLX Only)","Stochastic Model (sign corrected + $\lambda^e$)","Stochastic Model (with $\lambda^e$)","Stochastic Model","CESM1"]
expnames_short  = ["SM_LHFLX","SM_lbde_neg","SM_lbde","SM","CESM"]
ecols           = ["cyan","magenta","forestgreen","goldenrod","k"]
els             = ['dashed','dotted',"solid",'dashed','solid']
emarkers        = ["^",'+',"d","x","o"]

# SST Comparison (Paper Draft, essentially Updated CSU)
regionset       = "SSSCSU"
comparename     = "SST_Paper_Draft01"
expnames        = ["SST_EOF_LbddCorr_Rerun","SST_EOF_LbddCorr_Rerun_NoLbdd","SST_CESM"]
expnames_long   = ["Stochastic Model","Stochastic Model (No $\lambda^d$)","CESM1"]
expnames_short  = ["SM","SM_NoLbdd","CESM"]
ecols           = ["forestgreen","goldenrod","k"]
els             = ["solid",'dashed','solid']
emarkers        = ["d","x","o"]

# #  Same as comparing lbd_e effect, but with Evaporation forcing corrections
# regionset       = "SSSCSU"
# comparename     = "SSS_Paper_Draft01"
# expnames        = ["SSS_EOF_LbddCorr_Rerun_lbdE_neg","SSS_EOF_LbddCorr_Rerun","SSS_EOF_LbddCorr_Rerun_NoLbdd","SSS_CESM"]
# expnames_long   = ["Stochastic Model (sign corrected + $\lambda^e$)","Stochastic Model (with $\lambda^e$)","Stochastic Model","CESM1"]
# expnames_short  = ["SM_lbde_neg","SM_lbde","SM","CESM"]
# ecols           = ["magenta","forestgreen","goldenrod","k"]
# els             = ['dotted',"solid",'dashed','solid']
# emarkers        = ['+',"d","x","o"]

# regionset = "TCMPi24"
TCM_ver         = False # Set to just plot 2 panels for ACF
Draft01_ver     = True

# # # Compare SST with and without detrainment damping
# comparename     = "SST_Lbdd"
# expnames        = ["SST_EOF_LbddEnsMean","SST_EOF_NoLbdd","SST_CESM"]
# expnames_long   = ["Stochastic Model (with Detrainment Damping)","Stochastic Model","CESM1"]
# ecols           = ["forestgreen","goldenrod","k"]
# els             = ["solid",'dashed','solid']
# emarkers        = ["d","x","o"]




#%% Load Regional Average SSTs and Metrics for the selected experiments

nexps       = len(expnames)

seavar_all  = []
var_all     = []
tsm_all     = []
rssts_all   = []
acfs_all    = []
amv_all     = []
for e in range(nexps):
    
    # Get Experiment information
    expname        = expnames[e]
    
    if "SSS" in expname:
        varname = "SSS"
    elif "SST" in expname:
        varname = "SST"
    metrics_path    = output_path + expname + "/Metrics/"
    
    # Load Pointwise variance
    ds_std = xr.open_dataset(metrics_path+"Pointwise_Variance.nc").load()
    var_all.append(ds_std)
    
    # Load Seasonal variance
    ds_std2 = xr.open_dataset(metrics_path+"Pointwise_Variance_Seasonal.nc").load()
    seavar_all.append(ds_std2)
    
    # Load Regionally Averaged SSTs
    ds = xr.open_dataset(metrics_path+"Regional_Averages_%s.nc" % regionset).load()
    rssts_all.append(ds)
    
    # Load Regional Metrics
    ldz = np.load(metrics_path+"Regional_Averages_Metrics_%s.npz" % regionset,allow_pickle=True)
    tsm_all.append(ldz)
    
    # Load Pointwise_ACFs
    ds_acf = xr.open_dataset(metrics_path + "Pointwise_Autocorrelation_thresALL_lag00to60.nc")[varname].load()
    acfs_all.append(ds_acf)  
    
    # # Load AMV Information
    # ds_amv = xr.open_dataset(metrics_path + "AMV_Patterns_SMPaper.nc").load()
    
"""

tsm_all [experiment][region_name].item()[metric]
where metric is :  one of ['acfs', 'specs', 'freqs', 'monvars', 'CCs', 'dofs', 'r1s']
see scm.compute_sm_metrics()

"""

#%% Load Mask
masknc = metrics_path + "Land_Ice_Coast_Mask.nc"
dsmask = xr.open_dataset(masknc).mask#__xarray_dataarray_variable__

#%% Get Region Information, Set Plotting Parameters

# Get Region Info
rdict                       = rparams.region_sets[regionset]
regions                     = rdict['regions']
bboxes                      = rdict['bboxes']
rcols                       = rdict['rcols']
rsty                        = rdict['rsty']
regions_long                = rdict['regions_long']
nregs                       = len(bboxes)


# # Get latitude and longitude
# lon = ds_acf.lon.values
# lat = ds_acf.lat.values


#regions_long = ["Subpolar Gyre","Northern North Atlantic","Subtropical Gyre (East)","Subtropical Gyre (West)"]

# Plotting Information
bbplot                      = [-80,0,22,64]
mpl.rcParams['font.family'] = 'Avenir'
proj                        = ccrs.PlateCarree()
mons3                       = proc.get_monstr()

# Font Sizes
fsz_title                   = 20
fsz_ticks                   = 14
fsz_axis                    = 16
fsz_legend                  = 16

# --------------------------------
#%% Plot 1: Regional ACF
# --------------------------------

nregs     = len(bboxes)
kmonth    = 1         # Set month of analysis
plot_ens_indv = False
acf_lw    = 2.5

# Plotting Parameters
xtksl     = np.arange(0,37,3)
lags      = np.arange(37)

if (regionset == "SMPaper") or (regionset == "TCMPi24" and not TCM_ver): # Plot with 4 regions

    plotorder = [0,1,3,2] # Set Order of plotting

    
    fig,axs   = plt.subplots(2,2,constrained_layout=True,figsize=(16,8.5),sharey=True)
    lines     = []
    for aa in range(nregs):
        
        ax    = axs.flatten()[aa]
        rr    = plotorder[aa]
        rname = regions[rr]
        
        ax,_ = viz.init_acplot(kmonth,xtksl,lags,title="",ax=ax,fsz_axis=fsz_axis,fsz_ticks=fsz_ticks)
        ax   = viz.add_ticks(ax=ax)
        
        
        
        # Adjust Axis Labels
        if aa < 2:
            ax.set_xlabel("")
        else:
            ax.set_xlabel("Lag (Months, Lag 0=%s)" % (mons3[kmonth]))
        if aa % 2 != 0:
            ax.set_ylabel("")
        else:
            ax.set_ylabel("Correlation (%s)" % (varname))
        
        for ex in range(nexps):
            plotvar = np.nanmean(np.array(tsm_all[ex][rname].item()['acfs'][kmonth]),0)
            ll = ax.plot(lags,plotvar,label=expnames_long[ex],c=ecols[ex],ls=els[ex],marker=emarkers[ex],zorder=1)
            
            if aa == 0:
                lines.append(ll)
                
            # Add Ensemble plots
            plotens  = np.array(tsm_all[ex][rname].item()['acfs'][kmonth])
            if plot_ens_indv:
                nrunplot = len(plotens)
                for nn in range(nrunplot):
                    plotvarens = plotens[nn,:]
                    ax.plot(lags,plotvarens,label="",c=ecols[ex],ls=els[ex],alpha=0.05,zorder=-3)
            else:
                mu      =  plotens.mean(0)
                sigma   =  plotens.std(0) 
                ax.fill_between(lags,mu-sigma,mu+sigma,color=ecols[ex],alpha=0.10,zorder=-9,label='_nolegend_')
        
        ax.set_title(regions_long[rr],fontsize=fsz_title)
    
    labs = [l[0].get_label() for l in lines]
    fig.legend(lines,labels=labs,ncols=3,fontsize=fsz_legend,bbox_to_anchor=(.83, 1.075,))

elif regionset == "TCMPi24" and TCM_ver:
    print("Doing TCM Version Plot")
    plotorder = [0,1,] # Set Order of plotting
    
    
    fig,axs   = plt.subplots(2,1,constrained_layout=True,figsize=(10,9),sharey=True)
    lines     = []
    for aa in range(2):
        
        ax    = axs.flatten()[aa]
        rr    = plotorder[aa]
        rname = regions[rr]
        
        ax,_   = viz.init_acplot(kmonth,xtksl,lags,title="",ax=ax,fsz_axis=fsz_axis,fsz_ticks=fsz_ticks)
        ax   = viz.add_ticks(ax=ax)
        
        
        # Adjust Axis Labels
        if aa == 0:
            ax.set_xlabel("")
        else:
            ax.set_xlabel("Lag (Months, Lag 0=%s)" % (mons3[kmonth]))

        ax.set_ylabel("Correlation (%s)" % (varname))
        
        for ex in range(nexps):
            plotvar = np.nanmean(np.array(tsm_all[ex][rname].item()['acfs'][kmonth]),0)
            ll = ax.plot(lags,plotvar,label=expnames_long[ex],c=ecols[ex],ls=els[ex],marker=emarkers[ex])
            
            if aa == 0:
                lines.append(ll)
            
            # Add Ensemble plots
            plotens  = np.array(tsm_all[ex][rname].item()['acfs'][kmonth])
            if plot_ens_indv:
                nrunplot = len(plotens)
                for nn in range(nrunplot):
                    plotvarens = plotens[nn,:]
                    ax.plot(lags,plotvarens,label="",c=ecols[ex],ls=els[ex],alpha=0.05,zorder=-3)
            else:
                mu      =  plotens.mean(0)
                sigma   =  plotens.std(0) 
                zz = ax.fill_between(lags,mu-sigma,mu+sigma,color=ecols[ex],alpha=0.10,zorder=-9,label='_nolegend_')
            

        #ax.set_title(regions_long[rr],fontsize=fsz_title)
        ax=viz.label_sp(regions_long[rr],ax=ax,x=0,y=.125,alpha=0.45,fig=fig,
                     labelstyle="%s",usenumber=True,fontsize=fsz_title)
    
    
    labs = [l[0].get_label() for l in lines]
    fig.legend(lines,labels=labs,ncols=3,fontsize=fsz_legend,bbox_to_anchor=(1.04, 1.075,))

    
#else:
elif regionset == "OSM24":
    plotorder = [0,1,] # Set Order of plotting
    
        
    fig,axs   = plt.subplots(2,1,constrained_layout=True,figsize=(10,9),sharey=True)
    lines     = []
    for aa in range(nregs):
        
        ax    = axs.flatten()[aa]
        rr    = plotorder[aa]
        rname = regions[rr]
        
        ax,_ = viz.init_acplot(kmonth,xtksl,lags,title="",ax=ax,fsz_axis=fsz_axis,fsz_ticks=fsz_ticks)
        ax   = viz.add_ticks(ax=ax)
        
        
        # Adjust Axis Labels
        if aa == 0:
            ax.set_xlabel("")
        else:
            ax.set_xlabel("Lag (Months, Lag 0=%s)" % (mons3[kmonth]))

        ax.set_ylabel("Correlation (%s)" % (varname))
        
        for ex in range(nexps):
            plotvar = np.nanmean(np.array(tsm_all[ex][rname].item()['acfs'][kmonth]),0)
            ll,_ = ax.plot(lags,plotvar,label=expnames_long[ex],c=ecols[ex],ls=els[ex],marker=emarkers[ex])
            
            if aa == 0:
                lines.append(ll)
            
            # Add Ensemble plots
            plotens  = np.array(tsm_all[ex][rname].item()['acfs'][kmonth])
            nrunplot = len(plotens)
            for nn in range(nrunplot):
                plotvarens = plotens[nn,:]
                ax.plot(lags,plotvarens,label="",c=ecols[ex],ls=els[ex],alpha=0.4)
            

        #ax.set_title(regions_long[rr],fontsize=fsz_title)
        ax=viz.label_sp(regions_long[rr],ax=ax,x=0,y=.125,alpha=0.45,fig=fig,
                     labelstyle="%s",usenumber=True,fontsize=fsz_title)
    
    labs = [l[0].get_label() for l in lines]
    fig.legend(lines,labels=labs,ncols=3,fontsize=fsz_legend,bbox_to_anchor=(1.04, 1.075,))
    
elif regionset == "SSSCSU" and TCM_ver:    
    
    plotorder = [2,3]#[0,1,] # Set Order of plotting
    
    
    fig,axs   = plt.subplots(2,1,constrained_layout=True,figsize=(10,9),sharey=True)
    lines     = []
    for aa in range(2):
        
        ax    = axs.flatten()[aa]
        rr    = plotorder[aa]
        rname = regions[rr]
        
        ax,_   = viz.init_acplot(kmonth,xtksl,lags,title="",ax=ax,fsz_axis=fsz_axis,fsz_ticks=fsz_ticks)
        ax   = viz.add_ticks(ax=ax)
        
        
        # Adjust Axis Labels
        if aa == 0:
            ax.set_xlabel("")
        else:
            ax.set_xlabel("Lag (Months, Lag 0=%s)" % (mons3[kmonth]))

        ax.set_ylabel("Correlation (%s)" % (varname))
        
        for ex in range(nexps):
            plotvar = np.nanmean(np.array(tsm_all[ex][rname].item()['acfs'][kmonth]),0)
            ll = ax.plot(lags,plotvar,label=expnames_long[ex],c=ecols[ex],ls=els[ex],marker=emarkers[ex])
            
            if aa == 0:
                lines.append(ll)
            
            # Add Ensemble plots
            plotens  = np.array(tsm_all[ex][rname].item()['acfs'][kmonth])
            if plot_ens_indv:
                nrunplot = len(plotens)
                for nn in range(nrunplot):
                    plotvarens = plotens[nn,:]
                    ax.plot(lags,plotvarens,label="",c=ecols[ex],ls=els[ex],alpha=0.05,zorder=-3)
            else:
                mu      =  plotens.mean(0)
                sigma   =  plotens.std(0) 
                zz = ax.fill_between(lags,mu-sigma,mu+sigma,color=ecols[ex],alpha=0.10,zorder=-9,label='_nolegend_')
            

        #ax.set_title(regions_long[rr],fontsize=fsz_title)
        ax=viz.label_sp(regions_long[rr],ax=ax,x=0,y=.125,alpha=0.45,fig=fig,
                     labelstyle="%s",usenumber=True,fontsize=fsz_title)
    
    
    labs = [l[0].get_label() for l in lines]
    if nexps == 4:
        fig.legend(lines,labels=labs,ncols=2,fontsize=fsz_legend,bbox_to_anchor=(.90, 1.12,))
    else:
        fig.legend(lines,labels=labs,ncols=3,fontsize=fsz_legend,bbox_to_anchor=(.95, 1.075,))
elif Draft01_ver and regionset == "SSSCSU": # 

    
    plotorder = [0,1,3]
    
    fig,axs   = plt.subplots(1,3,constrained_layout=True,figsize=(26,5),sharey=True)
    lines     = []
    if varname == "SST":
        ii = 0
    else:
        ii = 3
    for aa in range(3):
        
        ax    = axs.flatten()[aa]
        rr    = plotorder[aa]
        rname = regions[rr]
        
        ax,_ = viz.init_acplot(kmonth,xtksl,lags,title="",ax=ax,fsz_axis=fsz_axis,fsz_ticks=fsz_ticks)
        ax   = viz.add_ticks(ax=ax)
        
        
        
        # Adjust Axis Labels
        if aa != 1:
            ax.set_xlabel("")
        else:
            ax.set_xlabel("Lag (Months, Lag 0=%s)" % (mons3[kmonth]))
        if aa != 0:
            ax.set_ylabel("")
        else:
            ax.set_ylabel("Correlation (%s)" % (varname))
        
        for ex in range(nexps):
            plotvar = np.nanmean(np.array(tsm_all[ex][rname].item()['acfs'][kmonth]),0)
            ll = ax.plot(lags,plotvar,label=expnames_long[ex],c=ecols[ex],ls=els[ex],marker=emarkers[ex],zorder=1,lw=acf_lw)
            
            if aa == 0:
                lines.append(ll)
                
            # Add Ensemble plots
            plotens  = np.array(tsm_all[ex][rname].item()['acfs'][kmonth])
            if plot_ens_indv:
                nrunplot = len(plotens)
                for nn in range(nrunplot):
                    plotvarens = plotens[nn,:]
                    ax.plot(lags,plotvarens,label="",c=ecols[ex],ls=els[ex],alpha=0.05,zorder=-3)
            else:
                mu      =  plotens.mean(0)
                sigma   =  plotens.std(0) 
                ax.fill_between(lags,mu-sigma,mu+sigma,color=ecols[ex],alpha=0.10,zorder=-9,label='_nolegend_')
        
        ax.set_title(regions_long[rr],fontsize=fsz_title)
        
        viz.label_sp(ii,alpha=0.75,ax=ax,fontsize=fsz_title,y=1.1,x=-.09)
        ii+=1
    
    labs = [l[0].get_label() for l in lines]
    if varname == "SSS":
        fig.legend(lines,labels=labs,ncols=4,fontsize=fsz_legend,bbox_to_anchor=(.77, 1.15,))
    else:
        fig.legend(lines,labels=labs,ncols=3,fontsize=fsz_legend,bbox_to_anchor=(.67, 1.15,))

    
    
    
    
else:
    
    plotorder = [0,1,3,2] # Set Order of plotting
    
    
    fig,axs   = plt.subplots(2,2,constrained_layout=True,figsize=(16,8.5),sharey=True)
    lines     = []
    for aa in range(nregs):
        
        ax    = axs.flatten()[aa]
        rr    = plotorder[aa]
        rname = regions[rr]
        
        ax,_ = viz.init_acplot(kmonth,xtksl,lags,title="",ax=ax,fsz_axis=fsz_axis,fsz_ticks=fsz_ticks)
        ax   = viz.add_ticks(ax=ax)
        
        
        
        # Adjust Axis Labels
        if aa < 2:
            ax.set_xlabel("")
        else:
            ax.set_xlabel("Lag (Months, Lag 0=%s)" % (mons3[kmonth]))
        if aa % 2 != 0:
            ax.set_ylabel("")
        else:
            ax.set_ylabel("Correlation (%s)" % (varname))
        
        for ex in range(nexps):
            plotvar = np.nanmean(np.array(tsm_all[ex][rname].item()['acfs'][kmonth]),0)
            ll = ax.plot(lags,plotvar,label=expnames_long[ex],c=ecols[ex],ls=els[ex],marker=emarkers[ex],zorder=1,lw=acf_lw)
            
            if aa == 0:
                lines.append(ll)
                
            # Add Ensemble plots
            plotens  = np.array(tsm_all[ex][rname].item()['acfs'][kmonth])
            if plot_ens_indv:
                nrunplot = len(plotens)
                for nn in range(nrunplot):
                    plotvarens = plotens[nn,:]
                    ax.plot(lags,plotvarens,label="",c=ecols[ex],ls=els[ex],alpha=0.05,zorder=-3)
            else:
                mu      =  plotens.mean(0)
                sigma   =  plotens.std(0) 
                ax.fill_between(lags,mu-sigma,mu+sigma,color=ecols[ex],alpha=0.10,zorder=-9,label='_nolegend_')
        
        ax.set_title(regions_long[rr],fontsize=fsz_title)
    
    labs = [l[0].get_label() for l in lines]
    if varname == "SSS":
        fig.legend(lines,labels=labs,ncols=4,fontsize=fsz_legend,bbox_to_anchor=(.93, 1.10,))
    else:
        fig.legend(lines,labels=labs,ncols=3,fontsize=fsz_legend,bbox_to_anchor=(.83, 1.12,))
    

savename = "%sRegional_ACF_Comparison_%s_%s_tcmver%i_draftver%0i_mon%02i.png" % (figpath,comparename,regionset,TCM_ver,Draft01_ver,kmonth+1)
plt.savefig(savename,dpi=150,bbox_inches='tight',transparent=True)

# --------------------------------
#%% Plot 2: the monthly variance
# --------------------------------

if varname == "SST":
    vunit = r"\degree C"
    ylims = [0,0.2]
elif varname == "SSS":
    vunit = r"psu"
    ylims = [0,0.005]

if Draft01_ver:
    plotorder   = [0,1,3] # Set Order of plotting
    fig,axs     = viz.init_monplot(1,3,figsize=(26,5))
    
    lines       = []
    if varname == "SST":
        ii = 0
    else:
        ii = 3
    for aa in range(3):
        
        ax    = axs.flatten()[aa]
        rr    = plotorder[aa]
        rname = regions[rr]
        
        if aa == 1:
            ax.set_xlabel("Month",fontsize=fsz_axis)
        
        if aa > 0:
            ax.set_ylabel("")
        else:
            ax.set_ylabel("%s Variance [$%s^2$]" % (varname,vunit),fontsize=fsz_axis)
        
        for ex in range(nexps):
            plotvar    = np.nanmean(np.array(tsm_all[ex][rname].item()['monvars']),0)
            sstregvar  = rssts_all[ex][varname].isel(regions=rr).var('time').mean('run')
            
            if varname == 'SST':
                plotlab = "%s: ($\sigma^2$=%.2f $%s^2$)" % (expnames_short[ex],sstregvar,vunit)
                ncolvar = 3
            else:
                plotlab = "%s: ($\sigma^2$=%.4f $%s^2$)" % (expnames_short[ex],sstregvar,vunit)
                ncolvar = 2
                
            ll = ax.plot(mons3,plotvar,label=plotlab,c=ecols[ex],ls=els[ex],marker=emarkers[ex],lw=2.5)
            #lines.append(ll)
            
            
            # Add Ensemble plots
            plotens  = np.array(tsm_all[ex][rname].item()['monvars'])
            if plot_ens_indv:
                nrunplot = len(plotens)
                for nn in range(nrunplot):
                    plotvarens = plotens[nn,:]
                    ax.plot(mons3,plotvarens,label="",c=ecols[ex],ls=els[ex],alpha=0.05,zorder=-3,lw=2.5)
            else:
                mu      =  plotens.mean(0)
                sigma   =  plotens.std(0) 
                ax.fill_between(mons3,mu-sigma,mu+sigma,color=ecols[ex],alpha=0.10,zorder=-9,label='_nolegend_')
            
        
        
        ax.legend(ncol=ncolvar,fontsize=12)
        ax.tick_params(axis='both', which='major', labelsize=fsz_ticks)
        
        #ax.set_ylim(ylims)
        ax.set_title(regions_long[rr],fontsize=fsz_title)
        viz.label_sp(ii,alpha=0.75,ax=ax,fontsize=fsz_title,y=1.1,x=-.09)
        ii+=1
    
    
    
    
    
    
else:
        
    plotorder = [0,1,3,2] # Set Order of plotting
    
    #fig,axs = plt.subplots(2,2,constrained_layout=True,figsize=(10,6.5))
    fig,axs = viz.init_monplot(2,2,figsize=(12,8))
    lines = []
    for aa in range(nregs):
        
        ax    = axs.flatten()[aa]
        rr    = plotorder[aa]
        rname = regions[rr]
        
        if aa % 2 != 0:
            ax.set_ylabel("")
        else:
            ax.set_ylabel("%s Variance [$%s^2$]" % (varname,vunit),fontsize=fsz_axis)
        
        for ex in range(nexps):
            plotvar    = np.nanmean(np.array(tsm_all[ex][rname].item()['monvars']),0)
            sstregvar  = rssts_all[ex][varname].isel(regions=rr).var('time').mean('run')
            
            if varname == 'SST':
                plotlab = "%s: ($\sigma^2$=%.2f $%s^2$)" % (expnames_short[ex],sstregvar,vunit)
            else:
                plotlab = "%s: ($\sigma^2$=%.4f $%s^2$)" % (expnames_short[ex],sstregvar,vunit)
            
            
            ll = ax.plot(mons3,plotvar,label=plotlab,c=ecols[ex],ls=els[ex],marker=emarkers[ex])
            #lines.append(ll)
            
            
            # Add Ensemble plots
            plotens  = np.array(tsm_all[ex][rname].item()['monvars'])
            if plot_ens_indv:
                nrunplot = len(plotens)
                for nn in range(nrunplot):
                    plotvarens = plotens[nn,:]
                    ax.plot(mons3,plotvarens,label="",c=ecols[ex],ls=els[ex],alpha=0.05,zorder=-3)
            else:
                mu      =  plotens.mean(0)
                sigma   =  plotens.std(0) 
                ax.fill_between(mons3,mu-sigma,mu+sigma,color=ecols[ex],alpha=0.10,zorder=-9,label='_nolegend_')
            
            
            
        ax.legend(ncol=2,fontsize=8)
        
        #ax.set_ylim(ylims)
        ax.set_title(regions_long[rr],fontsize=fsz_title)
    
    
    
    #labs = [l[0].get_label() for l in lines]
    #fig.legend(lines,labels=labs,ncols=3,fontsize=fsz_legend,bbox_to_anchor=(1.03, 1.088,))
    
savename = "%sRegional_MonthlyVariance_Comparison_%s.png" % (figpath,comparename)
if Draft01_ver:
    savename = proc.addstrtoext(savename,"_Draft01",)
    
#savename = "%s%s_Regional_MonthlyVariance_Comparison.png" % (figpath,expname)
plt.savefig(savename,dpi=150,bbox_inches='tight',transparent=True)

# --------------------------------
#%% Plot 3. Spectra
# --------------------------------


# Plotting Params
dtplot  = 3600*24*30
plotyrs = [100,50,25,10,5]
xtks    = 1/(np.array(plotyrs)*12)



fig,axs = plt.subplots(2,2,constrained_layout=True,figsize=(10,6.5))

for aa in range(nregs):
    
    ax    = axs.flatten()[aa]
    rr    = plotorder[aa]
    rname = regions[rr]
    
    if aa > 1:
        ax.set_xlabel("Period")
    if aa % 2 == 0:
        
        ax.set_ylabel("Power ($%s^2$ / cpy)" % vunit)
    
    for ex in range(nexps):
        
        plotfreq   = np.nanmean(np.array(tsm_all[ex][rname].item()['freqs']),0) * dtplot
        plotspec   = np.nanmean(np.array(tsm_all[ex][rname].item()['specs']),0) / dtplot
        
        sstregvar  = rssts_all[ex][varname].isel(regions=rr).var('time').mean('run')
        
        if varname == "SST":
            plotlab = "%s (var=%.3f $%s^2$)" % (expnames_short[ex],sstregvar,vunit)
        else:
            plotlab = "%s (var=%.5f $%s^2$)" % (expnames_short[ex],sstregvar,vunit)
        
        # Plot Spectra
        ax.plot(plotfreq,plotspec,label=plotlab,c=ecols[ex],ls=els[ex],lw=2.5)
        
        # # Plot Confidence (this was fitted to whole sepctra, need to limit to lower frequencies)
        # plotCCs =  tsm_regs[rr]['CCs'][ii] /dtplot
        # ax.plot(plotfreq,plotCCs[:,1] ,c=cols[ii],lw=.75,ls='dotted')
        # ax.plot(plotfreq,plotCCs[:,0] ,c=cols[ii],lw=.55,ls='solid')
    
    ax.legend()
    
    ax.set_xticks(xtks,labels=plotyrs)
    ax.set_xlim([xtks[0],xtks[-1]])
    
    
    ax.set_title(regions_long[rr],fontsize=fsz_title)
    
savename = "%sRegional_Spectra_%s.png" % (figpath,comparename)
plt.savefig(savename,dpi=150,bbox_inches='tight')

# -----------------------------------------------------------------------------
#%% Plot 4: Pointwise T2
# -----------------------------------------------------------------------------

# Compute T2
acfs_in      = acfs_all.copy()
acfs_in[-1]  = acfs_in[-1].mean('ens') # Take the ensemble mean ACF for CESM
acfs_in      = proc.resize_ds(acfs_in)
acfs_val     = [ds.squeeze().transpose('lon','lat','mons','lags') for ds in acfs_in] # Transpose to same dimensions

[print(ds.dims) for ds in acfs_val]
acfs_val     = [ds.values for ds in acfs_val] # Convert to Numpy Array
t2_exp       = [proc.calc_T2(acf,axis=3) for acf in acfs_val] # [(65, 69, 12)] # Compute T2

# Compute Winter T2
t2_wint      = [t2[:,:,[11,0,1]].mean(-1) for t2 in t2_exp]

#%% Visualize Pointwise Differences in persistence
# Used similar formats from basinwide_T2_OSM



plot_lbde    = False # Set to True to compare with lbd_e (currently works only) with Paper Draft Sequence...
if varname == "SST":
    plot_lbde = False

t2_in        = t2_wint

lon          = acfs_all[0].lon
lat          = acfs_all[1].lat

vlm_reg      = [0, 20]
vlm_diff     = [-25,25]

# Set Color Limits/Intervals for T2 -----------
vlms_sst = {0:np.arange(0,19,1),
            1:np.arange(0,19,1),
            2:np.arange(-36,42,6),
            3:np.arange(-12,13,1)}
vlms_sss = {0:np.arange(0,88,8),
            1:np.arange(0,32,2),
            2:np.arange(-105,120,15),
            3:np.arange(-55,60,5)}
vlms = dict(SST=vlms_sst,SSS=vlms_sss)
# --------------------------------------------

fig,axs,mdict = viz.init_orthomap(1,4,bbplot,figsize=(22,10),centlat=45,)

for aa in range(4):
     
    ax = axs.flatten()[aa]
    ax = viz.add_coast_grid(ax,bbox=bbplot,fill_color="k",line_color="k")
    
    if aa == 0:
        if plot_lbde:
            
            plotvar = t2_in[0] # Plot Just Stochastic Model
        else:
            plotvar = t2_in[1] # Plot Just Stochastic Model
        title   = "Stochastic Model"
        cmap    = 'cmo.deep'
        cints   = vlms[varname][aa]#np.arange(0,19,1)
        #vlm     = [0,10]#vlm_reg
        
    elif aa == 1:
        plotvar = t2_in[-1]
        title   = "CESM1 (Ens. Average)"
        cmap    = 'cmo.dense'
        cints   = vlms[varname][aa]
        #vlm     = [0,18]#vlm_reg
    elif aa == 2:
        if plot_lbde:
            plotvar = t2_in[0] - t2_in[1]
            title   = "Effect of Adding SST-Evaporation Feedback \n($\lambda^e$ - No $\lambda^e$)"
        else:
            plotvar = t2_in[1] - t2_in[0]
            title   = "Effect of Adding Detrainment Damping\n(Detrainment Damping - No Detrainment Damping)"
            
        cmap    = 'cmo.balance'
        cints   = vlms[varname][aa]
        #vlm     = [-50,50]#vlm_diff
    elif aa == 3:
        if plot_lbde:
            plotvar = t2_in[0] - t2_in[-1]
        else:
            plotvar = t2_in[1] - t2_in[-1]
        title   = "Stochastic Model - CESM1"
        cmap    = 'cmo.balance'
        cints   = vlms[varname][aa]
        #vlm     = vlm_diff
        
    # Apply Mask
    plotvar = plotvar* dsmask.values.T
    
    print(plotvar.shape)
    #pcm = ax.pcolormesh(lon,lat,plotvar.T,transform=proj,zorder=-3,cmap=cmap,vmin=vlm[0],vmax=vlm[-1])
    pcm = ax.contourf(lon,lat,plotvar.T,transform=proj,levels=cints,zorder=-3,cmap=cmap,extend='both')
    cl  = ax.contour(lon,lat,plotvar.T,transform=proj,levels=cints,zorder=-3,colors="darkslategray",linewidths=0.55,
                     )
    ax.clabel(cl,levels=cints[::2],fontsize=fsz_ticks)
    cb = fig.colorbar(pcm,ax=ax,fraction=0.05,orientation="horizontal",pad=0.01)
    cb.set_label("%s Persistence Timescale $T^2$ (Months)" % varname,fontsize=fsz_axis-2)
    ax.set_title(title,fontsize=fsz_axis)

savename = "%sWintertime_Persistence_%s.png" % (figpath,comparename)
if plot_lbde:
    savename = proc.addstrtoext(savename,"_lbdE")
plt.savefig(savename,dpi=150,bbox_inches='tight')      


# --------------------
# %% Plot 4.5: Locator
# -------------------

# Load Re-emergence Pattern
bbplot2      = [-80,0,20,65]

fig,ax,mdict = viz.init_orthomap(1,1,bbplot2,figsize=(8,8),centlat=45,)
ax           = viz.add_coast_grid(ax,bbplot2,fill_color="k",line_color="k")

for rr in range(nregs):
    rbbx = bboxes[rr]
    viz.plot_box(rbbx,color=rcols[rr],linestyle=rsty[rr],leglab=regions_long[rr],linewidth=2.5,return_line=True)
#ax.legend(fontsize=fsz_axis,bbox_to_anchor=(0,1.1),ncol=2)

#viz.plot_box([-65,-40,40,47],color="k",linestyle="dashed",leglab="A",linewidth=4.5,return_line=True)
#viz.plot_box([-50,-20,20,30],color="k",linestyle="dashed",leglab="B",linewidth=4.5,return_line=True)
#viz.plot_box([-20,-10,30,50],color="k",linestyle="dashed",leglab="C",linewidth=4.5,return_line=True)

t2_in   = t2_wint
plotvar = t2_in[-1] * dsmask.T
cints   = vlms[varname][1]
cmap    = 'cmo.dense'
pcm     = ax.contourf(lon,lat,plotvar.T,transform=proj,levels=cints,zorder=-3,cmap=cmap,extend='both')
cl      = ax.contour(lon,lat,plotvar.T,transform=proj,levels=cints,zorder=-3,colors="darkslategray",linewidths=0.55,)

cb      = fig.colorbar(pcm,ax=ax,fraction=0.05,orientation="horizontal",pad=0.01)
cb.set_label("%s Persistence Timescale $T^2$ (Months)" % varname,fontsize=fsz_axis-2)
ax.set_title(title,fontsize=fsz_axis)

savename = "%sLocator_%s_%s.png" % (figpath,regionset,comparename)
plt.savefig(savename,dpi=150,bbox_inches='tight')  

# -----------------------------
#%% Plot 6. Pointwise Variance
# -----------------------------

plot_lbde    = True # Set to True to compare with lbd_e (currently works only) with Paper Draft Sequence...
if varname == "SST":
    plot_lbde = False

cmap_diff     = 'cmo.balance'
slvls         = np.arange(-150,160,15)
pmesh         = False

bbplot        = [-80,0,20,65]

if plot_lbde:
    var_sm        = var_all[0][varname]
else:
    var_sm        = var_all[1][varname]
    
var_cesm      = var_all[-1][varname]

# Initialize Figure
fig,axs,mdict = viz.init_orthomap(1,2,bbplot,figsize=(10,4.5))

for a in range(2):
    
    ax = axs[a]
    ax   = viz.add_coast_grid(ax,bbox=bbplot,fill_color='lightgray')
    
    if a == 0:
        
        pv     = (var_sm.mean('run') - var_cesm.mean('run')) * dsmask
        plon   = pv.lon
        plat   = pv.lat
        
        title  = "Diff. ($\sigma_{SM} - \sigma_{CESM}$)"
        
        if varname == 'SST':
            vlm    = [-.5,.5]
            vlvls  = np.arange(-.5,.55,0.05)
        elif varname == 'SSS':
            # if plot_lbde:
            #     vlm    = [-1.9,1.9]
            #     vlvls  = np.arange(-10.9,10.99,0.09)
            # else:
            vlm    = [-.3,.3]
            vlvls  = np.arange(-.3,.33,0.03)
            
            
    elif a == 1:
        pv     = np.log((var_sm.mean('run') / var_cesm.mean('run'))) * dsmask
        title  = "Log($\sigma_{SM}/\sigma_{CESM}$)"
        if varname == 'SST':
            vlm    = [-1,1]
            vlvls  = np.arange(-1,1.1,0.1)
        elif varname == 'SSS':
            vlm    = [-2.3,2.3]
            vlvls  = np.arange(-2.5,2.75,0.25)
    #pv     = pv[varname].values
    
    # Plot the values
    ax.set_title(title)
    if pmesh:
        pcm = ax.pcolormesh(plon,plat,pv,transform=proj,cmap=cmap_diff,vmin=vlm[0],vmax=vlm[1])
    else:
        pcm = ax.contourf(plon,plat,pv,transform=proj,cmap=cmap_diff,levels=vlvls)
    cb = fig.colorbar(pcm,ax=ax,orientation='horizontal',fraction=0.05,pad=0.01)
    cb.set_label(title)
    
    
    # # Plot contours    
    # current = dscurr[1].mean('mon').SSH
    # cl = ax.contour(long,latg,current,colors="k",
    #                 linewidths=0.35,transform=mdict['noProj'],levels=slvls,alpha=0.8)
    # ax.clabel(cl)
    
    # # Plot Mask
    # cl2 = ax.contour(ds_mask.lon,ds_mask.lat,plotmask,colors="w",linestyles='dashed',linewidths=.95,
    #                 levels=[0,1],transform=mdict['noProj'],zorder=1)

savename = "%s%s_Overall_Variance_Differences.png" % (figpath,comparename,)
if plot_lbde:
    savename = proc.addstrtoext(savename,"_lbdE")
plt.savefig(savename,dpi=150,bbox_inches='tight')

# ----------------------------
#%% Plot 5: Re-emergence Index
# ----------------------------
    
    
    






