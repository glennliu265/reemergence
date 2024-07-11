#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Scrap script to recalculate and visualize spectra for regionally averaged
SST/SSS.

Copied upper section from from compare_regional_metrics.py

Created on Thu Apr 11 10:30:02 2024

@author: gliu
"""

import sys
import copy
import glob
import time

from tqdm import tqdm

import numpy as np
import xarray as xr
import matplotlib as mpl
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
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

# Compare impact of adding lbd_e
regionset       = "TCMPi24"
comparename     = "lbde_comparison"
expnames        = ["SSS_EOF_LbddCorr_Rerun_lbdE","SSS_EOF_LbddCorr_Rerun","SSS_CESM"]
expnames_long   = ["Stochastic Model (with $\lambda^e$)","Stochastic Model","CESM1"]
expnames_short  = ["SM_lbde","SM","CESM"]
ecols           = ["forestgreen","goldenrod","k"]
els             = ["solid",'dashed','solid']
emarkers        = ["d","x","o"]

# CSU Comparisons (SSS)
regionset       = "SSSCSU"
comparename     = "lbde_comparison_CSU"
expnames        = ["SSS_EOF_LbddCorr_Rerun","SSS_EOF_LbddCorr_Rerun_NoLbdd","SSS_EOF_LbddCorr_Rerun_lbdE","SSS_CESM",]
expnames_long   = ["Stochastic Model","Stochastic Model (No $\lambda^d$)","Stochastic Model (with $\lambda^e$)","CESM1"]
expnames_short  = ["SM","SM_NoLbdd","SM_lbde","CESM"]
ecols           = ["forestgreen","goldenrod","magenta","k"]
els             = ["solid",'dashed','dotted','solid']
emarkers        = ["d","x",'+',"o"]

# CSU Comparisons (SST)
# regionset       = "SSSCSU"
# comparename     = "SST_comparison_CSU"
# expnames        = ["SST_EOF_LbddCorr_Rerun","SST_EOF_LbddCorr_Rerun_NoLbdd","SST_CESM"]
# expnames_long   = ["Stochastic Model","Stochastic Model (No $\lambda^d$)","CESM1"]
# expnames_short  = ["SM","SM_NoLbdd","CESM"]
# ecols           = ["forestgreen","goldenrod","k"]
# els             = ["solid",'dashed','solid']
# emarkers        = ["d","x","o"]


# # Check updates after switching detrainment and correting Fprime (SSS)
# regionset       = "TCMPi24"
# comparename     = "SSS_AprilUpdate"
# expnames        = ["SSS_EOF_Qek_LbddEnsMean","SSS_EOF_LbddCorr_Rerun","SSS_CESM"]
# expnames_long   = ["Stochastic Model (Exp Fit)","Stochastic Model (Corr.)","CESM1"]
# expnames_short  = ["SM_old","SM_new","CESM"]
# ecols           = ["forestgreen","goldenrod","k"]
# els             = ["solid",'dashed','solid']
# emarkers        = ["d","x","o"]

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
regionset       = "SSSCSU"
comparename     = "SSS_Paper_Draft01"
expnames        = ["SSS_EOF_LbddCorr_Rerun_lbdE_neg","SSS_EOF_LbddCorr_Rerun","SSS_EOF_LbddCorr_Rerun_NoLbdd","SSS_CESM"]
expnames_long   = ["Stochastic Model (sign corrected + $\lambda^e$)","Stochastic Model (with $\lambda^e$)","Stochastic Model","CESM1"]
expnames_short  = ["SM_lbde_neg","SM_lbde","SM","CESM"]
ecols           = ["magenta","forestgreen","goldenrod","k"]
els             = ['dotted',"solid",'dashed','solid']
emarkers        = ['+',"d","x","o"]


load_ravgparam  =True
regionset       ="SSSCSU"
TCM_ver         = True # Set to just plot 2 panels


Draft01_ver     = True

# Get Region Info
rdict                       = rparams.region_sets[regionset]
regions                     = rdict['regions']
bboxes                      = rdict['bboxes']
rcols                       = rdict['rcols']
rsty                        = rdict['rsty']
regions_long                = rdict['regions_long']
nregs                       = len(bboxes)


# Section between this copied from compare_regional_metrics ===================
#%% Load Regional Average SSTs and Metrics for the selected experiments

nexps = len(expnames)

seavar_all = []
var_all    = []
tsm_all   = []
rssts_all = []
acfs_all  = []
amv_all   = []
for e in range(nexps):
    
    # Get Experiment information
    expname        = expnames[e]
    
    if "SSS" in expname:
        varname = "SSS"
    elif "SST" in expname:
        varname = "SST"
    metrics_path    = output_path + expname + "/Metrics/"
    
    
    # Load Regionally Averaged SSTs
    ds = xr.open_dataset(metrics_path+"Regional_Averages_%s.nc" % regionset).load()
    rssts_all.append(ds)
    
    # # Load Regional Metrics
    ldz = np.load(metrics_path+"Regional_Averages_Metrics_%s.npz" % regionset,allow_pickle=True)
    tsm_all.append(ldz)
    
    # # Load Pointwise_ACFs
    # ds_acf = xr.open_dataset(metrics_path + "Pointwise_Autocorrelation_thresALL_lag00to60.nc")[varname].load()
    # acfs_all.append(ds_acf)  
    
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
regions                     = ds.regions.values
bboxes                      = ds.bboxes.values
rdict                       = rparams.region_sets[regionset]
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

# End Section copied from compare_regional_metrics ============================

# Begin section copied from 
#%% Make this into xarray ufunc eventually...

# Spectra Options
nsmooths = [20,20,20,20,5] # Choose different smoothing by experiment
pct      = 0.10
dtin     = 3600*24*365

specexp = []
for ex in range(nexps): 
    print(expnames[ex])
    nsmooth = nsmooths[ex]
    
    specreg = []
    for rr in tqdm(range(nregs)):
        
        rsst_in = rssts_all[ex].isel(regions=rr)[varname] # [Run x Time]
        nens    = len(rsst_in.run)
        
        # Take Annual Average
        rsst_ann = rsst_in.groupby('time.year').mean('time')
        
        # Copy Section From vizsualize_atmospheric_persistence --------
        tsens    = [rsst_ann.isel(run=e).values for e in range(nens)]
        specout = scm.quick_spectrum(tsens, nsmooth, pct, dt=dtin,make_arr=True,return_dict=True)
        specreg.append(specout)
    
    specexp.append(specreg)
        

#%%

def init_logspec(nrows,ncols,figsize=(10,4.5),ax=None,
                 xtks=None,dtplot=None,
                 fsz_axis=16,fsz_ticks=14,toplab=True,botlab=True):
    if dtplot is None:
        dtplot     = 3600*24*365  # Assume Annual data
    if xtks is None:
        xpers      = [100, 50,25, 20, 15,10, 5, 2]
        xtks       = np.array([1/(t) for t in xpers])
    
    if ax is None:
        newfig = True
        fig,ax = plt.subplots(nrows,ncols,constrained_layout=True,figsize=figsize)
    else:
        newfig = False
        
    ax = viz.add_ticks(ax)
    
    #ax.set_xticks(xtks,labels=xpers)
    ax.set_xscale('log')
    ax.set_xlim([xtks[0], xtks[-1]])
    if botlab:
        ax.set_xlabel("Frequency (Cycles/Year)",fontsize=fsz_axis)
    ax.tick_params(labelsize=fsz_ticks)
    
    ax2 = ax.twiny()
    ax2.set_xscale('log')
    ax2.set_xticks(xtks,labels=xpers,fontsize=fsz_ticks)
    ax2.set_xlim([xtks[0], xtks[-1]])
    if toplab:
        ax2.set_xlabel("Period (Years)",fontsize=fsz_axis)
    ax2.grid(True,ls='dotted',c="gray")
    
    if newfig:
        return fig,ax
    return ax
        



fig,ax=init_logspec(1,1,figsize=(10,4.5))
#%%
vunits = ["degC","psu"]



dtplot = dtin  
plot_regions = [0,1,3]


if Draft01_ver:
    fig,axs = plt.subplots(1,3,constrained_layout=True,figsize=(26,5))
    
    
    if varname == "SST":
        ii = 0
        vunit = vunits[0]
    else:
        ii = 3
        vunit = vunits[1]
        
    
    # Initialize the plot
    for rr in range(3):
        
        ir = plot_regions[rr]
        
        ax = axs.flatten()[rr]
        
        if rr < 2:
            toplab=True
            botlab=False
        if rr > 2:
            toplab=False
            botlab=True
        
        ax = init_logspec(1,1,ax=ax,toplab=toplab,botlab=botlab)
        ax.set_title(regions_long[ir],fontsize=22)
        
        # Plot for each experiment
        for ex in range(nexps):
            
            svarsin = specexp[ex][ir]
            
            P     = svarsin['specs']
            freq  = svarsin['freqs']
            
            cflab = "Red Noise"
            CCs   = svarsin['CCs']
            
            # Convert units
            freq     = freq[0, :] * dtplot
            P        = P / dtplot
            Cbase    = CCs.mean(0)[:, 0]/dtplot
            Cupbound = CCs.mean(0)[:, 1]/dtplot
            
            # Plot Ens Mean
            mu    = P.mean(0)
            sigma = P.std(0)
            
            # Plot Spectra
            ax.loglog(freq, mu, c=ecols[ex], lw=3,
                    label=expnames_long[ex], marker=emarkers[ex],markersize=1)
            
            # Plot Significance
            if ex ==0:
                labc1 = cflab
                labc2 = "95% Confidence"
            else:
                labc1=""
                labc2=""
            ax.plot(freq, Cbase, color=ecols[ex], ls='solid', lw=1.2, label=labc1)
            ax.plot(freq, Cupbound, color=ecols[ex], ls="dotted",
                    lw=2, label=labc2)
        if rr == 0:
            ax.legend(ncol=2,fontsize=12,)
            ax.set_ylabel(u"%s$^2 \, cpy^{-1}$" % vunit,fontsize=fsz_axis)
            
        if rr == 1:
            ax.set_xlabel("Frequency (Cycles/Year)",fontsize=fsz_axis)
            
        if varname == "SSS":
            ax.set_ylim([1e-4,1e-1])
        viz.label_sp(ii,alpha=0.75,ax=ax,fontsize=fsz_title,y=1.12,x=-.09)
        ii+=1
    
    
else:
    fig,axs = plt.subplots(2,2,constrained_layout=True,figsize=(18,8))
    
    
    # Initialize the plot
    for rr in range(nregs):
        
        ax = axs.flatten()[rr]
        
        if rr < 2:
            toplab=True
            botlab=False
        if rr > 2:
            toplab=False
            botlab=True
        
        ax = init_logspec(1,1,ax=ax,toplab=toplab,botlab=botlab)
        ax.set_title(regions_long[rr],fontsize=22)
        
        # Plot for each experiment
        for ex in range(nexps):
            
            svarsin = specexp[ex][rr]
            
            P     = svarsin['specs']
            freq  = svarsin['freqs']
            
            cflab = "Red Noise"
            CCs   = svarsin['CCs']
            
            # Convert units
            freq     = freq[0, :] * dtplot
            P        = P / dtplot
            Cbase    = CCs.mean(0)[:, 0]/dtplot
            Cupbound = CCs.mean(0)[:, 1]/dtplot
            
            # Plot Ens Mean
            mu    = P.mean(0)
            sigma = P.std(0)
            
            # Plot Spectra
            ax.loglog(freq, mu, c=ecols[ex], lw=2.5,
                    label=expnames_long[ex], marker=emarkers[ex],markersize=1)
            
            # Plot Significance
            if ex ==0:
                labc1 = cflab
                labc2 = "95% Confidence"
            else:
                labc1=""
                labc2=""
            ax.plot(freq, Cbase, color=ecols[ex], ls='solid', lw=1.2, label=labc1)
            ax.plot(freq, Cupbound, color=ecols[ex], ls="dotted",
                    lw=2, label=labc2)
        if rr == 0:
            ax.legend(ncol=2)
        
        
savename = "%s%s_Regional_Spectra_Differences.png" % (figpath,comparename,)
if Draft01_ver:
    savename = proc.addstrtoext(savename,"_Draft01")
plt.savefig(savename,dpi=150,bbox_inches='tight')   

#%% TCM_ver (2 regions, stacked Plot) 

dtplot = dtin  
fig,axs = plt.subplots(2,1,constrained_layout=True,figsize=(10,9))

nregs_plot = 2

# Initialize the plot
for rr in range(nregs_plot):
    
    ax = axs.flatten()[rr]
    
    if rr ==0:
        toplab=True
        botlab=False
    else:
        toplab=False
        botlab=True
    
    ax = init_logspec(1,1,ax=ax,toplab=toplab,botlab=botlab)
    ax.set_title(regions_long[rr],fontsize=22)
    
    # Plot for each experiment
    for ex in range(nexps):
        
        svarsin = specexp[ex][rr]
        
        P     = svarsin['specs']
        freq  = svarsin['freqs']
        
        cflab = "Red Noise"
        CCs   = svarsin['CCs']
        
        # Convert units
        freq     = freq[0, :] * dtplot
        P        = P / dtplot
        Cbase    = CCs.mean(0)[:, 0]/dtplot
        Cupbound = CCs.mean(0)[:, 1]/dtplot
        
        # Plot Ens Mean
        mu    = P.mean(0)
        sigma = P.std(0)
        
        # Plot Spectra
        ax.loglog(freq, mu, c=ecols[ex], lw=2.5,
                label=expnames_long[ex], marker=emarkers[ex],markersize=1)
        
        # Plot Significance
        if ex ==0:
            labc1 = cflab
            labc2 = "95% Confidence"
        else:
            labc1=""
            labc2=""
        ax.plot(freq, Cbase, color=ecols[ex], ls='solid', lw=1.2, label=labc1)
        ax.plot(freq, Cupbound, color=ecols[ex], ls="dotted",
                lw=2, label=labc2)
    if rr == 0:
        ax.legend(ncol=2)
        
    if comparename ==  "lbde_comparison_CSU":
        ax.set_ylim([1e-4,1e-1])
        vunit = "psu"
    else:
        vunit = "$\degree$C"
    
    ax.set_ylabel("Power (%s$^2$/cpy)" % vunit,fontsize=fsz_axis)
    
    
savename = "%s%s_Regional_Spectra_Differences.png" % (figpath,comparename,)
plt.savefig(savename,dpi=150,bbox_inches='tight',transparent=True)   

#%% New Section (Compare with Regionally Averaged Parameter Runs)

nsmooth   = 20  # Check to make sure it is the same as above!!
pct       = 0.10
dtin      = 3600*24*365

# Load NetCDF
iexp      = 1 # Select which experiment to compare
expdir    = "%s%s/" % (output_path,expnames[iexp])
rparamnc  = "%sOutput/%s_%s_ravgparams.nc" % (expdir,varname,regionset)
ds_rparam = xr.open_dataset(rparamnc)[varname].load() # (run, time, region)

# Loop by Region
specreg_rparam = []
acf_rparam     = []
for rr in tqdm(range(nregs)):
    
    # Get number of ensemble members
    rsst_in  = ds_rparam.isel(region=rr) # (run, time)
    nens     = len(rsst_in.run)
    
    # Take Annual Average
    rsst_ann = rsst_in.groupby('time.year').mean('time')
    
    # Copy Section From vizsualize_atmospheric_persistence --------
    tsens    = [rsst_ann.isel(run=e).values for e in range(nens)]
    specout  = scm.quick_spectrum(tsens, nsmooth, pct, dt=dtin,make_arr=True,return_dict=True)
    specreg_rparam.append(specout)
    
    

specexp.append(specreg)


#%% Try to do pointwise computation of lags
# (This could be a test case of how to do it...)

lags   = np.arange(37)
infunc = lambda a: scm.calc_autocorr_mon(a,lags)
st = time.time()
acf_rparam = acfreg_rparam = xr.apply_ufunc(
    infunc,  # Pass the function
    ds_rparam,  # The inputs in order that is expected
    # Which dimensions to operate over for each argument...
    input_core_dims =[['time']],
    output_core_dims=[['mon','lag'],],  # Output Dimension
    exclude_dims    =set(("mon","lag")),
    vectorize       =True,  # True to loop over non-core dims
)

acf_rparam['mon'] = np.arange(1,13,1) #  Fix Dimensions
acf_rparam['lag'] = lags
print("Complete in %.2fs" % (time.time()-st))

#%% Compare the outputs

# Select Lag Basemonth
kmonth = 1
xtks   = np.arange(0,37,3)

# Select Region
rr     = 0
for rr in range(len(regions)):
    rname  = regions[rr]
    title  = "%s, %s ACF (lag 0 = %s)" % (rname,varname,mons3[kmonth],)
    
    # Make the figure
    fig,ax = plt.subplots(1,1,constrained_layout=True,figsize=(8,4))
    ax,_ = viz.init_acplot(kmonth,xtks,lags,ax=ax,title="")
    ax.set_title(title,fontsize=fsz_title)
    
    # Plot Regionally Averaged Parameters
    plotvar_rparam = acf_rparam.isel(mon=kmonth,region=rr) # [Runs x Lags]
    nruns1 = len(plotvar_rparam.run)
    for nn in range(nruns1):
        ax.plot(lags,plotvar_rparam.isel(run=nn),alpha=0.1,c='limegreen',label="",zorder=-2)
    ax.plot(lags,plotvar_rparam.mean('run'),alpha=1,c='limegreen',label="Regionally Averaged Parameters")  
    #ax.legend()
    
    # Plot Regionally Averaged SST
    plotvar = tsm_all[iexp][rname].item()['acfs'][kmonth] # Months
    nruns = len(plotvar)
    for nn in range(nruns):
        ax.plot(lags,plotvar[nn],alpha=0.1,c='navy',label="",zorder=-2)
    ax.plot(lags,np.array(plotvar).mean(0),alpha=1,c='navy',label="Regionally Averaged SST")  
    ax.legend()
    
    
    # Plot CESM1
    plotvar = tsm_all[-1][rname].item()['acfs'][kmonth] # Months
    nruns   = len(plotvar)
    for nn in range(nruns):
        ax.plot(lags,plotvar[nn],alpha=0.05,c='k',label="",zorder=-2)
    ax.plot(lags,np.array(plotvar).mean(0),alpha=1,c='k',label="CESM1")  
    ax.legend()
    # Plot Regionally Averaged ACF
    #
    
    savename = "%s%s_%s_%sACF_mon%02i.png" % (figpath,comparename,rname,varname,kmonth+1)
    plt.savefig(savename,dpi=150,bbox_inches='tight')
    
#%%
